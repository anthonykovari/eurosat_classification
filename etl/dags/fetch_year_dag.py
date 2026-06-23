"""
Fetch one year of Sentinel-2 imagery into S3.

Two-task pipeline:
  fetch_tiles   — calls SentinelHub API for each tile, saves each to S3 staging
  stitch_upload — downloads tiles, stitches mosaic, uploads imagery/{year}/rgb.npy

Trigger:
  airflow dags trigger chicago_fetch_year --conf '{"year": 2019}'
"""

from __future__ import annotations

import io
import logging
import math
import os
import tempfile

import boto3
import numpy as np
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.utils.dates import days_ago
from botocore.exceptions import ClientError
from datetime import timedelta

log = logging.getLogger(__name__)

S3_BUCKET = Variable.get("sprawl_s3_bucket", default_var="chicago-land-use")
AOI_BBOX  = (-88.4, 41.45, -87.5, 42.75)

EVALSCRIPT = """
//VERSION=3
// Single best scene per tile — catalog pre-selects the least-cloudy date so
// the time_interval passed here is always a 1-day window containing exactly
// one Sentinel-2 pass. ORBIT mosaicking is used because SIMPLE silently
// returns null samples on CDSE even for valid scenes.
// SCL is requested alongside RGB so fetch_best_scene can check per-tile
// cloud fraction instead of relying solely on scene-level eo:cloud_cover.
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "SCL"] }],
    output: [
      { id: "rgb", bands: 3, sampleType: "UINT8" },
      { id: "scl", bands: 1, sampleType: "UINT8" }
    ],
    mosaicking: "ORBIT"
  };
}
function evaluatePixel(samples) {
  if (!samples || samples.length === 0) return { rgb: [0,0,0], scl: [0] };
  var s = samples[0];
  if (!s) return { rgb: [0,0,0], scl: [0] };
  var f = 255.0 / 0.3;
  return {
    rgb: [
      Math.min(255, Math.round(s.B04 * f)),
      Math.min(255, Math.round(s.B03 * f)),
      Math.min(255, Math.round(s.B02 * f)),
    ],
    scl: [s.SCL]
  };
}
"""

# SCL pixel classes treated as unusable: cloud shadow, cloud medium/high, thin cirrus.
_CLOUD_SCL_CLASSES = [3, 8, 9, 10]

with DAG(
    dag_id="chicago_fetch_year",
    description="Fetch one year of Sentinel-2 imagery (trigger with conf={'year': YYYY})",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    default_args={"owner": "ml-team", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["etl", "sentinel2", "chicago", "cdse"],
) as dag:

    @task()
    def fetch_tiles(**context) -> dict:
        """
        Fetch Sentinel-2 tiles for one year.

        Normal mode  (no mosaic in S3): fetch all tiles, upload staging, hand off to stitch_upload.
        Patch mode   (mosaic exists):   scan each tile's pixel region for zeros; re-fetch only the
                                        bad ones from CDSE, patch them in-place, re-upload mosaic.
                                        Skips stitch_upload (returns skipped=True).

        Conf options:
          year       — int, required
          max_tiles  — int, limit for testing (normal mode only)
          patch_zeros — bool (default true), scan existing mosaic for bad tiles on re-trigger
        """
        import datetime as dt
        from sentinelhub import (
            BBox, CRS, DataCollection, MimeType,
            SHConfig, SentinelHubCatalog, SentinelHubRequest, bbox_to_dimensions,
        )

        run_conf  = context.get("dag_run").conf or {}
        year      = int(run_conf.get("year", __import__("datetime").date.today().year))
        max_tiles = int(run_conf.get("max_tiles", 0)) or None
        patch_zeros     = str(run_conf.get("patch_zeros", "true")).lower() != "false"
        zero_threshold  = float(run_conf.get("zero_threshold", 0.02))  # re-fetch tiles above this (patch mode)
        max_cloud_pct   = float(run_conf.get("max_cloud_pct", 0.10))   # per-tile SCL cloud fraction ceiling

        s3 = boto3.client("s3")
        final_key = f"imagery/{year}/rgb.npy"

        # ── Grid geometry (same for both modes) ──────────────────────────────
        MAX_PX = 2500
        lon_min, lat_min, lon_max, lat_max = AOI_BBOX
        full_bbox = BBox(bbox=AOI_BBOX, crs=CRS.WGS84)
        full_w, full_h = bbox_to_dimensions(full_bbox, resolution=10)
        n_cols = math.ceil(full_w / MAX_PX)
        n_rows = math.ceil(full_h / MAX_PX)
        col_step = (lon_max - lon_min) / n_cols
        row_step = (lat_max - lat_min) / n_rows

        def make_tile_bbox(r: int, c: int) -> "BBox":
            return BBox(
                bbox=(
                    lon_min + c * col_step, lat_min + r * row_step,
                    lon_min + (c + 1) * col_step, lat_min + (r + 1) * row_step,
                ),
                crs=CRS.WGS84,
            )

        # ── SentinelHub clients (initialised lazily so patch-mode skips if unneeded) ──
        _sh_config = None
        _S2_L2A   = None
        _catalog  = None

        def sh_clients():
            nonlocal _sh_config, _S2_L2A, _catalog
            if _sh_config is None:
                _sh_config = SHConfig(
                    sh_client_id=os.environ["CDSE_CLIENT_ID"],
                    sh_client_secret=os.environ["CDSE_CLIENT_SECRET"],
                    sh_base_url="https://sh.dataspace.copernicus.eu",
                    sh_token_url=os.environ.get(
                        "CDSE_TOKEN_URL",
                        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                    ),
                    download_timeout_seconds=300,
                )
                _S2_L2A  = DataCollection.SENTINEL2_L2A.define_from(
                    "s2l2a_cdse", service_url="https://sh.dataspace.copernicus.eu"
                )
                _catalog = SentinelHubCatalog(config=_sh_config)
            return _sh_config, _S2_L2A, _catalog

        def fetch_best_scene(
            fetch_bbox: "BBox",
            req_w: int,
            req_h: int,
            max_cloud_pct: float,
            label: str = "?",
        ) -> np.ndarray:
            """Fetch the best available scene for fetch_bbox at (req_w × req_h) pixels.

            Quality gates (either triggers a retry):
              - zero_pct  > 2%           — NoData / swath edge black pixels
              - cloud_pct > max_cloud_pct — SCL-measured cloud/shadow pixels

            Scene-level eo:cloud_cover is used only for ranking candidates; SCL gives
            the actual per-pixel quality measurement over the requested footprint.
            Accepts any bbox, so patch mode can pass a small sub-region.
            """
            sh_config, S2_L2A, catalog = sh_clients()

            scenes = list(catalog.search(
                S2_L2A, bbox=fetch_bbox,
                time=(f"{year}-06-01", f"{year}-09-30"),
                fields={"include": ["properties.datetime", "properties.eo:cloud_cover"]},
            ))
            if not scenes:
                log.warning("Region %s: no scenes — returning zeros", label)
                return np.zeros((req_h, req_w, 3), dtype=np.uint8)

            MAX_ZERO_PCT = 0.02
            MAX_RETRIES  = 5
            scenes_sorted = sorted(scenes, key=lambda s: s["properties"].get("eo:cloud_cover", 100))
            result = None
            for attempt, scene in enumerate(scenes_sorted[:MAX_RETRIES]):
                scene_date = dt.date.fromisoformat(scene["properties"]["datetime"][:10])
                t_start = str(scene_date)
                t_end   = str(scene_date + dt.timedelta(days=1))
                log.info("Region %s attempt %d: %s  cc=%.0f%%  (%d candidates)",
                         label, attempt + 1, t_start,
                         scene["properties"].get("eo:cloud_cover", 0), len(scenes))
                sh_req = SentinelHubRequest(
                    evalscript=EVALSCRIPT,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=S2_L2A, time_interval=(t_start, t_end), maxcc=1.0,
                    )],
                    responses=[
                        SentinelHubRequest.output_response("rgb", MimeType.TIFF),
                        SentinelHubRequest.output_response("scl", MimeType.TIFF),
                    ],
                    bbox=fetch_bbox, size=(req_w, req_h), config=sh_config,
                )
                data = sh_req.get_data(redownload=True)
                if not data or data[0] is None:
                    log.warning("Region %s attempt %d: no data", label, attempt + 1)
                    continue

                item = data[0]
                if isinstance(item, dict):
                    # sentinelhub-py includes the file extension in dict keys ("rgb.tif")
                    keyed = {k.split(".")[0]: v for k, v in item.items()}
                    rgb_arr = keyed["rgb"].astype(np.uint8)
                    scl_raw = keyed.get("scl")
                else:
                    rgb_arr = item.astype(np.uint8)
                    scl_raw = None

                zero_pct  = float((rgb_arr.sum(axis=2) == 0).mean())
                cloud_pct = 0.0
                if scl_raw is not None:
                    cloud_pct = float(np.isin(scl_raw.ravel(), _CLOUD_SCL_CLASSES).mean())

                log.info("Region %s attempt %d: zero=%.1f%%  cloud=%.1f%%",
                         label, attempt + 1, zero_pct * 100, cloud_pct * 100)

                is_clean = zero_pct <= MAX_ZERO_PCT and cloud_pct <= max_cloud_pct
                if is_clean or attempt == MAX_RETRIES - 1:
                    if not is_clean:
                        log.warning("Region %s: no clean scene after %d attempts "
                                    "(best: zero=%.1f%%  cloud=%.1f%%)",
                                    label, MAX_RETRIES, zero_pct * 100, cloud_pct * 100)
                    result = rgb_arr
                    break
                log.warning("Region %s: zero=%.1f%%  cloud=%.1f%% — trying next scene",
                            label, zero_pct * 100, cloud_pct * 100)

            return result if result is not None else np.zeros((req_h, req_w, 3), dtype=np.uint8)

        # ── PATCH MODE — mosaic already exists, fix only bad tiles ───────────
        mosaic_exists = False
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=final_key, Range="bytes=0-3")
            if int(resp["ContentRange"].split("/")[-1]) >= 1_000_000:
                mosaic_exists = True
        except ClientError:
            pass

        if mosaic_exists and patch_zeros:
            log.info("Year %d mosaic already in S3 — scanning for bad tiles (patch mode)", year)
            rgb = np.load(io.BytesIO(
                s3.get_object(Bucket=S3_BUCKET, Key=final_key)["Body"].read()
            ))

            # Recompute each tile's pixel dimensions from geography (deterministic).
            tile_dims: dict[tuple[int, int], tuple[int, int]] = {}
            for r in range(n_rows):
                for c in range(n_cols):
                    tw, th = bbox_to_dimensions(make_tile_bbox(r, c), resolution=10)
                    tile_dims[(r, c)] = (th, tw)  # (height, width)

            # Reconstruct mosaic pixel bounds for each tile, mirroring stitch_upload logic:
            # strips are built row-by-row (r=0..n_rows-1), then reversed vertically.
            row_min_h = {r: min(tile_dims[(r, c)][0] for c in range(n_cols)) for r in range(n_rows)}
            strip_widths = {r: sum(tile_dims[(r, c)][1] for c in range(n_cols)) for r in range(n_rows)}
            min_w = min(strip_widths.values())

            # Mosaic top → r=n_rows-1, bottom → r=0 (strips[::-1] in stitch_upload).
            tile_bounds: dict[tuple[int, int], tuple[int, int, int, int]] = {}
            y = 0
            for r in range(n_rows - 1, -1, -1):
                x = 0
                for c in range(n_cols):
                    th, tw = tile_dims[(r, c)]
                    y1 = y + row_min_h[r]
                    x1 = min(x + tw, min_w)
                    tile_bounds[(r, c)] = (y, y1, x, x1)
                    x += tw
                y += row_min_h[r]

            BAD_ZERO_PCT = zero_threshold
            bad_tiles = []
            for r in range(n_rows):
                for c in range(n_cols):
                    y0, y1, x0, x1 = tile_bounds[(r, c)]
                    region = rgb[y0:y1, x0:x1]
                    zero_pct = float((region.sum(axis=2) == 0).mean())
                    if zero_pct > BAD_ZERO_PCT:
                        bad_tiles.append((r, c, zero_pct))
                        log.info("Bad tile (%d,%d): %.1f%% zeros → queued for re-fetch", r, c, zero_pct * 100)

            if not bad_tiles:
                log.info("Year %d: no bad tiles found — mosaic is clean", year)
                return {"year": year, "tiles": [], "skipped": True}

            log.info("Year %d: filling zero sub-regions in %d tile(s)", year, len(bad_tiles))
            for r, c, _ in bad_tiles:
                y0_m, y1_m, x0_m, x1_m = tile_bounds[(r, c)]
                tile_region = rgb[y0_m:y1_m, x0_m:x1_m]
                zero_mask = (tile_region.sum(axis=2) == 0)

                # Tight bounding box of all zero pixels within this tile's mosaic region.
                zero_rows = np.where(zero_mask.any(axis=1))[0]
                zero_cols = np.where(zero_mask.any(axis=0))[0]
                if len(zero_rows) == 0:
                    continue

                zy0, zy1 = int(zero_rows[0]), int(zero_rows[-1]) + 1
                zx0, zx1 = int(zero_cols[0]), int(zero_cols[-1]) + 1
                patch_h = zy1 - zy0
                patch_w = zx1 - zx0

                # Map zero-pixel bounds to geographic coordinates.
                # The mosaic tile region covers (th_crop × tw_crop) pixels which correspond
                # to a fraction of the tile's full geographic extent.
                full_th, full_tw = tile_dims[(r, c)]
                th_crop = y1_m - y0_m
                tw_crop = x1_m - x0_m
                tile_lon0 = lon_min + c * col_step
                tile_lat1 = lat_min + (r + 1) * row_step   # north edge of tile
                lon_span  = (tw_crop / full_tw) * col_step
                lat_span  = (th_crop / full_th) * row_step

                sub_lon0 = tile_lon0 + (zx0 / tw_crop) * lon_span
                sub_lon1 = tile_lon0 + (zx1 / tw_crop) * lon_span
                sub_lat1 = tile_lat1 - (zy0 / th_crop) * lat_span  # y=0 → north
                sub_lat0 = tile_lat1 - (zy1 / th_crop) * lat_span  # y=th → south

                sub_bbox = BBox(bbox=(sub_lon0, sub_lat0, sub_lon1, sub_lat1), crs=CRS.WGS84)
                log.info(
                    "Tile (%d,%d): zero sub-region %d×%d px  %.4f°×%.4f° → targeted fetch",
                    r, c, patch_w, patch_h, sub_lon1 - sub_lon0, sub_lat1 - sub_lat0,
                )

                patch_arr = fetch_best_scene(
                    sub_bbox, patch_w, patch_h, max_cloud_pct, label=f"({r},{c})-fill",
                )

                # Overlay: write only where the mosaic is still zero.
                region = tile_region[zy0:zy1, zx0:zx1]
                still_zero = (region.sum(axis=2) == 0)
                region[still_zero] = patch_arr[:patch_h, :patch_w][still_zero]
                rgb[y0_m + zy0:y0_m + zy1, x0_m + zx0:x0_m + zx1] = region
                log.info("Tile (%d,%d): filled %d zero pixels", r, c, int(still_zero.sum()))

            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                np.save(tmp, rgb)
                tmp_path = tmp.name
            try:
                s3.upload_file(tmp_path, S3_BUCKET, final_key)
            finally:
                os.unlink(tmp_path)
            log.info("Year %d patched mosaic re-uploaded (%d tiles fixed)", year, len(bad_tiles))
            # Bust the cached satellite PNG so the backend re-generates it.
            try:
                s3.delete_object(Bucket=S3_BUCKET, Key=f"viz/{year}/satellite.png")
            except ClientError:
                pass
            return {"year": year, "tiles": [], "skipped": True}

        elif mosaic_exists and not patch_zeros:
            log.info("Year %d already in S3 — skipping (pass patch_zeros=true to fix bad tiles)", year)
            return {"year": year, "tiles": [], "skipped": True}

        # ── NORMAL MODE — full fetch ──────────────────────────────────────────
        log.info("AOI %d×%d px → %d cols × %d rows (%d tiles)", full_w, full_h, n_cols, n_rows, n_cols * n_rows)

        fetched: list[list[int]] = []
        for r in range(n_rows):
            for c in range(n_cols):
                if max_tiles is not None and len(fetched) >= max_tiles:
                    log.info("max_tiles=%d reached — stopping early", max_tiles)
                    return {"year": year, "tiles": fetched, "skipped": False}

                tile_key = f"imagery/{year}/tiles/r{r}_c{c}.npy"
                try:
                    s3.get_object(Bucket=S3_BUCKET, Key=tile_key, Range="bytes=0-3")
                    log.info("Tile (%d,%d) already in S3 — skipping", r, c)
                    fetched.append([r, c])
                    continue
                except ClientError:
                    pass

                tile_bbox = make_tile_bbox(r, c)
                tile_w, tile_h = bbox_to_dimensions(tile_bbox, resolution=10)
                arr = fetch_best_scene(tile_bbox, tile_w, tile_h, max_cloud_pct, label=f"({r},{c})")
                buf = io.BytesIO()
                np.save(buf, arr)
                buf.seek(0)
                s3.upload_fileobj(buf, S3_BUCKET, tile_key)
                fetched.append([r, c])
                log.info("Tile (%d,%d) ✓  shape=%s  %.1f MB", r, c, arr.shape, buf.tell() / 1e6)

        return {"year": year, "tiles": fetched, "skipped": False}

    @task()
    def stitch_upload(tile_meta: dict) -> dict:
        """
        Download staged tiles from S3, stitch into a mosaic, upload imagery/{year}/rgb.npy.
        Works from the explicit tile list returned by fetch_tiles, so partial runs
        (max_tiles) and full runs use the same code path.
        Cleans up staging tiles on success.
        """
        if tile_meta.get("skipped"):
            log.info("Year %d already present — nothing to stitch", tile_meta["year"])
            return tile_meta

        year  = tile_meta["year"]
        tiles = tile_meta["tiles"]  # [[r, c], ...]

        if not tiles:
            log.warning("No tiles to stitch for year %d", year)
            return {"year": year, "skipped": True}

        s3 = boto3.client("s3")

        # Group by row, preserving column order within each row.
        from collections import defaultdict
        rows: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for r, c in tiles:
            tile_key = f"imagery/{year}/tiles/r{r}_c{c}.npy"
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=tile_key)
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    raise RuntimeError(
                        f"Staging tile {tile_key} is missing — re-run fetch_tiles"
                    ) from e
                raise
            arr = np.load(io.BytesIO(obj["Body"].read()))
            rows[r].append((c, arr))

        log.info("Stitching %d tile(s) across %d row(s) for year %d", len(tiles), len(rows), year)

        strips: list[np.ndarray] = []
        for r in sorted(rows):
            row_tiles = [arr for _, arr in sorted(rows[r])]
            min_h = min(t.shape[0] for t in row_tiles)
            strips.append(np.concatenate([t[:min_h] for t in row_tiles], axis=1))

        min_w = min(s.shape[1] for s in strips)
        # Reverse so row-0 (southernmost) ends up at the bottom of the image.
        rgb = np.concatenate([s[:, :min_w] for s in strips[::-1]], axis=0)
        log.info("Mosaic shape=%s", rgb.shape)

        final_key = f"imagery/{year}/rgb.npy"
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp, rgb)
            tmp_path = tmp.name
        size_mb = os.path.getsize(tmp_path) / 1e6
        try:
            s3.upload_file(tmp_path, S3_BUCKET, final_key)
        finally:
            os.unlink(tmp_path)
        log.info("Year %d uploaded  %.1f MB → %s", year, size_mb, final_key)

        for r, c in tiles:
            s3.delete_object(Bucket=S3_BUCKET, Key=f"imagery/{year}/tiles/r{r}_c{c}.npy")
        log.info("Staging tiles deleted")

        return {"year": year, "key": final_key, "shape": list(rgb.shape)}

    stitch_upload(fetch_tiles())
