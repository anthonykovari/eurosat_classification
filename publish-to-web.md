# Publish to Web — Static Export Plan

## Goal

Publish the Chicagoland Land Use Change viewer as a static site embedded in
`anthonykovari.com` (GitHub Pages). No live backend — all data pre-baked as
static files. One-time effort; update manually if new years are added.

## Prerequisites

- Full stack running: `docker compose up -d` from project root
- All years 2019–2026 confirmed in MinIO (imagery + classifications)
- Personal site repo checked out at `~/Documents/anthonykovari.com`


## Step 1 — Export static assets

Run the export script (to be written) while the stack is live. It hits the
backend at `localhost:8000` and writes files into the personal site repo.

**Script to create:** `scripts/export_static.py`

```
python3 scripts/export_static.py \
  --years 2019 2020 2021 2022 2023 2024 2025 2026 \
  --out ~/Documents/anthonykovari.com/static/eurosat
```

**What the script does for each year:**
1. `GET /satellite/{year}.png`  → `static/eurosat/satellite/{year}.png`
2. `GET /overlay/{year}.png`    → `static/eurosat/overlay/{year}.png`
3. `GET /grid/{year}`           → `static/eurosat/grid/{year}.json`

**Plus these one-time fetches:**
4. `GET /available`             → `static/eurosat/available.json`
5. `GET /sprawl`                → `static/eurosat/sprawl.json`
6. `GET /timeseries`            → `static/eurosat/timeseries.json`

The backend already caches satellite PNGs in MinIO (`viz/{year}/satellite.png`)
so the first call per year may be slow; subsequent calls are instant.


## Step 2 — Write the export script

Create `scripts/export_static.py`. Key logic:

```python
import argparse, json, pathlib, requests

API = "http://localhost:8000"

def export(years, out_dir):
    out = pathlib.Path(out_dir)
    (out / "satellite").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    (out / "grid").mkdir(parents=True, exist_ok=True)

    # one-time JSON endpoints
    for name, path in [("available", "/available"), ("sprawl", "/sprawl"), ("timeseries", "/timeseries")]:
        r = requests.get(f"{API}{path}")
        if r.ok:
            (out / f"{name}.json").write_bytes(r.content)

    for year in years:
        print(f"Exporting {year}...")
        for kind, url, dest in [
            ("satellite", f"/satellite/{year}.png",  out / "satellite" / f"{year}.png"),
            ("overlay",   f"/overlay/{year}.png",    out / "overlay"   / f"{year}.png"),
            ("grid",      f"/grid/{year}",            out / "grid"      / f"{year}.json"),
        ]:
            r = requests.get(f"{API}{url}")
            if r.ok:
                dest.write_bytes(r.content)
            else:
                print(f"  WARNING: {kind} {year} returned {r.status_code}")
```


## Step 3 — Adapt the frontend

Copy `frontend/index.html` to `anthonykovari.com/eurosat.html` (or a subdir).

**Key change — replace the API constant and all fetch calls:**

```js
// BEFORE
const API = 'http://localhost:8000';
// ...
L.imageOverlay(`${API}/satellite/${year}.png`, AOI)
fetch(`${API}/available`)
fetch(`${API}/grid/${year}`)
fetch(`${API}/sprawl`)

// AFTER — relative paths into static/eurosat/
const STATIC = '/static/eurosat';
// ...
L.imageOverlay(`${STATIC}/satellite/${year}.png`, AOI)
fetch(`${STATIC}/available.json`)
fetch(`${STATIC}/grid/${year}.json`)
fetch(`${STATIC}/sprawl.json`)
```

**Remove or stub out:**
- `/predict/` upload endpoint (not needed for portfolio view)
- Any polling/retry logic that depends on a live backend

**Add to `anthonykovari.com` nav:** link from `satellite-classifier.html` to
`eurosat.html` (or iframe it in — the map is full-viewport so a dedicated page
is cleaner).


## Step 4 — Commit and push

```bash
cd ~/Documents/anthonykovari.com
git add static/eurosat/ eurosat.html
git commit -m "Add Chicagoland land-use change viewer (static export)"
git push
```

GitHub Pages will serve immediately (usually < 2 min propagation).


## File layout in anthonykovari.com after export

```
static/eurosat/
  available.json
  sprawl.json
  timeseries.json
  satellite/
    2019.png  2020.png  ... 2026.png
  overlay/
    2019.png  2020.png  ... 2026.png
  grid/
    2019.json 2020.json ... 2026.json
eurosat.html    ← adapted copy of frontend/index.html
```


## Updating later (if a new year is added)

1. `docker compose up -d`
2. Trigger `chicago_fetch_year` + `chicago_classify_year` for the new year
3. `python3 scripts/export_static.py --years <new_year> --out ~/Documents/anthonykovari.com/static/eurosat`
4. Update `available.json` (re-fetch `/available`)
5. `git add . && git commit && git push`


## What is NOT included in the static version

- `/seg/overlay/{year}.png` — SegFormer masks not yet run; add when pipeline is complete.
- Live `/predict/` upload — skip entirely, portfolio doesn't need it.

## What IS included (clarification)

- **Compare mode** — the split-screen year comparison slider works fine. It just loads two
  `satellite/{year}.png` images side by side. No special backend call needed.
- `/change/{year_a}/{year_b}.png` is a backend-only endpoint the frontend doesn't actually
  use — ignore it.
