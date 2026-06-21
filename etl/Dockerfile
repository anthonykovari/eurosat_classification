FROM apache/airflow:2.9.0-python3.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
# Install CPU-only PyTorch first so pip doesn't pull the CUDA variant
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/etl-requirements.txt
RUN pip install --no-cache-dir -r /tmp/etl-requirements.txt
