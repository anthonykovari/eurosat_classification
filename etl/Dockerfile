FROM apache/airflow:2.9.0-python3.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
# CUDA-enabled PyTorch — wheels bundle libcudart so no separate toolkit install needed.
# GPU access is granted at runtime via nvidia-container-toolkit.
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt /tmp/etl-requirements.txt
RUN pip install --no-cache-dir -r /tmp/etl-requirements.txt
