FROM apache/airflow:2.9.0-python3.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/etl-requirements.txt
RUN pip install --no-cache-dir -r /tmp/etl-requirements.txt
