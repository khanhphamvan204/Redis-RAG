FROM apache/spark:3.4.3

USER root

# Install netcat for health checks
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create directories
RUN mkdir -p /opt/spark/jobs && \
    mkdir -p /tmp/checkpoint && \
    chmod -R 777 /tmp/checkpoint

# Copy streaming job and entrypoint
COPY streaming_job.py /opt/spark/jobs/streaming_job.py
COPY entrypoint.sh /opt/spark/entrypoint.sh
RUN chmod +x /opt/spark/entrypoint.sh

WORKDIR /opt/spark

ENTRYPOINT ["/opt/spark/entrypoint.sh"]


