FROM flink:1.18-java11

USER root

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip netcat-openbsd && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyFlink and dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download Kafka connector JARs for Flink
RUN wget -P /opt/flink/lib/ \
    https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.1.0-1.18/flink-sql-connector-kafka-3.1.0-1.18.jar

# Create necessary directories
RUN mkdir -p /opt/flink/jobs && \
    mkdir -p /tmp/flink-checkpoints && \
    mkdir -p /tmp/flink-savepoints && \
    chmod -R 777 /tmp/flink-checkpoints && \
    chmod -R 777 /tmp/flink-savepoints

# Copy job files
COPY streaming_job.py /opt/flink/jobs/streaming_job.py
COPY entrypoint.sh /opt/flink/entrypoint.sh
RUN chmod +x /opt/flink/entrypoint.sh

WORKDIR /opt/flink

# Default to running as flink user
USER flink
