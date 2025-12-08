#!/bin/bash
set -e

echo "=== Spark Streaming Job Startup ==="

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if nc -z kafka 29092 2>/dev/null; then
        echo "✓ Kafka is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting for Kafka... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "ERROR: Kafka not available after $max_attempts attempts"
    exit 1
fi

# Additional startup delay to ensure Kafka is fully initialized
echo "Waiting additional 10 seconds for Kafka initialization..."
sleep 10

echo "Starting Spark Streaming Job..."

# Submit Spark job
exec spark-submit \
    --master local[2] \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.mongodb.spark:mongo-spark-connector_2.12:10.2.0 \
    --conf spark.driver.memory=2g \
    --conf spark.executor.memory=2g \
    --conf spark.sql.streaming.checkpointLocation=/tmp/checkpoint \
    --conf spark.mongodb.write.connection.uri="${MONGO_URI}" \
    /opt/spark/jobs/streaming_job.py


