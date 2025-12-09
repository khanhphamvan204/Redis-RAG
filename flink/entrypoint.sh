#!/bin/bash
set -e

echo "=== Flink Streaming Job Submission ==="

# Wait for JobManager to be ready
echo "Waiting for Flink JobManager to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://flink-jobmanager:8081/overview > /dev/null 2>&1; then
        echo "✓ Flink JobManager is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting for JobManager... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "ERROR: Flink JobManager not available after $max_attempts attempts"
    exit 1
fi

# Wait for TaskManager to register with JobManager
echo "Waiting for TaskManager to register..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    slots=$(curl -s http://flink-jobmanager:8081/taskmanagers 2>/dev/null | grep -o '"slotsNumber":[0-9]*' | head -1 | grep -o '[0-9]*')
    if [ -n "$slots" ] && [ "$slots" -gt 0 ]; then
        echo "✓ TaskManager registered with $slots slots!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting for TaskManager to register... ($attempt/$max_attempts)"
    sleep 3
done

if [ $attempt -eq $max_attempts ]; then
    echo "ERROR: TaskManager not registered after $max_attempts attempts"
    exit 1
fi

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

# Additional delay to ensure all services are initialized
echo "Waiting additional 10 seconds for service initialization..."
sleep 10

echo "Submitting Flink job..."

# Submit Python job to Flink cluster
exec /opt/flink/bin/flink run \
    --jobmanager flink-jobmanager:8081 \
    --python /opt/flink/jobs/streaming_job.py \
    --pyFiles /opt/flink/jobs/streaming_job.py

echo "✓ Flink job submitted successfully"
