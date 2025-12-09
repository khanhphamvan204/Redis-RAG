"""
Apache Flink Streaming Job for Real-time Query Analytics
SIMPLE SOLUTION: Use DataStream API instead of SQL for nested field access
"""
import logging
import os
import json
import hashlib
from datetime import datetime
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import Types, WatermarkStrategy, Duration
from pyflink.datastream.window import TumblingProcessingTimeWindows, Time
from pyflink.datastream.functions import ProcessWindowFunction, MapFunction
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "redis-stack-db")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_TTL = int(os.getenv("REDIS_TTL", "2592000"))  # 30 days in seconds

KAFKA_BROKER = os.getenv("KAFKA_BROKER_URL", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_QUERIES", "user-queries")

# Redis client
redis_client = None


def get_redis_client():
    """Get or create Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        logger.info(f"✓ Redis client connected: {REDIS_HOST}:{REDIS_PORT}")
    return redis_client


class ParseQueryEvent(MapFunction):
    """Parse JSON string to structured data"""
    
    def map(self, value):
        try:
            data = json.loads(value)
            
            # Flatten nested structures for easier processing
            user = data.get('user', {})
            rag_params = data.get('rag_params', {})
            rag_metrics = data.get('rag_metrics', {})
            metadata = data.get('metadata', {})
            
            return {
                'query_id': data.get('query_id'),
                'timestamp': data.get('timestamp'),
                'ts': data.get('ts'),
                'user_id': user.get('user_id'),
                'user_type': user.get('user_type'),
                'department_id': user.get('department_id'),
                'user_code': user.get('code'),
                'user_years': user.get('years'),
                'session_id': data.get('session_id'),
                'query_text': data.get('query_text'),
                'rewritten_query': data.get('rewritten_query'),
                'rag_k': rag_params.get('k'),
                'rag_similarity_threshold': rag_params.get('similarity_threshold'),
                'rag_context_found': rag_params.get('context_found'),
                'rag_response_time_ms': rag_metrics.get('response_time_ms'),
                'rag_answer_length': rag_metrics.get('answer_length'),
                'rag_answer_text': rag_metrics.get('answer_text'),
                'rag_success': rag_metrics.get('success'),
                'metadata_model_used': metadata.get('model_used'),
                'metadata_history_used': metadata.get('history_used'),
                'metadata_history_count': metadata.get('history_count'),
                'metadata_query_rewritten': metadata.get('query_rewritten'),
                'event_time': datetime.fromisoformat(data.get('timestamp', data.get('ts', datetime.utcnow().isoformat())))
            }
        except Exception as e:
            logger.error(f"Error parsing event: {e}")
            return None


class FacultyAggregator(ProcessWindowFunction):
    """Aggregate queries by faculty"""
    
    def process(self, key, context, elements):
        elements_list = list(elements)
        
        if not elements_list:
            return
        
        user_ids = set()
        response_times = []
        contexts_found = []
        rewritten_count = 0
        history_used_count = 0
        
        for elem in elements_list:
            if elem is None:
                continue
            user_ids.add(elem.get('user_id'))
            
            rt = elem.get('rag_response_time_ms')
            if rt is not None:
                response_times.append(rt)
            
            cf = elem.get('rag_context_found')
            if cf is not None:
                contexts_found.append(cf)
            
            if elem.get('metadata_query_rewritten'):
                rewritten_count += 1
            
            if elem.get('metadata_history_used'):
                history_used_count += 1
        
        result = {
            'faculty': key,
            'window_start': datetime.fromtimestamp(context.window().start / 1000).isoformat(),
            'window_end': datetime.fromtimestamp(context.window().end / 1000).isoformat(),
            'query_count': len(elements_list),
            'unique_users': len(user_ids),
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'avg_contexts_found': sum(contexts_found) / len(contexts_found) if contexts_found else 0,
            'rewritten_queries': rewritten_count,
            'history_used_count': history_used_count,
            'analytics_type': 'faculty'
        }
        
        yield result


class RedisWriter(MapFunction):
    """Write aggregated data to Redis"""
    
    def map(self, value):
        try:
            r = get_redis_client()
            timestamp = datetime.now().isoformat()
            analytics_type = value.pop('analytics_type', 'unknown')
            
            # Create Redis key
            if analytics_type == 'faculty':
                faculty = value.get('faculty', 'unknown')
                key = f"analytics:faculty:{faculty}:{timestamp}"
            elif analytics_type == 'year':
                year = value.get('year', 'unknown')
                key = f"analytics:year:{year}:{timestamp}"
            elif analytics_type == 'heatmap':
                faculty = value.get('faculty', 'unknown')
                year = value.get('year', 'unknown')
                key = f"analytics:heatmap:{faculty}:{year}:{timestamp}"
            else:
                key = f"analytics:{analytics_type}:{timestamp}"
            
            # Store in Redis with TTL
            r.setex(name=key, time=REDIS_TTL, value=json.dumps(value))
            
            # Update latest timestamp
            r.set("analytics:latest", timestamp)
            
            # Publish update notification
            r.publish("analytics:updates", json.dumps({
                "type": analytics_type,
                "timestamp": timestamp,
                "count": 1
            }))
            
            logger.info(f"✓ Written {analytics_type} to Redis: {key}")
            return f"Written: {key}"
            
        except Exception as e:
            logger.error(f"Error writing to Redis: {e}", exc_info=True)
            return f"Error: {str(e)}"


def main():
    """Main Flink streaming job using DataStream API"""
    logger.info("=" * 80)
    logger.info("Starting Apache Flink DataStream Job for Query Analytics...")
    logger.info("=" * 80)
    
    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)
    
    logger.info(f"Kafka Broker: {KAFKA_BROKER}")
    logger.info(f"Kafka Topic: {KAFKA_TOPIC}")
    logger.info(f"Redis Host: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Redis TTL: {REDIS_TTL} seconds (~30 days)")
    
    # Create Kafka source
    kafka_source = KafkaSource.builder() \
        .set_bootstrap_servers(KAFKA_BROKER) \
        .set_topics(KAFKA_TOPIC) \
        .set_group_id("flink-analytics-group") \
        .set_starting_offsets(KafkaOffsetsInitializer.earliest()) \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()
    
    # Read from Kafka - use earliest to get existing messages for testing
    kafka_stream = env.from_source(
        kafka_source,
        WatermarkStrategy.no_watermarks(),
        "Kafka Source"
    )
    

    
    # Parse JSON events
    parsed_stream = kafka_stream.map(ParseQueryEvent())
    
    # Filter out None values (failed parsing)
    valid_stream = parsed_stream.filter(lambda x: x is not None)
    
    # ========================================
    # AGGREGATION 1: Faculty Analytics (20-second window)
    # ========================================
    logger.info("Setting up Faculty Analytics...")
    
    faculty_stream = valid_stream \
        .filter(lambda x: x.get('user_type') is not None) \
        .key_by(lambda x: x.get('user_type')) \
        .window(TumblingProcessingTimeWindows.of(Time.seconds(20))) \
        .process(FacultyAggregator()) \
        .map(RedisWriter())
    
    faculty_stream.print()
    
    # Execute
    logger.info("=" * 80)
    logger.info("Starting Flink job execution...")
    logger.info("=" * 80)
    
    env.execute("Query Analytics Streaming Job")
    
    logger.info("✓ Flink streaming job started successfully")


if __name__ == "__main__":
    main()