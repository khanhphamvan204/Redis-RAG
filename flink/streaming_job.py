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
                'department_name': user.get('department_name'),  # Added for department analytics
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


class DepartmentAggregator(ProcessWindowFunction):
    """Aggregate queries by department name for pie chart"""
    
    def process(self, key, context, elements):
        elements_list = list(elements)
        
        if not elements_list:
            return
        
        user_ids = set()
        response_times = []
        success_count = 0
        
        for elem in elements_list:
            if elem is None:
                continue
            user_ids.add(elem.get('user_id'))
            
            rt = elem.get('rag_response_time_ms')
            if rt is not None:
                response_times.append(rt)
            
            if elem.get('rag_success'):
                success_count += 1
        
        result = {
            'department_name': key,
            'window_start': datetime.fromtimestamp(context.window().start / 1000).isoformat(),
            'window_end': datetime.fromtimestamp(context.window().end / 1000).isoformat(),
            'query_count': len(elements_list),
            'unique_users': len(user_ids),
            'success_count': success_count,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'analytics_type': 'department'
        }
        
        yield result


class YearAggregator(ProcessWindowFunction):
    """Aggregate queries by student year"""
    
    def process(self, key, context, elements):
        elements_list = list(elements)
        
        if not elements_list:
            return
        
        user_ids = set()
        response_times = []
        contexts_found = []
        
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
        
        result = {
            'year': key,
            'window_start': datetime.fromtimestamp(context.window().start / 1000).isoformat(),
            'window_end': datetime.fromtimestamp(context.window().end / 1000).isoformat(),
            'query_count': len(elements_list),
            'unique_users': len(user_ids),
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'avg_contexts_found': sum(contexts_found) / len(contexts_found) if contexts_found else 0,
            'analytics_type': 'student_year'
        }
        
        yield result


class PopularQuestionsAggregator(ProcessWindowFunction):
    """Aggregate popular questions across all users"""
    
    def process(self, key, context, elements):
        elements_list = list(elements)
        
        if not elements_list:
            return
        
        user_ids = set()
        query_text = None
        
        for elem in elements_list:
            if elem is None:
                continue
            user_ids.add(elem.get('user_id'))
            if query_text is None:
                query_text = elem.get('query_text', '')
        
        result = {
            'query_hash': key,
            'query_text': query_text,
            'window_start': datetime.fromtimestamp(context.window().start / 1000).isoformat(),
            'window_end': datetime.fromtimestamp(context.window().end / 1000).isoformat(),
            'query_count': len(elements_list),
            'unique_users': len(user_ids),
            'analytics_type': 'popular_queries'
        }
        
        yield result


class PopularQuestionsByYearAggregator(ProcessWindowFunction):
    """Aggregate popular questions grouped by student year for heatmap"""
    
    def process(self, key, context, elements):
        elements_list = list(elements)
        
        if not elements_list:
            return
        
        # key is tuple: (year, query_hash)
        year, query_hash = key
        
        user_ids = set()
        query_text = None
        
        for elem in elements_list:
            if elem is None:
                continue
            user_ids.add(elem.get('user_id'))
            if query_text is None:
                query_text = elem.get('query_text', '')
        
        result = {
            'year': year,
            'query_hash': query_hash,
            'query_text': query_text,
            'window_start': datetime.fromtimestamp(context.window().start / 1000).isoformat(),
            'window_end': datetime.fromtimestamp(context.window().end / 1000).isoformat(),
            'query_count': len(elements_list),
            'unique_users': len(user_ids),
            'analytics_type': 'popular_by_year'
        }
        
        yield result


class OverallSummaryAggregator(ProcessWindowFunction):
    """Aggregate overall analytics summary for dashboard cards"""
    
    def process(self, key, context, elements):
        elements_list = list(elements)
        
        if not elements_list:
            return
        
        user_ids = set()
        response_times = []
        success_count = 0
        
        for elem in elements_list:
            if elem is None:
                continue
            user_ids.add(elem.get('user_id'))
            
            rt = elem.get('rag_response_time_ms')
            if rt is not None:
                response_times.append(rt)
            
            if elem.get('rag_success'):
                success_count += 1
        
        total_queries = len(elements_list)
        failure_count = total_queries - success_count
        success_rate = round((success_count / total_queries * 100), 2) if total_queries > 0 else 0
        
        result = {
            'window_start': datetime.fromtimestamp(context.window().start / 1000).isoformat(),
            'window_end': datetime.fromtimestamp(context.window().end / 1000).isoformat(),
            'total_queries': total_queries,
            'success_count': success_count,
            'failure_count': failure_count,
            'success_rate': success_rate,
            'unique_users': len(user_ids),
            'avg_response_time': round(sum(response_times) / len(response_times), 2) if response_times else 0,
            'analytics_type': 'overall'
        }
        
        yield result


class RedisWriter(MapFunction):
    """Write aggregated data to Redis"""
    
    def map(self, value):
        try:
            r = get_redis_client()
            timestamp = datetime.now().isoformat()
            analytics_type = value.pop('analytics_type', 'unknown')
            
            # Create Redis key based on analytics type
            if analytics_type == 'department':
                dept = value.get('department_name', 'unknown')
                key = f"analytics:department:{dept}:{timestamp}"
            elif analytics_type == 'student_year':
                year = value.get('year', 'unknown')
                key = f"analytics:student_year:{year}:{timestamp}"
            elif analytics_type == 'popular_queries':
                query_hash = value.get('query_hash', 'unknown')
                key = f"analytics:popular_queries:{query_hash}:{timestamp}"
            elif analytics_type == 'popular_by_year':
                year = value.get('year', 'unknown')
                query_hash = value.get('query_hash', 'unknown')
                key = f"analytics:popular_by_year:{year}:{query_hash}:{timestamp}"
            elif analytics_type == 'overall':
                key = f"analytics:overall:{timestamp}"
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
    # AGGREGATION 1: Department Analytics (20-second window) - For Pie Chart
    # ========================================
    logger.info("Setting up Department Analytics (Pie Chart)...")
    
    department_stream = valid_stream \
        .filter(lambda x: x.get('department_name') is not None) \
        .key_by(lambda x: x.get('department_name')) \
        .window(TumblingProcessingTimeWindows.of(Time.seconds(20))) \
        .process(DepartmentAggregator()) \
        .map(RedisWriter())
    
    department_stream.print()
    
    # ========================================
    # AGGREGATION 2: Student Year Analytics (20-second window) - For Bar Chart
    # ========================================
    logger.info("Setting up Student Year Analytics (Bar Chart)...")
    
    year_stream = valid_stream \
        .filter(lambda x: x.get('user_years') is not None) \
        .key_by(lambda x: str(x.get('user_years'))) \
        .window(TumblingProcessingTimeWindows.of(Time.seconds(20))) \
        .process(YearAggregator()) \
        .map(RedisWriter())
    
    year_stream.print()
    
    # ========================================
    # AGGREGATION 3: Popular Questions (20-second window) - For Table/List
    # ========================================
    logger.info("Setting up Popular Questions Analytics...")
    
    popular_stream = valid_stream \
        .filter(lambda x: x.get('query_text')) \
        .key_by(lambda x: hashlib.md5(x.get('query_text', '').encode()).hexdigest()[:8]) \
        .window(TumblingProcessingTimeWindows.of(Time.seconds(20))) \
        .process(PopularQuestionsAggregator()) \
        .map(RedisWriter())
    
    popular_stream.print()
    
    # ========================================
    # AGGREGATION 3.5: Popular Questions by Year (20-second window) - For Heatmap
    # ========================================
    logger.info("Setting up Popular Questions by Year Analytics (Heatmap)...")
    
    popular_by_year_stream = valid_stream \
        .filter(lambda x: x.get('user_years') is not None and x.get('query_text')) \
        .key_by(lambda x: (
            str(x.get('user_years')), 
            hashlib.md5(x.get('query_text', '').encode()).hexdigest()[:8]
        )) \
        .window(TumblingProcessingTimeWindows.of(Time.seconds(20))) \
        .process(PopularQuestionsByYearAggregator()) \
        .map(RedisWriter())
    
    popular_by_year_stream.print()

    
    # ========================================
    # AGGREGATION 4: Overall Summary (20-second window) - For Dashboard Cards
    # ========================================
    logger.info("Setting up Overall Summary Analytics (Dashboard Cards)...")
    
    overall_stream = valid_stream \
        .key_by(lambda x: "all") \
        .window(TumblingProcessingTimeWindows.of(Time.seconds(20))) \
        .process(OverallSummaryAggregator()) \
        .map(RedisWriter())
    
    overall_stream.print()
    
    # Execute
    logger.info("=" * 80)
    logger.info("Starting Flink job execution...")
    logger.info("=" * 80)
    
    env.execute("Query Analytics Streaming Job")
    
    logger.info("✓ Flink streaming job started successfully")


if __name__ == "__main__":
    main()