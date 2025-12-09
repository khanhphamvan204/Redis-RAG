"""
Spark Streaming Job for Real-time Query Analytics
Consumes query events from Kafka and writes to Redis with TTL
"""
import logging
import os
import json
import redis
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, count, avg, approx_count_distinct,
    to_timestamp, current_timestamp, lower, trim, md5,
    when, sum, expr, hour
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, BooleanType, TimestampType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "redis-stack-db")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_TTL = 30 * 24 * 60 * 60  # 30 days in seconds (2,592,000)

KAFKA_BROKER = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_QUERIES", "user-queries")

# Initialize Redis client (will be used in batch processing)
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


def create_spark_session():
    """Create Spark session for Kafka streaming (OPTIMIZED)"""
    return SparkSession.builder \
        .appName("QueryAnalyticsStreaming") \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.3") \
        .config("spark.sql.streaming.metricsEnabled", "true") \
        .config("spark.sql.streaming.numRecentProgressUpdates", "10") \
        .config("spark.streaming.kafka.maxRatePerPartition", "5000") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.default.parallelism", "8") \
        .config("spark.sql.streaming.stateStore.maintenanceInterval", "30s") \
        .getOrCreate()


def define_schema():
    """Define schema for query log events (NESTED STRUCTURE)"""
    
    # Nested structures
    user_schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("user_type", StringType(), True),
        StructField("department_id", IntegerType(), True),
        StructField("code", StringType(), True),
        StructField("years", IntegerType(), True)
    ])
    
    rag_params_schema = StructType([
        StructField("k", IntegerType(), True),
        StructField("similarity_threshold", FloatType(), True),
        StructField("context_found", IntegerType(), True)
    ])
    
    rag_metrics_schema = StructType([
        StructField("response_time_ms", FloatType(), True),
        StructField("answer_length", IntegerType(), True),
        StructField("answer_text", StringType(), True),
        StructField("success", BooleanType(), True)
    ])
    
    metadata_schema = StructType([
        StructField("model_used", StringType(), True),
        StructField("history_used", BooleanType(), True),
        StructField("history_count", IntegerType(), True),
        StructField("query_rewritten", BooleanType(), True)
    ])
    
    # Main schema
    return StructType([
        StructField("query_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("user", user_schema, True),
        StructField("session_id", StringType(), True),
        StructField("query_text", StringType(), True),
        StructField("rewritten_query", StringType(), True),
        StructField("rag_params", rag_params_schema, True),
        StructField("rag_metrics", rag_metrics_schema, True),
        StructField("metadata", metadata_schema, True)
    ])


def write_to_redis(batch_df, batch_id, analytics_type):
    """
    Write batch dataframe to Redis with TTL
    
    Args:
        batch_df: Spark DataFrame with aggregated data
        batch_id: Batch ID for logging
        analytics_type: Type of analytics
    """
    try:
        logger.info(f"[{analytics_type}] Processing batch {batch_id}...")
        
        # Check if DataFrame is empty - FAST check
        try:
            row_count = batch_df.count()
            logger.info(f"[{analytics_type}] Batch {batch_id} has {row_count} records")
        except Exception as count_err:
            logger.error(f"[{analytics_type}] Error counting rows: {count_err}")
            return
        
        if row_count == 0:
            logger.info(f"[{analytics_type}] Batch {batch_id} is empty, skipping")
            return
        
        # Show sample data for debugging
        logger.info(f"[{analytics_type}] Sample data:")
        batch_df.show(3, truncate=False)
        
        # Get Redis client
        r = get_redis_client()
        
        # Convert DataFrame to list of dictionaries
        rows = batch_df.collect()
        
        for row in rows:
            row_dict = row.asDict()
            
            # Create Redis key based on analytics type
            timestamp = datetime.now().isoformat()
            
            # Legacy types (for backward compatibility)
            if analytics_type == 'faculty':
                faculty = row_dict.get('faculty', 'unknown')
                key = f"analytics:faculty:{faculty}:{timestamp}"
                
            elif analytics_type == 'year':
                year = row_dict.get('year', 'unknown')
                key = f"analytics:year:{year}:{timestamp}"
                
            elif analytics_type == 'heatmap':
                faculty = row_dict.get('faculty', 'unknown')
                year = row_dict.get('year', 'unknown')
                key = f"analytics:heatmap:{faculty}:{year}:{timestamp}"
                
            elif analytics_type == 'popular_queries':
                query_hash = row_dict.get('query_hash', 'unknown')
                key = f"analytics:popular_queries:{query_hash}:{timestamp}"
            
            # New analytics types
            elif analytics_type == 'student_year':
                years = row_dict.get('years', 'unknown')
                key = f"analytics:student_year:{years}:{timestamp}"
            
            elif analytics_type == 'popular_by_year':
                years = row_dict.get('years', 'unknown')
                query_hash = row_dict.get('query_hash', 'unknown')
                key = f"analytics:popular_by_year:{years}:{query_hash}:{timestamp}"
            
            elif analytics_type == 'department':
                dept_id = row_dict.get('department_id', 'unknown')
                key = f"analytics:department:{dept_id}:{timestamp}"
            
            elif analytics_type == 'user_type':
                user_type = row_dict.get('user_type', 'unknown')
                key = f"analytics:user_type:{user_type}:{timestamp}"
            
            elif analytics_type == 'success_rate':
                key = f"analytics:success_rate:{timestamp}"
            
            elif analytics_type == 'response_time':
                user_type = row_dict.get('user_type', 'unknown')
                key = f"analytics:response_time:{user_type}:{timestamp}"
            
            elif analytics_type == 'timeseries_minute':
                key = f"analytics:timeseries:minute:{timestamp}"
            
            elif analytics_type == 'timeseries_hour':
                key = f"analytics:timeseries:hour:{timestamp}"
            
            elif analytics_type == 'timeseries_day':
                key = f"analytics:timeseries:day:{timestamp}"
            
            elif analytics_type == 'heatmap_hourly':
                hour = row_dict.get('hour_of_day', 'unknown')
                key = f"analytics:heatmap:hourly:{hour}:{timestamp}"
            
            else:
                logger.warning(f"Unknown analytics_type: {analytics_type}")
                continue
            
            # Convert to JSON-serializable format
            for k, v in row_dict.items():
                if hasattr(v, 'isoformat'):  # datetime object
                    row_dict[k] = v.isoformat()
            
            # Store in Redis with TTL
            r.setex(
                name=key,
                time=REDIS_TTL,
                value=json.dumps(row_dict)
            )
        
        # Update latest timestamp
        r.set("analytics:latest", timestamp)
        
        # Publish update notification for WebSocket clients
        r.publish("analytics:updates", json.dumps({
            "type": analytics_type,
            "timestamp": timestamp,
            "count": row_count
        }))
        
        logger.info(f"✓ [{analytics_type}] Batch {batch_id} written to Redis: {row_count} records")
        
    except Exception as e:
        logger.error(f"✗ [{analytics_type}] Error writing batch {batch_id}: {e}", exc_info=True)


def main():
    """Main streaming job"""
    logger.info("=" * 80)
    logger.info("Starting Spark Streaming Job for Query Analytics...")
    logger.info("=" * 80)
    
    # Create Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Kafka Broker: {KAFKA_BROKER}")
    logger.info(f"Kafka Topic: {KAFKA_TOPIC}")
    logger.info(f"Redis Host: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Redis TTL: {REDIS_TTL} seconds (~30 days)")
    
    # Read stream from Kafka
    logger.info("Connecting to Kafka...")
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    logger.info("✓ Connected to Kafka successfully")
    
    # Parse JSON messages
    schema = define_schema()
    logger.info("✓ Schema defined")
    
    # Parse and validate
    query_stream = raw_stream.selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), schema).alias("data")) \
        .select("data.*") \
        .withColumn("event_time", to_timestamp(col("timestamp"))) \
        .filter(col("query_id").isNotNull())  # Filter out invalid records
    
    logger.info("✓ Stream pipeline configured")
    logger.info("=" * 80)
    
    # ========================================
    # AGGREGATION 1: By User Type (20-second window)
    # ========================================
    faculty_agg = query_stream \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(
            window(col("event_time"), "20 seconds"),
            col("user.user_type").alias("user_type")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user.user_id").alias("unique_users"),
            avg("rag_metrics.response_time_ms").alias("avg_response_time"),
            avg("rag_params.context_found").alias("avg_contexts_found"),
            count(when(col("metadata.query_rewritten") == True, 1)).alias("rewritten_queries"),
            count(when(col("metadata.history_used") == True, 1)).alias("history_used_count")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("user_type").alias("faculty"),
            col("query_count"),
            col("unique_users"),
            col("avg_response_time"),
            col("avg_contexts_found"),
            col("rewritten_queries"),
            col("history_used_count"),
            current_timestamp().alias("created_at")
        )
    
    faculty_query = faculty_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "faculty")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/faculty") \
        .start()

    logger.info("✓ Faculty aggregation stream started")    # ========================================
    # AGGREGATION 2: By Student Year (20-second window)
    # ========================================
    year_agg = query_stream \
        .filter(col("user.years").isNotNull()) \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(
            window(col("event_time"), "20 seconds"),
            col("user.years").alias("years")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user.user_id").alias("unique_users"),
            avg("rag_metrics.response_time_ms").alias("avg_response_time"),
            avg("rag_params.context_found").alias("avg_contexts_found")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("years").alias("year"),
            col("query_count"),
            col("unique_users"),
            col("avg_response_time"),
            col("avg_contexts_found"),
            current_timestamp().alias("created_at")
        )
    
    year_query = year_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "year")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/year") \
        .start()

    logger.info("✓ Year aggregation stream started")    # ========================================
    # AGGREGATION 3: User Type-Year Heatmap (30-second window)
    # ========================================
    heatmap_agg = query_stream \
        .filter(col("user.years").isNotNull()) \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(
            window(col("event_time"), "30 seconds"),
            col("user.user_type").alias("user_type"),
            col("user.years").alias("years")
        ) \
        .agg(
            count("*").alias("query_count"),
            avg("rag_metrics.response_time_ms").alias("avg_response_time")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("user_type").alias("faculty"),
            col("years").alias("year"),
            col("query_count"),
            col("avg_response_time"),
            current_timestamp().alias("created_at")
        )
    
    heatmap_query = heatmap_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "heatmap")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/heatmap") \
        .start()

    logger.info("✓ Heatmap aggregation stream started")    # ========================================
    # AGGREGATION 4: Popular Questions (30-second window)
    # ========================================
    popular_queries_agg = query_stream \
        .withWatermark("event_time", "20 seconds") \
        .withColumn("normalized_query", lower(trim(col("query_text")))) \
        .groupBy(
            window(col("event_time"), "30 seconds"),
            col("normalized_query")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user.user_id").alias("unique_users")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("normalized_query").alias("query_text"),
            md5(col("normalized_query")).alias("query_hash"),
            col("query_count"),
            col("unique_users"),
            current_timestamp().alias("created_at")
        )
    
    popular_queries_query = popular_queries_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "popular_queries")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/popular_queries") \
        .start()
    
    logger.info("✓ Popular queries aggregation stream started")
    
    # ========================================
    # AGGREGATION 5: By Student Year (1-minute window)
    # ========================================
    student_year_agg = query_stream \
        .filter(col("user.user_type").isin(["Học sinh", "Sinh viên"])) \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(
            window(col("event_time"), "1 minute"),
            col("user.years").alias("years")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user.user_id").alias("unique_users"),
            avg("rag_metrics.response_time_ms").alias("avg_response_time"),
            avg("rag_params.context_found").alias("avg_contexts_found")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("years"),
            col("query_count"),
            col("unique_users"),
            col("avg_response_time"),
            col("avg_contexts_found"),
            current_timestamp().alias("created_at")
        )
    
    student_year_query = student_year_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "student_year")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/student_year") \
        .start()
    
    logger.info("✓ Student year aggregation stream started")
    
    # ========================================
    # AGGREGATION 6: Popular Questions by Year (2-minute window)
    # ========================================
    popular_by_year_agg = query_stream \
        .filter(col("user.user_type").isin(["Học sinh", "Sinh viên"])) \
        .withColumn("normalized_query", lower(trim(col("query_text")))) \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(
            window(col("event_time"), "2 minutes"),
            col("user.years").alias("years"),
            col("normalized_query")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user.user_id").alias("unique_users")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("years"),
            col("normalized_query").alias("query_text"),
            md5(col("normalized_query")).alias("query_hash"),
            col("query_count"),
            col("unique_users"),
            current_timestamp().alias("created_at")
        )
    
    popular_by_year_query = popular_by_year_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "popular_by_year")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/popular_by_year") \
        .start()
    
    logger.info("✓ Popular by year aggregation stream started")
    
    # ========================================
    # AGGREGATION 7: By Department (1-minute window)
    # ========================================
    department_agg = query_stream \
        .filter(col("user.department_id").isNotNull()) \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(
            window(col("event_time"), "1 minute"),
            col("user.department_id").alias("department_id")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user.user_id").alias("unique_users"),
            avg("rag_metrics.response_time_ms").alias("avg_response_time")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("department_id"),
            col("query_count"),
            col("unique_users"),
            col("avg_response_time"),
            current_timestamp().alias("created_at")
        )
    
    department_query = department_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "department")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/department") \
        .start()
    
    logger.info("✓ Department aggregation stream started")
    
    # ========================================
    # AGGREGATION 8: User Type Distribution (1-minute window)
    # ========================================
    user_type_agg = query_stream \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(
            window(col("event_time"), "1 minute"),
            col("user.user_type").alias("user_type")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user.user_id").alias("unique_users")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("user_type"),
            col("query_count"),
            col("unique_users"),
            current_timestamp().alias("created_at")
        )
    
    user_type_query = user_type_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "user_type")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/user_type") \
        .start()

    logger.info("✓ User type aggregation stream started")    # ========================================
    # AGGREGATION 9: Success Rate (1-minute window)
    # ========================================
    success_rate_agg = query_stream \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(window(col("event_time"), "1 minute")) \
        .agg(
            count("*").alias("total_queries"),
            sum(when(col("rag_metrics.success") == True, 1).otherwise(0)).alias("success_count"),
            sum(when(col("rag_metrics.success") == False, 1).otherwise(0)).alias("failure_count")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("total_queries"),
            col("success_count"),
            col("failure_count"),
            (col("success_count") / col("total_queries") * 100).alias("success_rate_pct"),
            current_timestamp().alias("created_at")
        )
    
    success_rate_query = success_rate_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "success_rate")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/success_rate") \
        .start()

    logger.info("✓ Success rate aggregation stream started")    # ========================================
    # AGGREGATION 10: Response Time Analytics (1-minute window)
    # ========================================
    response_time_agg = query_stream \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(
            window(col("event_time"), "1 minute"),
            col("user.user_type").alias("user_type")
        ) \
        .agg(
            avg("rag_metrics.response_time_ms").alias("avg_time"),
            expr("percentile_approx(rag_metrics.response_time_ms, 0.5)").alias("median_time"),
            expr("percentile_approx(rag_metrics.response_time_ms, 0.95)").alias("p95_time"),
            count("*").alias("query_count")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("user_type"),
            col("avg_time"),
            col("median_time"),
            col("p95_time"),
            col("query_count"),
            current_timestamp().alias("created_at")
        )
    
    response_time_query = response_time_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "response_time")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/response_time") \
        .start()

    logger.info("✓ Response time aggregation stream started")    # ========================================
    # AGGREGATION 11-13: Time-Series (minute/hour/day)
    # ========================================
    timeseries_minute_agg = query_stream \
        .withWatermark("event_time", "20 seconds") \
        .groupBy(window(col("event_time"), "1 minute")) \
        .agg(count("*").alias("query_count")) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("query_count"),
            current_timestamp().alias("created_at")
        )
    
    timeseries_minute_query = timeseries_minute_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "timeseries_minute")) \
        .trigger(processingTime="2 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/timeseries_minute") \
        .start()
    
    timeseries_hour_agg = query_stream \
        .withWatermark("event_time", "2 minutes") \
        .groupBy(window(col("event_time"), "1 hour")) \
        .agg(count("*").alias("query_count")) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("query_count"),
            current_timestamp().alias("created_at")
        )
    
    timeseries_hour_query = timeseries_hour_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "timeseries_hour")) \
        .trigger(processingTime="30 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/timeseries_hour") \
        .start()
    
    timeseries_day_agg = query_stream \
        .withWatermark("event_time", "30 minutes") \
        .groupBy(window(col("event_time"), "1 day")) \
        .agg(count("*").alias("query_count")) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("query_count"),
            current_timestamp().alias("created_at")
        )
    
    timeseries_day_query = timeseries_day_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "timeseries_day")) \
        .trigger(processingTime="120 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/timeseries_day") \
        .start()
    
    logger.info("✓ Time-series aggregation streams started (minute/hour/day)")
    
    # ========================================
    # AGGREGATION 14: Hourly Heatmap (1-hour window)
    # ========================================
    heatmap_hourly_agg = query_stream \
        .withColumn("hour_of_day", hour(col("event_time"))) \
        .withWatermark("event_time", "2 minutes") \
        .groupBy(
            window(col("event_time"), "1 hour"),
            col("hour_of_day")
        ) \
        .agg(count("*").alias("query_count")) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("hour_of_day"),
            col("query_count"),
            current_timestamp().alias("created_at")
        )
    
    heatmap_hourly_query = heatmap_hourly_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_redis(df, id, "heatmap_hourly")) \
        .trigger(processingTime="30 seconds") \
        .option("checkpointLocation", "/tmp/checkpoint/heatmap_hourly") \
        .start()
    
    logger.info("✓ Hourly heatmap aggregation stream started")
    
    # ========================================
    # Wait for termination
    # ========================================
    logger.info("=" * 80)
    logger.info("✓ ALL STREAMING QUERIES STARTED (14 total)")
    logger.info("=" * 80)
    logger.info("Waiting for data from Kafka...")
    
    try:
        faculty_query.awaitTermination()
        year_query.awaitTermination()
        heatmap_query.awaitTermination()
        popular_queries_query.awaitTermination()
        student_year_query.awaitTermination()
        popular_by_year_query.awaitTermination()
        department_query.awaitTermination()
        user_type_query.awaitTermination()
        success_rate_query.awaitTermination()
        response_time_query.awaitTermination()
        timeseries_minute_query.awaitTermination()
        timeseries_hour_query.awaitTermination()
        timeseries_day_query.awaitTermination()
        heatmap_hourly_query.awaitTermination()
        
    except KeyboardInterrupt:
        logger.info("Streaming job interrupted by user")
    except Exception as e:
        logger.error(f"Streaming job error: {e}", exc_info=True)
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()