# spark/streaming_job.py
"""
Spark Streaming Job for Real-time Query Analytics
Consumes query events from Kafka and performs window-based aggregations
"""
import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, count, avg, approx_count_distinct,
    to_timestamp, current_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, BooleanType, TimestampType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MONGO_URI = os.getenv("DATABASE_URL", "mongodb://127.0.0.1:27017/faiss_db")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "faiss_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION_ANALYTICS", "query_analytics")

KAFKA_BROKER = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_QUERIES", "user-queries")


def create_spark_session():
    """Create Spark session with MongoDB connector"""
    return SparkSession.builder \
        .appName("QueryAnalyticsStreaming") \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
        .getOrCreate()


def define_schema():
    """Define schema for query log events"""
    return StructType([
        StructField("query_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("session_id", StringType(), True),
        StructField("query_text", StringType(), True),
        StructField("rewritten_query", StringType(), True),
        StructField("faculty", StringType(), True),
        StructField("year", StringType(), True),
        StructField("file_type", StringType(), True),
        StructField("k", IntegerType(), True),
        StructField("response_time_ms", FloatType(), True),
        StructField("contexts_found", IntegerType(), True),
        StructField("similarity_threshold", FloatType(), True),
        StructField("query_rewritten", BooleanType(), True),
        StructField("history_used", BooleanType(), True),
        StructField("history_count", IntegerType(), True),
        StructField("timestamp", StringType(), True)
    ])


def write_to_mongo(batch_df, batch_id, collection_suffix):
    """Write batch dataframe to MongoDB"""
    try:
        if batch_df.count() > 0:
            batch_df.write \
                .format("mongodb") \
                .mode("append") \
                .option("database", MONGO_DATABASE) \
                .option("collection", f"{MONGO_COLLECTION}_{collection_suffix}") \
                .option("replaceDocument", "false") \
                .save()
            logger.info(f"Batch {batch_id} written to MongoDB ({collection_suffix}): {batch_df.count()} records")
    except Exception as e:
        logger.error(f"Error writing batch {batch_id} to MongoDB: {e}")


def main():
    """Main streaming job"""
    logger.info("Starting Spark Streaming Job for Query Analytics...")
    
    # Create Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Connecting to Kafka broker: {KAFKA_BROKER}, topic: {KAFKA_TOPIC}")
    
    # Read stream from Kafka
    raw_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    # Parse JSON messages
    schema = define_schema()
    
    query_stream = raw_stream.selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), schema).alias("data")) \
        .select("data.*") \
        .withColumn("event_time", to_timestamp(col("timestamp")))
    
    logger.info("Schema defined. Starting aggregations...")
    
    # ========================================
    # AGGREGATION 1: By Faculty (5-minute window)
    # ========================================
    faculty_agg = query_stream \
        .withWatermark("event_time", "10 minutes") \
        .groupBy(
            window(col("event_time"), "5 minutes", "1 minute"),
            col("faculty")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user_id").alias("unique_users"),
            avg("response_time_ms").alias("avg_response_time"),
            avg("contexts_found").alias("avg_contexts_found"),
            count(col("query_rewritten")).alias("rewritten_queries"),
            count(col("history_used")).alias("history_used_count")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("faculty"),
            col("query_count"),
            col("unique_users"),
            col("avg_response_time"),
            col("avg_contexts_found"),
            col("rewritten_queries"),
            col("history_used_count"),
            current_timestamp().alias("created_at")
        )
    
    # Write faculty aggregations to MongoDB
    faculty_query = faculty_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_mongo(df, id, "by_faculty")) \
        .option("checkpointLocation", "/tmp/checkpoint/faculty") \
        .start()
    
    logger.info("Faculty aggregation stream started")
    
    # ========================================
    # AGGREGATION 2: By Year (5-minute window)
    # ========================================
    year_agg = query_stream \
        .withWatermark("event_time", "10 minutes") \
        .groupBy(
            window(col("event_time"), "5 minutes", "1 minute"),
            col("year")
        ) \
        .agg(
            count("*").alias("query_count"),
            approx_count_distinct("user_id").alias("unique_users"),
            avg("response_time_ms").alias("avg_response_time"),
            avg("contexts_found").alias("avg_contexts_found")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("year"),
            col("query_count"),
            col("unique_users"),
            col("avg_response_time"),
            col("avg_contexts_found"),
            current_timestamp().alias("created_at")
        )
    
    # Write year aggregations to MongoDB
    year_query = year_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_mongo(df, id, "by_year")) \
        .option("checkpointLocation", "/tmp/checkpoint/year") \
        .start()
    
    logger.info("Year aggregation stream started")
    
    # ========================================
    # AGGREGATION 3: Faculty-Year Heatmap (15-minute window)
    # ========================================
    heatmap_agg = query_stream \
        .withWatermark("event_time", "20 minutes") \
        .groupBy(
            window(col("event_time"), "15 minutes", "5 minutes"),
            col("faculty"),
            col("year")
        ) \
        .agg(
            count("*").alias("query_count"),
            avg("response_time_ms").alias("avg_response_time")
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("faculty"),
            col("year"),
            col("query_count"),
            col("avg_response_time"),
            current_timestamp().alias("created_at")
        )
    
    # Write heatmap aggregations to MongoDB
    heatmap_query = heatmap_agg.writeStream \
        .outputMode("append") \
        .foreachBatch(lambda df, id: write_to_mongo(df, id, "heatmap")) \
        .option("checkpointLocation", "/tmp/checkpoint/heatmap") \
        .start()
    
    logger.info("Heatmap aggregation stream started")
    
    # ========================================
    # Wait for termination
    # ========================================
    logger.info("All streaming queries started. Waiting for termination...")
    
    try:
        faculty_query.awaitTermination()
        year_query.awaitTermination()
        heatmap_query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Streaming job interrupted by user")
    except Exception as e:
        logger.error(f"Streaming job error: {e}")
    finally:
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
