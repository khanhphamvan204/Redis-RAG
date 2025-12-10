# app/services/redis_analytics_service.py
"""
Redis Analytics Service
Query analytics data from Redis with time-based filtering
"""
import logging
import json
import redis
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis-stack-db") 
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Initialize Redis client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5
)


def _scan_keys_by_pattern(pattern: str) -> List[str]:
    """Helper to scan all keys matching a pattern"""
    keys = []
    cursor = 0
    while True:
        cursor, partial_keys = redis_client.scan(
            cursor=cursor,
            match=pattern,
            count=100
        )
        keys.extend(partial_keys)
        if cursor == 0:
            break
    return keys


def _filter_by_time_and_aggregate(keys: List[str], days: int, key_parts_count: int, aggregator_fn):
    """Helper to filter keys by time and aggregate data"""
    cutoff_time = datetime.now() - timedelta(days=days)
    aggregated_data = {}
    
    for key in keys:
        try:
            parts = key.split(":")
            if len(parts) >= key_parts_count:
                # Last part(s) is timestamp
                timestamp_str = ":".join(parts[key_parts_count-1:])
                timestamp = datetime.fromisoformat(timestamp_str)
                
                if timestamp >= cutoff_time:
                    data_json = redis_client.get(key)
                    if data_json:
                        data = json.loads(data_json)
                        aggregator_fn(aggregated_data, parts, data, timestamp_str)
        except Exception as e:
            logger.warning(f"Error processing key {key}: {e}")
            continue
    
    return aggregated_data


## LEGACY FUNCTIONS (keeping as is)

def get_faculty_analytics(days: int = 30) -> List[Dict]:
    """Get faculty analytics from Redis for the last N days"""
    try:
        keys = _scan_keys_by_pattern("analytics:faculty:*")
        logger.info(f"Found {len(keys)} faculty analytics keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            faculty = parts[2]
            if faculty not in agg_data:
                agg_data[faculty] = {
                    "faculty": faculty,
                    "query_count": 0,
                    "unique_users": 0,
                    "total_response_time": 0,
                    "count": 0,
                    "last_updated": timestamp_str
                }
            
            agg_data[faculty]["query_count"] += data.get("query_count", 0)
            agg_data[faculty]["unique_users"] += data.get("unique_users", 0)
            agg_data[faculty]["total_response_time"] += data.get("avg_response_time", 0) * data.get("query_count", 1)
            agg_data[faculty]["count"] += data.get("query_count", 1)
            
            if timestamp_str > agg_data[faculty]["last_updated"]:
                agg_data[faculty]["last_updated"] = timestamp_str
        
        faculty_data = _filter_by_time_and_aggregate(keys, days, 4, aggregator)
        
        # Calculate averages
        result = []
        for faculty, data in faculty_data.items():
            if data["count"] > 0:
                data["avg_response_time"] = data["total_response_time"] / data["count"]
            del data["total_response_time"]
            del data["count"]
            result.append(data)
        
        logger.info(f"Returning {len(result)} faculty analytics records")
        return result
        
    except Exception as e:
        logger.error(f"Error getting faculty analytics: {e}", exc_info=True)
        return []


def get_year_analytics(days: int = 30) -> List[Dict]:
    """Get year analytics from Redis for the last N days"""
    try:
        keys = _scan_keys_by_pattern("analytics:year:*")
        logger.info(f"Found {len(keys)} year analytics keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            year = parts[2]
            if year not in agg_data:
                agg_data[year] = {
                    "year": year,
                    "query_count": 0,
                    "unique_users": 0,
                    "total_response_time": 0,
                    "count": 0,
                    "last_updated": timestamp_str
                }
            
            agg_data[year]["query_count"] += data.get("query_count", 0)
            agg_data[year]["unique_users"] += data.get("unique_users", 0)
            agg_data[year]["total_response_time"] += data.get("avg_response_time", 0) * data.get("query_count", 1)
            agg_data[year]["count"] += data.get("query_count", 1)
            
            if timestamp_str > agg_data[year]["last_updated"]:
                agg_data[year]["last_updated"] = timestamp_str
        
        year_data = _filter_by_time_and_aggregate(keys, days, 4, aggregator)
        
        # Calculate averages
        result = []
        for year, data in year_data.items():
            if data["count"] > 0:
                data["avg_response_time"] = data["total_response_time"] / data["count"]
            del data["total_response_time"]
            del data["count"]
            result.append(data)
        
        logger.info(f"Returning {len(result)} year analytics records")
        return result
        
    except Exception as e:
        logger.error(f"Error getting year analytics: {e}", exc_info=True)
        return []


def get_heatmap_analytics(days: int = 30) -> List[Dict]:
    """Get heatmap analytics from Redis for the last N days"""
    try:
        keys = _scan_keys_by_pattern("analytics:heatmap:*:*")
        logger.info(f"Found {len(keys)} heatmap analytics keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            faculty = parts[2]
            year = parts[3]
            cell_key = f"{faculty}:{year}"
            
            if cell_key not in agg_data:
                agg_data[cell_key] = {
                    "faculty": faculty,
                    "year": year,
                    "query_count": 0,
                    "total_response_time": 0,
                    "count": 0,
                    "last_updated": timestamp_str
                }
            
            agg_data[cell_key]["query_count"] += data.get("query_count", 0)
            agg_data[cell_key]["total_response_time"] += data.get("avg_response_time", 0) * data.get("query_count", 1)
            agg_data[cell_key]["count"] += data.get("query_count", 1)
            
            if timestamp_str > agg_data[cell_key]["last_updated"]:
                agg_data[cell_key]["last_updated"] = timestamp_str
        
        heatmap_data = _filter_by_time_and_aggregate(keys, days, 5, aggregator)
        
        # Calculate averages
        result = []
        for cell_key, data in heatmap_data.items():
            if data["count"] > 0:
                data["avg_response_time"] = data["total_response_time"] / data["count"]
            del data["total_response_time"]
            del data["count"]
            result.append(data)
        
        logger.info(f"Returning {len(result)} heatmap analytics records")
        return result
        
    except Exception as e:
        logger.error(f"Error getting heatmap analytics: {e}", exc_info=True)
        return []


def get_latest_update() -> Optional[str]:
    """Get the timestamp of the latest analytics update"""
    try:
        timestamp = redis_client.get("analytics:latest")
        return timestamp
    except Exception as e:
        logger.error(f"Error getting latest update timestamp: {e}")
        return None


def get_popular_questions(days: int = 30, limit: int = 10) -> List[Dict]:
    """Get popular questions from Redis"""
    try:
        keys = _scan_keys_by_pattern("analytics:popular_queries:*")
        logger.info(f"Found {len(keys)} popular question keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            query_hash = parts[2]
            query_text = data.get("query_text", "")
            
            if query_hash not in agg_data:
                agg_data[query_hash] = {
                    "query_text": query_text,
                    "total_count": 0,
                    "unique_users": 0
                }
            
            agg_data[query_hash]["total_count"] += data.get("query_count", 0)
            agg_data[query_hash]["unique_users"] += data.get("unique_users", 0)
        
        questions_data = _filter_by_time_and_aggregate(keys, days, 4, aggregator)
        
        # Sort by count and limit
        result = sorted(questions_data.values(), key=lambda x: x["total_count"], reverse=True)[:limit]
        
        logger.info(f"Returning {len(result)} popular questions")
        return result
        
    except Exception as e:
        logger.error(f"Error getting popular questions: {e}", exc_info=True)
        return []


def get_department_name_analytics(days: int = 30) -> List[Dict]:
    """
    Get department analytics from Redis for pie chart
    Uses department_name as key
    """
    try:
        keys = _scan_keys_by_pattern("analytics:department:*")
        logger.info(f"Found {len(keys)} department analytics keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            dept_name = parts[2]
            if dept_name not in agg_data:
                agg_data[dept_name] = {
                    "department_name": dept_name,
                    "query_count": 0,
                    "unique_users": 0,
                    "success_count": 0,
                    "total_response_time": 0,
                    "count": 0
                }
            
            agg_data[dept_name]["query_count"] += data.get("query_count", 0)
            agg_data[dept_name]["unique_users"] += data.get("unique_users", 0)
            agg_data[dept_name]["success_count"] += data.get("success_count", 0)
            agg_data[dept_name]["total_response_time"] += data.get("avg_response_time", 0) * data.get("query_count", 1)
            agg_data[dept_name]["count"] += data.get("query_count", 1)
        
        dept_data = _filter_by_time_and_aggregate(keys, days, 4, aggregator)
        
        # Calculate totals for percentage
        total_queries = sum(d["query_count"] for d in dept_data.values())
        
        # Calculate averages and percentages
        result = []
        for dept_name, data in dept_data.items():
            if data["count"] > 0:
                data["avg_response_time"] = round(data["total_response_time"] / data["count"], 2)
                data["percentage"] = round((data["query_count"] / total_queries * 100), 2) if total_queries > 0 else 0
            del data["total_response_time"]
            del data["count"]
            result.append(data)
        
        # Sort by query count
        result.sort(key=lambda x: x["query_count"], reverse=True)
        
        logger.info(f"Returning {len(result)} department analytics records")
        return result
        
    except Exception as e:
        logger.error(f"Error getting department analytics: {e}", exc_info=True)
        return []


def subscribe_to_updates():
    """Subscribe to Redis Pub/Sub for analytics updates"""
    try:
        pubsub = redis_client.pubsub()
        pubsub.subscribe("analytics:updates")
        logger.info("Subscribed to analytics:updates channel")
        return pubsub
    except Exception as e:
        logger.error(f"Error subscribing to updates: {e}")
        return None



## NEW ANALYTICS FUNCTIONS

def get_overall_summary(days: int = 30) -> Dict:
    """
    Get overall summary statistics for dashboard cards
    
    Args:
        days: Number of days to look back
        
    Returns:
        Dict with total_queries, success_count, failure_count, success_rate, avg_response_time
    """
    try:
        keys = _scan_keys_by_pattern("analytics:overall:*")
        logger.info(f"Found {len(keys)} overall summary keys")
        
        total_queries = 0
        success_count = 0
        failure_count = 0
        unique_users = set()
        response_times = []
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        for key in keys:
            try:
                parts = key.split(":")
                if len(parts) >= 3:
                    timestamp_str = ":".join(parts[2:])
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    if timestamp >= cutoff_time:
                        data_json = redis_client.get(key)
                        if data_json:
                            data = json.loads(data_json)
                            total_queries += data.get("total_queries", 0)
                            success_count += data.get("success_count", 0)
                            failure_count += data.get("failure_count", 0)
                            
                            avg_rt = data.get("avg_response_time", 0)
                            count = data.get("total_queries", 1)
                            if avg_rt > 0:
                                response_times.extend([avg_rt] * count)
            except Exception as e:
                logger.warning(f"Error processing key {key}: {e}")
                continue
        
        # Calculate success rate
        success_rate = round((success_count / total_queries * 100), 2) if total_queries > 0 else 0
        avg_response_time = round(sum(response_times) / len(response_times), 2) if response_times else 0
        
        result = {
            "total_queries": total_queries,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time
        }
        
        logger.info(f"Returning overall summary: {total_queries} queries, {success_rate}% success rate")
        return result
        
    except Exception as e:
        logger.error(f"Error getting overall summary: {e}", exc_info=True)
        return {
            "total_queries": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0,
            "avg_response_time": 0
        }


def get_student_year_analytics(days: int = 30) -> List[Dict]:
    """
    Get student year analytics from Redis
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of analytics by student year (1-6)
    """
    try:
        keys = _scan_keys_by_pattern("analytics:student_year:*")
        logger.info(f"Found {len(keys)} student year analytics keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            years = parts[2]
            if years not in agg_data:
                agg_data[years] = {
                    "years": int(years) if years.isdigit() else years,
                    "query_count": 0,
                    "unique_users": 0,
                    "total_response_time": 0,
                    "total_contexts": 0,
                    "count": 0
                }
            
            agg_data[years]["query_count"] += data.get("query_count", 0)
            agg_data[years]["unique_users"] += data.get("unique_users", 0)
            agg_data[years]["total_response_time"] += data.get("avg_response_time", 0) * data.get("query_count", 1)
            agg_data[years]["total_contexts"] += data.get("avg_contexts_found", 0) * data.get("query_count", 1)
            agg_data[years]["count"] += data.get("query_count", 1)
        
        year_data = _filter_by_time_and_aggregate(keys, days, 4, aggregator)
        
        # Calculate averages and sort
        result = []
        for years, data in year_data.items():
            if data["count"] > 0:
                data["avg_response_time"] = round(data["total_response_time"] / data["count"], 2)
                data["avg_contexts_found"] = round(data["total_contexts"] / data["count"], 2)
            del data["total_response_time"]
            del data["total_contexts"]
            del data["count"]
            result.append(data)
        
        # Sort by year
        result.sort(key=lambda x: x["years"] if isinstance(x["years"], int) else 999)
        
        logger.info(f"Returning {len(result)} student year analytics records")
        return result
        
    except Exception as e:
        logger.error(f"Error getting student year analytics: {e}", exc_info=True)
        return []


def get_popular_questions_by_year(year: Optional[int] = None, days: int = 30, limit: int = 10) -> Dict:
    """
    Get popular questions grouped by student year
    
    Args:
        year: Specific year to filter (None = all years)
        days: Number of days to look back
        limit: Max questions per year
        
    Returns:
        Dict with year as key and list of popular questions as value
    """
    try:
        pattern = f"analytics:popular_by_year:{year}:*" if year else "analytics:popular_by_year:*"
        keys = _scan_keys_by_pattern(pattern)
        logger.info(f"Found {len(keys)} popular by year keys")
        
        # Use cutoff_time to filter
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Group by year
        by_year = defaultdict(lambda: defaultdict(lambda: {"query_text": "", "total_count": 0, "unique_users": 0}))
        
        for key in keys:
            try:
                parts = key.split(":")
                if len(parts) >= 5:
                    # parts: [analytics, popular_by_year, year, query_hash, timestamp]
                    timestamp_str = parts[4]
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    if timestamp >= cutoff_time:
                        data_json = redis_client.get(key)
                        if data_json:
                            data = json.loads(data_json)
                            years = parts[2]
                            query_hash = parts[3]
                            query_text = data.get("query_text", "")
                            
                            by_year[years][query_hash]["query_text"] = query_text
                            by_year[years][query_hash]["total_count"] += data.get("query_count", 0)
                            by_year[years][query_hash]["unique_users"] += data.get("unique_users", 0)
            except Exception as e:
                logger.warning(f"Error processing key {key}: {e}")
                continue
        
        # Sort and limit each year
        result = {}
        for yr, questions in by_year.items():
            sorted_questions = sorted(questions.values(), key=lambda x: x["total_count"], reverse=True)[:limit]
            result[yr] = sorted_questions
        
        logger.info(f"Returning popular questions for {len(result)} years")
        return result
        
    except Exception as e:
        logger.error(f"Error getting popular questions by year: {e}", exc_info=True)
        return {}


def get_user_type_distribution(days: int = 30) -> Dict:
    """
    Get user type distribution (% students vs % teachers)
    
    Args:
        days: Number of days to look back
        
    Returns:
        Dict with user type stats and percentages
    """
    try:
        keys = _scan_keys_by_pattern("analytics:user_type:*")
        logger.info(f"Found {len(keys)} user type analytics keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            user_type = parts[2]
            if user_type not in agg_data:
                agg_data[user_type] = {
                    "user_type": user_type,
                    "query_count": 0,
                    "unique_users": 0
                }
            
            agg_data[user_type]["query_count"] += data.get("query_count", 0)
            agg_data[user_type]["unique_users"] += data.get("unique_users", 0)
        
        type_data = _filter_by_time_and_aggregate(keys, days, 4, aggregator)
        
        # Calculate percentages
        total_queries = sum(d["query_count"] for d in type_data.values())
        result = []
        
        for user_type, data in type_data.items():
            data["percentage"] = round((data["query_count"] / total_queries * 100), 2) if total_queries > 0 else 0
            result.append(data)
        
        logger.info(f"Returning {len(result)} user type records")
        return {
            "distribution": result,
            "total_queries": total_queries
        }
        
    except Exception as e:
        logger.error(f"Error getting user type distribution: {e}", exc_info=True)
        return {"distribution": [], "total_queries": 0}


def get_success_rate_analytics(days: int = 7) -> List[Dict]:
    """
    Get success rate analytics
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of success rate records
    """
    try:
        keys = _scan_keys_by_pattern("analytics:success_rate:*")
        logger.info(f"Found {len(keys)} success rate keys")
        
        total_success = 0
        total_failure = 0
        total_queries = 0
        records = []
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        for key in keys:
            try:
                parts = key.split(":")
                if len(parts) >= 3:
                    timestamp_str = ":".join(parts[2:])
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    if timestamp >= cutoff_time:
                        data_json = redis_client.get(key)
                        if data_json:
                            data = json.loads(data_json)
                            total_queries += data.get("total_queries", 0)
                            total_success += data.get("success_count", 0)
                            total_failure += data.get("failure_count", 0)
                            records.append({
                                "timestamp": timestamp_str,
                                **data
                            })
            except Exception as e:
                logger.warning(f"Error processing key {key}: {e}")
                continue
        
        # Calculate overall success rate
        success_rate = round((total_success / total_queries * 100), 2) if total_queries > 0 else 0
        
        logger.info(f"Returning success rate: {success_rate}%")
        return {
            "success_rate": success_rate,
            "total_queries": total_queries,
            "total_success": total_success,
            "total_failure": total_failure,
            "records": sorted(records, key=lambda x: x["timestamp"], reverse=True)[:20]
        }
        
    except Exception as e:
        logger.error(f"Error getting success rate analytics: {e}", exc_info=True)
        return {"success_rate": 0, "total_queries": 0, "total_success": 0, "total_failure": 0, "records": []}


def get_response_time_analytics(days: int = 7) -> List[Dict]:
    """
    Get response time analytics by user type
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of response time records by user type
    """
    try:
        keys = _scan_keys_by_pattern("analytics:response_time:*")
        logger.info(f"Found {len(keys)} response time keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            user_type = parts[2]
            if user_type not in agg_data:
                agg_data[user_type] = {
                    "user_type": user_type,
                    "avg_time": [],
                    "median_time": [],
                    "p95_time": [],
                    "count": 0
                }
            
            agg_data[user_type]["avg_time"].append(data.get("avg_time", 0))
            agg_data[user_type]["median_time"].append(data.get("median_time", 0))
            agg_data[user_type]["p95_time"].append(data.get("p95_time", 0))
            agg_data[user_type]["count"] += data.get("query_count", 0)
        
        time_data = _filter_by_time_and_aggregate(keys, days, 4, aggregator)
        
        # Calculate averages
        result = []
        for user_type, data in time_data.items():
            if data["count"] > 0:
                result.append({
                    "user_type": user_type,
                    "avg_time": round(sum(data["avg_time"]) / len(data["avg_time"]), 2),
                    "median_time": round(sum(data["median_time"]) / len(data["median_time"]), 2),
                    "p95_time": round(sum(data["p95_time"]) / len(data["p95_time"]), 2),
                    "query_count": data["count"]
                })
        
        logger.info(f"Returning {len(result)} response time records")
        return result
        
    except Exception as e:
        logger.error(f"Error getting response time analytics: {e}", exc_info=True)
        return []


def get_timeseries_analytics(granularity: str = "hour", days: int = 7) -> List[Dict]:
    """
    Get time-series analytics
    
    Args:
        granularity: 'minute', 'hour', or 'day'
        days: Number of days to look back
        
    Returns:
        List of time-series data points
    """
    try:
        pattern = f"analytics:timeseries:{granularity}:*"
        keys = _scan_keys_by_pattern(pattern)
        logger.info(f"Found {len(keys)} timeseries keys for {granularity}")
        
        records = []
        cutoff_time = datetime.now() - timedelta(days=days)
        
        for key in keys:
            try:
                parts = key.split(":")
                if len(parts) >= 4:
                    timestamp_str = ":".join(parts[3:])
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    if timestamp >= cutoff_time:
                        data_json = redis_client.get(key)
                        if data_json:
                            data = json.loads(data_json)
                            records.append({
                                "timestamp": timestamp_str,
                                "query_count": data.get("query_count", 0),
                                "window_start": data.get("window_start"),
                                "window_end": data.get("window_end")
                            })
            except Exception as e:
                logger.warning(f"Error processing key {key}: {e}")
                continue
        
        # Sort by timestamp
        records.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"Returning {len(records)} timeseries records")
        return records
        
    except Exception as e:
        logger.error(f"Error getting timeseries analytics: {e}", exc_info=True)
        return []


def get_hourly_heatmap(days: int = 7) -> List[Dict]:
    """
    Get hourly heatmap data (queries by hour of day)
    
    Args:
        days: Number of days to look back
        
    Returns:
        List of hourly heatmap data
    """
    try:
        keys = _scan_keys_by_pattern("analytics:heatmap:hourly:*")
        logger.info(f"Found {len(keys)} hourly heatmap keys")
        
        def aggregator(agg_data, parts, data, timestamp_str):
            hour = parts[3]
            if hour not in agg_data:
                agg_data[hour] = {
                    "hour_of_day": int(hour) if hour.isdigit() else hour,
                    "query_count": 0
                }
            
            agg_data[hour]["query_count"] += data.get("query_count", 0)
        
        hour_data = _filter_by_time_and_aggregate(keys, days, 5, aggregator)
        
        # Sort by hour
        result = sorted(hour_data.values(), key=lambda x: x["hour_of_day"] if isinstance(x["hour_of_day"], int) else 99)
        
        logger.info(f"Returning {len(result)} hourly heatmap records")
        return result
        
    except Exception as e:
        logger.error(f"Error getting hourly heatmap: {e}", exc_info=True)
        return []
