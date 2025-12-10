# app/api/routes/analytics.py
"""
Analytics API Routes  
Redis-based real-time analytics
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List
import logging
import os

router = APIRouter(prefix="/analytics", tags=["analytics"])
logger = logging.getLogger(__name__)

# Import Redis analytics service
from app.services.redis_analytics_service import (
    get_faculty_analytics,
    get_year_analytics,
    get_heatmap_analytics,
    get_latest_update,
    get_popular_questions,
    get_department_name_analytics,
    get_overall_summary,
    # NEW FUNCTIONS
    get_student_year_analytics,
    get_popular_questions_by_year,
    get_user_type_distribution,
    get_success_rate_analytics,
    get_response_time_analytics,
    get_timeseries_analytics,
    get_hourly_heatmap
)



@router.get("/health")
async def check_analytics_health():
    """
    Check Redis connection health for Analytics
    
    Returns:
        {
            "status": "healthy" | "unhealthy",
            "redis_host": "...",
            "latest_update": "..."
        }
    """
    try:
        latest = get_latest_update()
        
        return {
            "status": "healthy" if latest else "degraded",
            "redis_host": os.getenv("REDIS_HOST", "redis-stack-db"),
            "redis_port": os.getenv("REDIS_PORT", "6379"),
            "latest_update": latest,
            "message": "Analytics data available" if latest else "No analytics data yet"
        }
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/redis/faculty")
async def get_faculty_analytics_endpoint(days: int = 30):
    """
    Get faculty analytics from Redis
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        List of faculty analytics
    """
    try:
        data = get_faculty_analytics(days=days)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting faculty analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/year")
async def get_year_analytics_endpoint(days: int = 30):
    """
    Get year analytics from Redis
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        List of year analytics
    """
    try:
        data = get_year_analytics(days=days)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting year analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/heatmap")
async def get_heatmap_analytics_endpoint(days: int = 30):
    """
    Get heatmap analytics from Redis
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        List of heatmap analytics
    """
    try:
        data = get_heatmap_analytics(days=days)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting heatmap analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/latest")
async def get_latest_update_endpoint():
    """
    Get the timestamp of the latest analytics update
    
    Returns:
        Latest update timestamp
    """
    try:
        timestamp = get_latest_update()
        return {
            "status": "success",
            "latest_update": timestamp
        }
    except Exception as e:
        logger.error(f"Error getting latest update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/popular-questions")
async def get_popular_questions_endpoint(days: int = 30, limit: int = 10):
    """
    Get most popular questions from Redis
    
    Args:
        days: Number of days to look back (default: 30)
        limit: Maximum number of results (default: 10)
        
    Returns:
        List of popular questions sorted by frequency
    """
    try:
        data = get_popular_questions(days=days, limit=limit)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "days": days,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting popular questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/department")
async def get_department_analytics_endpoint(days: int = 30):
    """
    Get department analytics from Redis for pie chart
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        List of department analytics with percentage
    """
    try:
        data = get_department_name_analytics(days=days)
        total_queries = sum(d.get("query_count", 0) for d in data)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "total_queries": total_queries,
            "days": days,
            "chart_type": "pie"
        }
    except Exception as e:
        logger.error(f"Error getting department analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/overall-summary")
async def get_overall_summary_endpoint(days: int = 30):
    """
    Get overall summary statistics for dashboard cards
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        Dict with total_queries, success_count, success_rate, avg_response_time
    """
    try:
        data = get_overall_summary(days=days)
        return {
            "status": "success",
            **data,
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting overall summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


## NEW ANALYTICS ENDPOINTS

@router.get("/redis/student-year")
async def get_student_year_analytics_endpoint(days: int = 30):
    """
    Get student year analytics from Redis
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        List of student year analytics (years 1-6)
    """
    try:
        data = get_student_year_analytics(days=days)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting student year analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/popular-by-year")
async def get_popular_by_year_endpoint(year: int = None, days: int = 30, limit: int = 10):
    """
    Get popular questions grouped by student year
    
    Args:
        year: Specific year to filter (None = all years)
        days: Number of days to look back (default: 30)
        limit: Max questions per year (default: 10)
        
    Returns:
        Dict with year as key and list of popular questions as value
    """
    try:
        data = get_popular_questions_by_year(year=year, days=days, limit=limit)
        return {
            "status": "success",
            "data": data,
            "years": list(data.keys()),
            "days": days,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting popular by year: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/user-type")
async def get_user_type_endpoint(days: int = 30):
    """
    Get user type distribution (% students vs % teachers)
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        User type distribution with percentages
    """
    try:
        data = get_user_type_distribution(days=days)
        return {
            "status": "success",
            **data,
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting user type distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/success-rate")
async def get_success_rate_endpoint(days: int = 7):
    """
    Get success rate analytics
    
    Args:
        days: Number of days to look back (default: 7)
        
    Returns:
        Success rate statistics
    """
    try:
        data = get_success_rate_analytics(days=days)
        return {
            "status": "success",
            **data,
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting success rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/response-time")
async def get_response_time_endpoint(days: int = 7):
    """
    Get response time analytics by user type
    
    Args:
        days: Number of days to look back (default: 7)
        
    Returns:
        Response time statistics (avg, median, p95)
    """
    try:
        data = get_response_time_analytics(days=days)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting response time analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/timeseries")
async def get_timeseries_endpoint(granularity: str = "hour", days: int = 7):
    """
    Get time-series analytics
    
    Args:
        granularity: 'minute', 'hour', or 'day' (default: 'hour')
        days: Number of days to look back (default: 7)
        
    Returns:
        Time-series data points
    """
    try:
        if granularity not in ["minute", "hour", "day"]:
            raise HTTPException(status_code=400, detail="granularity must be 'minute', 'hour', or 'day'")
        
        data = get_timeseries_analytics(granularity=granularity, days=days)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "granularity": granularity,
            "days": days
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting timeseries analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/hourly-heatmap")
async def get_hourly_heatmap_endpoint(days: int = 7):
    """
    Get hourly heatmap data (queries by hour of day 0-23)
    
    Args:
        days: Number of days to look back (default: 7)
        
    Returns:
        Hourly heatmap data
    """
    try:
        data = get_hourly_heatmap(days=days)
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "days": days
        }
    except Exception as e:
        logger.error(f"Error getting hourly heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/stats")
async def get_collections_stats():
    """
    Get statistics about analytics collections
    Useful for debugging and monitoring
    """
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[MONGODB_DATABASE]
        
        stats = {}
        
        # Check each analytics collection
        analytics_collections = [
            "query_analytics_by_faculty",
            "query_analytics_by_year",
            "query_analytics_heatmap"
        ]
        
        for col_name in analytics_collections:
            if col_name in db.list_collection_names():
                collection = db[col_name]
                count = collection.count_documents({})
                
                # Get sample document
                sample = collection.find_one()
                
                stats[col_name] = {
                    "exists": True,
                    "document_count": count,
                    "sample_fields": list(sample.keys()) if sample else [],
                    "last_updated": sample.get("timestamp") if sample else None
                }
            else:
                stats[col_name] = {
                    "exists": False,
                    "document_count": 0
                }
        
        client.close()
        
        return {
            "status": "success",
            "collections": stats,
            "mongodb_uri": MONGODB_URI,
            "database": MONGODB_DATABASE
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection stats: {str(e)}"
        )
