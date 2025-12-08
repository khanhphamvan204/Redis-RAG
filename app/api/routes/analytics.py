# app/api/routes/analytics.py
"""
Analytics API Routes  
Metabase integration for real-time analytics visualization
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List
import logging
import os
from pymongo import MongoClient

router = APIRouter(prefix="/analytics", tags=["analytics"])
logger = logging.getLogger(__name__)

# Metabase configuration
METABASE_BASE_URL = os.getenv("METABASE_URL", "http://localhost:8090")

# MongoDB connection for health checks
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "faiss_db")


@router.get("/health")
async def check_analytics_health():
    """
    Check MongoDB connection health for Charts
    
    Returns:
        {
            "status": "healthy" | "unhealthy",
            "mongodb_uri": "...",
            "collections": [...],
            "charts_url": "..."
        }
    """
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[MONGODB_DATABASE]
        
        # Check if analytics collections exist
        collections = db.list_collection_names()
        analytics_collections = [
            col for col in collections 
            if col.startswith("query_analytics")
        ]
        
        # Get sample counts
        collection_stats = {}
        for col_name in analytics_collections:
            count = db[col_name].count_documents({})
            collection_stats[col_name] = count
        
        client.close()
        
        return {
            "status": "healthy",
            "mongodb_uri": MONGODB_URI,
            "database": MONGODB_DATABASE,
            "analytics_collections": analytics_collections,
            "collection_stats": collection_stats,
            "charts_url": METABASE_BASE_URL
        }
        
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        return {
            "status": "unhealthy",
            "mongodb_uri": MONGODB_URI,
            "error": str(e),
            "charts_url": METABASE_BASE_URL
        }


@router.get("/charts/embed-info")
async def get_charts_embed_info():
    """
    Get Metabase embedding information
    
    Returns chart configuration that frontend needs to embed dashboards.
    After creating questions/dashboards in Metabase UI, update these URLs.
    """
    
    # TODO: Update these URLs after creating charts in Metabase UI
    # Access Metabase at http://localhost:8090 to create questions/dashboards
    
    charts_info = {
        "charts_base_url": METABASE_BASE_URL,
        "instructions": {
            "step_1": "Access Metabase at http://localhost:8090",
            "step_2": "Complete initial setup wizard (create admin account)",
            "step_3": "Add database: MongoDB at host.docker.internal:27017",
            "step_4": "Select database: faiss_db",
            "step_5": "Create questions (charts) from collections: query_analytics_by_faculty, query_analytics_by_year, query_analytics_heatmap",
            "step_6": "Create dashboard and add questions",
            "step_7": "Enable public sharing for dashboard and individual questions",
            "step_8": "Copy public URLs and update the embed_urls below"
        },
        "embed_urls": {
            "overview_dashboard": "http://localhost:8090/public/dashboard/02d0ec6b-cdce-4947-a0dd-4cb216b14990",
            "faculty_chart": f"{METABASE_BASE_URL}/public/question/YOUR_FACULTY_QUESTION_HASH?bordered=false&titled=false",
            "year_chart": f"{METABASE_BASE_URL}/public/question/YOUR_YEAR_QUESTION_HASH?bordered=false&titled=false",
            "heatmap_chart": f"{METABASE_BASE_URL}/public/question/YOUR_HEATMAP_QUESTION_HASH?bordered=false&titled=false"
        },
        "available_collections": [
            "query_analytics_by_faculty",
            "query_analytics_by_year", 
            "query_analytics_heatmap"
        ]
    }
    
    return charts_info


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
