# app/services/query_tracker.py
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.config import Config
from app.models import QueryLog

logger = logging.getLogger(__name__)

# Thread pool for async MongoDB operations
_executor = ThreadPoolExecutor(max_workers=5)


class QueryTracker:
    """
    Service for tracking user queries to MongoDB
    Handles logging query metadata for analytics
    """
    
    def __init__(self):
        self.db_url = Config.DATABASE_URL
        self.db_name = "faiss_db"
        self.collection_name = "query_logs"
        self._client: Optional[MongoClient] = None
        self._ensure_indexes()
    
    def _get_client(self) -> MongoClient:
        """Get or create MongoDB client"""
        if self._client is None:
            self._client = MongoClient(self.db_url, serverSelectionTimeoutMS=5000)
        return self._client
    
    def _ensure_indexes(self):
        """Create indexes for efficient querying"""
        try:
            client = self._get_client()
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Create indexes
            indexes = [
                ("timestamp", ASCENDING),
                ("user_id", ASCENDING),
                ("faculty", ASCENDING),
                ("year", ASCENDING),
                ("file_type", ASCENDING),
                ("session_id", ASCENDING),
            ]
            
            for field, direction in indexes:
                collection.create_index([(field, direction)], background=True)
            
            # Compound index for analytics queries
            collection.create_index([
                ("timestamp", ASCENDING),
                ("faculty", ASCENDING),
                ("year", ASCENDING)
            ], background=True)
            
            logger.info(f"Indexes created for {self.collection_name}")
            
        except PyMongoError as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def _log_query_sync(self, query_log: Dict) -> bool:
        """Synchronous query logging (runs in thread pool)"""
        try:
            client = self._get_client()
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            # Insert query log
            result = collection.insert_one(query_log)
            
            logger.info(f"Query logged: {query_log['query_id'][:8]}... | "
                       f"Faculty: {query_log['faculty']} | "
                       f"Year: {query_log['year']} | "
                       f"Time: {query_log['response_time_ms']:.2f}ms")
            
            return result.inserted_id is not None
            
        except PyMongoError as e:
            logger.error(f"Failed to log query: {e}")
            return False
    
    async def log_query(
        self,
        user_id: str,
        session_id: str,
        query_text: str,
        faculty: str,
        year: str,
        file_type: str,
        response_time_ms: float,
        contexts_found: int,
        k: int = 5,
        similarity_threshold: float = 0.0,
        rewritten_query: Optional[str] = None,
        query_rewritten: bool = False,
        history_used: bool = False,
        history_count: int = 0
    ) -> Optional[str]:
        """
        Log a user query asynchronously
        
        Args:
            user_id: User ID from JWT token
            session_id: Chat session ID
            query_text: Original query text
            faculty: User's faculty
            year: User's academic year
            file_type: Document type queried
            response_time_ms: Response time in milliseconds
            contexts_found: Number of contexts found
            k: Number of results requested
            similarity_threshold: Similarity threshold used
            rewritten_query: Rewritten query (if applicable)
            query_rewritten: Whether query was rewritten
            history_used: Whether chat history was used
            history_count: Number of history messages used
        
        Returns:
            Query ID if successful, None otherwise
        """
        query_id = str(uuid.uuid4())
        
        # Create query log model
        query_log = QueryLog(
            query_id=query_id,
            user_id=user_id,
            session_id=session_id,
            query_text=query_text,
            rewritten_query=rewritten_query,
            faculty=faculty,
            year=year,
            file_type=file_type,
            k=k,
            response_time_ms=response_time_ms,
            contexts_found=contexts_found,
            similarity_threshold=similarity_threshold,
            query_rewritten=query_rewritten,
            history_used=history_used,
            history_count=history_count,
            timestamp=datetime.utcnow()
        )
        
        # Convert to dict for MongoDB
        query_dict = query_log.model_dump()
        
        # Log asynchronously (non-blocking)
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                _executor,
                self._log_query_sync,
                query_dict
            )
            
            if success:
                # Publish to Kafka (fire-and-forget, non-blocking)
                try:
                    from app.services.kafka_service import publish_query_event
                    asyncio.create_task(publish_query_event(query_dict))
                except Exception as kafka_err:
                    logger.warning(f"Kafka publish failed (non-critical): {kafka_err}")
                
                return query_id
            else:
                return None
                
        except Exception as e:
            logger.error(f"Async query logging failed: {e}")
            return None
    
    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")


# Singleton instance
_query_tracker: Optional[QueryTracker] = None


def get_query_tracker() -> QueryTracker:
    """Get singleton QueryTracker instance"""
    global _query_tracker
    if _query_tracker is None:
        _query_tracker = QueryTracker()
    return _query_tracker


def shutdown_query_tracker():
    """Shutdown QueryTracker"""
    global _query_tracker
    if _query_tracker:
        _query_tracker.close()
        _query_tracker = None
