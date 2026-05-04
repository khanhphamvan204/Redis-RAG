# app/services/cache_service.py
"""
Semantic Cache Service for RAG System

Provides two types of caching:
1. Embedding Cache: Cache query embeddings to avoid re-computation
2. LLM Semantic Cache: Cache LLM responses for semantically similar queries

Both caches can be enabled/disabled via environment variables.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
import redis

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

class CacheConfig:
    """Cache configuration from environment variables"""
    
    # Embedding Cache Settings
    EMBEDDING_CACHE_ENABLED = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "3600"))  # 1 hour
    EMBEDDING_CACHE_NAME = os.getenv("EMBEDDING_CACHE_NAME", "embed_cache")
    
    # LLM Cache Settings
    LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
    LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "86400"))  # 24 hours
    LLM_CACHE_DISTANCE_THRESHOLD = float(os.getenv("LLM_CACHE_DISTANCE_THRESHOLD", "0.1"))  # Balanced: 0.1 = 90% similarity required
    LLM_CACHE_NAME = os.getenv("LLM_CACHE_NAME", "llm_cache")
    
    # Redis Connection
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis connection URL"""
        return f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
    
    @classmethod
    def log_config(cls):
        """Log cache configuration"""
        logger.info("Cache Configuration:")
        logger.info(f"  Embedding Cache: {'ENABLED' if cls.EMBEDDING_CACHE_ENABLED else 'DISABLED'}")
        if cls.EMBEDDING_CACHE_ENABLED:
            logger.info(f"    - TTL: {cls.EMBEDDING_CACHE_TTL}s")
            logger.info(f"    - Name: {cls.EMBEDDING_CACHE_NAME}")
        logger.info(f"  LLM Cache: {'ENABLED' if cls.LLM_CACHE_ENABLED else 'DISABLED'}")
        if cls.LLM_CACHE_ENABLED:
            logger.info(f"    - TTL: {cls.LLM_CACHE_TTL}s")
            logger.info(f"    - Distance Threshold: {cls.LLM_CACHE_DISTANCE_THRESHOLD}")
            logger.info(f"    - Name: {cls.LLM_CACHE_NAME}")


# ============================================
# EMBEDDING CACHE MANAGER
# ============================================

_embedding_cache_instance = None
_embedding_cache_lock = None

try:
    import threading
    _embedding_cache_lock = threading.Lock()
except ImportError:
    class DummyLock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    _embedding_cache_lock = DummyLock()


class EmbeddingsCacheManager:
    """
    Manager for embedding cache using RedisVL EmbeddingsCache
    
    Caches embeddings to avoid re-computation for identical queries.
    Thread-safe singleton pattern.
    """
    
    def __init__(self):
        if not CacheConfig.EMBEDDING_CACHE_ENABLED:
            logger.info("Embedding cache is DISABLED")
            self._cache = None
            return
        
        try:
            logger.info("Initializing Embedding Cache...")
            
            self._cache = EmbeddingsCache(
                name=CacheConfig.EMBEDDING_CACHE_NAME,
                redis_url=CacheConfig.get_redis_url(),
                ttl=CacheConfig.EMBEDDING_CACHE_TTL
            )
            
            logger.info(f"Embedding cache initialized: {CacheConfig.EMBEDDING_CACHE_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding cache: {e}")
            self._cache = None
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get cached embedding for text
        
        Args:
            text: Query text
            model_name: Embedding model name
            
        Returns:
            Cached embedding vector or None if not found
        """
        if not CacheConfig.EMBEDDING_CACHE_ENABLED or self._cache is None:
            return None
        
        try:
            result = self._cache.get(text=text, model_name=model_name)
            if result:
                logger.debug(f"Embedding cache HIT for: {text[:50]}...")
                return result.get('embedding')
            else:
                logger.debug(f"Embedding cache MISS for: {text[:50]}...")
                return None
        except Exception as e:
            logger.warning(f"Embedding cache get error: {e}")
            return None
    
    def store(self, text: str, model_name: str, embedding: List[float], metadata: Optional[Dict] = None) -> bool:
        """
        Store embedding in cache
        
        Args:
            text: Query text
            model_name: Embedding model name
            embedding: Embedding vector
            metadata: Optional metadata
            
        Returns:
            True if stored successfully
        """
        if not CacheConfig.EMBEDDING_CACHE_ENABLED or self._cache is None:
            return False
        
        try:
            key = self._cache.set(
                text=text,
                model_name=model_name,
                embedding=embedding,
                metadata=metadata or {}
            )
            logger.debug(f"Stored embedding in cache: {key[:20]}...")
            return True
        except Exception as e:
            logger.warning(f"Embedding cache store error: {e}")
            return False
    
    def clear(self):
        """Clear all cached embeddings"""
        if not CacheConfig.EMBEDDING_CACHE_ENABLED or self._cache is None:
            return
        
        try:
            # EmbeddingsCache doesn't have a clear method, so we use Redis directly
            client = redis.Redis(
                host=CacheConfig.REDIS_HOST,
                port=CacheConfig.REDIS_PORT,
                db=CacheConfig.REDIS_DB,
                decode_responses=True
            )
            pattern = f"{CacheConfig.EMBEDDING_CACHE_NAME}:*"
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
                logger.info(f"Cleared {len(keys)} embedding cache entries")
        except Exception as e:
            logger.error(f"Failed to clear embedding cache: {e}")


def get_embedding_cache() -> EmbeddingsCacheManager:
    """Get singleton embedding cache instance"""
    global _embedding_cache_instance
    
    if _embedding_cache_instance is None:
        with _embedding_cache_lock:
            if _embedding_cache_instance is None:
                _embedding_cache_instance = EmbeddingsCacheManager()
    
    return _embedding_cache_instance


# ============================================
# LLM SEMANTIC CACHE MANAGER
# ============================================

_llm_cache_instance = None
_llm_cache_lock = None

try:
    import threading
    _llm_cache_lock = threading.Lock()
except ImportError:
    class DummyLock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    _llm_cache_lock = DummyLock()


class LLMSemanticCacheManager:
    """
    Manager for LLM semantic cache using RedisVL SemanticCache
    
    Caches LLM responses and returns them for semantically similar prompts.
    Thread-safe singleton pattern.
    """
    
    def __init__(self):
        if not CacheConfig.LLM_CACHE_ENABLED:
            logger.info("LLM cache is DISABLED")
            self._cache = None
            self._vectorizer = None
            return
        
        try:
            logger.info("Initializing LLM Semantic Cache...")
            
            # Create vectorizer for semantic matching
            # Use the same model as the main embedding model
            self._vectorizer = HFTextVectorizer(
                model="dangvantuan/vietnamese-document-embedding",
                device="cuda",
                trust_remote_code=True
            )
            
            self._cache = SemanticCache(
                name=CacheConfig.LLM_CACHE_NAME,
                redis_url=CacheConfig.get_redis_url(),
                distance_threshold=CacheConfig.LLM_CACHE_DISTANCE_THRESHOLD,
                vectorizer=self._vectorizer,
                ttl=CacheConfig.LLM_CACHE_TTL
            )
            
            logger.info(f"LLM cache initialized: {CacheConfig.LLM_CACHE_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM cache: {e}")
            self._cache = None
            self._vectorizer = None
    
    def check(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Check cache for semantically similar prompt
        
        Args:
            prompt: Full LLM prompt
            
        Returns:
            Dict with 'response' and optional 'metadata' if cache hit, None otherwise
        """
        if not CacheConfig.LLM_CACHE_ENABLED or self._cache is None:
            return None
        
        try:
            results = self._cache.check(prompt=prompt, return_fields=["prompt", "response", "metadata"])
            
            if results and len(results) > 0:
                result = results[0]  # Get the closest match
                logger.debug(f"LLM cache HIT for: {prompt[:50]}...")
                return {
                    'response': result.get('response'),
                    'metadata': result.get('metadata', {}),
                    'cached_prompt': result.get('prompt'),
                    'distance': result.get('vector_distance', 0.0)
                }
            else:
                logger.debug(f"LLM cache MISS for: {prompt[:50]}...")
                return None
        except Exception as e:
            logger.warning(f"LLM cache check error: {e}")
            return None
    
    def store(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> bool:
        """
        Store LLM response in cache
        
        Args:
            prompt: Full LLM prompt
            response: LLM response
            metadata: Optional metadata (e.g., model name, token count)
            
        Returns:
            True if stored successfully
        """
        if not CacheConfig.LLM_CACHE_ENABLED or self._cache is None:
            return False
        
        try:
            key = self._cache.store(
                prompt=prompt,
                response=response,
                metadata=metadata or {}
            )
            logger.debug(f"Stored LLM response in cache: {key[:20]}...")
            return True
        except Exception as e:
            logger.warning(f"LLM cache store error: {e}")
            return False
    
    def clear(self):
        """Clear all cached LLM responses"""
        if not CacheConfig.LLM_CACHE_ENABLED or self._cache is None:
            return
        
        try:
            self._cache.clear()
            logger.info("Cleared LLM cache")
        except Exception as e:
            logger.error(f"Failed to clear LLM cache: {e}")
    
    def set_threshold(self, threshold: float):
        """Update distance threshold"""
        if not CacheConfig.LLM_CACHE_ENABLED or self._cache is None:
            return
        
        try:
            self._cache.set_threshold(threshold)
            logger.info(f"Updated LLM cache threshold to {threshold}")
        except Exception as e:
            logger.error(f"Failed to set threshold: {e}")


def get_llm_cache() -> LLMSemanticCacheManager:
    """Get singleton LLM cache instance"""
    global _llm_cache_instance
    
    if _llm_cache_instance is None:
        with _llm_cache_lock:
            if _llm_cache_instance is None:
                _llm_cache_instance = LLMSemanticCacheManager()
    
    return _llm_cache_instance


# ============================================
# INITIALIZATION & UTILITIES
# ============================================

async def ensure_cache_indices():
    """
    Initialize cache indices on application startup
    
    This ensures cache indices are created before first use.
    Safe to call multiple times.
    """
    try:
        CacheConfig.log_config()
        
        # Initialize both caches (if enabled)
        embedding_cache = get_embedding_cache()
        llm_cache = get_llm_cache()
        
        logger.info("Cache services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize cache services: {e}")
        raise


def clear_all_caches():
    """Clear all caches (embedding + LLM)"""
    try:
        embedding_cache = get_embedding_cache()
        embedding_cache.clear()
        
        llm_cache = get_llm_cache()
        llm_cache.clear()
        
        logger.info("All caches cleared")
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics and status"""
    return {
        "embedding_cache": {
            "enabled": CacheConfig.EMBEDDING_CACHE_ENABLED,
            "ttl": CacheConfig.EMBEDDING_CACHE_TTL if CacheConfig.EMBEDDING_CACHE_ENABLED else None
        },
        "llm_cache": {
            "enabled": CacheConfig.LLM_CACHE_ENABLED,
            "ttl": CacheConfig.LLM_CACHE_TTL if CacheConfig.LLM_CACHE_ENABLED else None,
            "distance_threshold": CacheConfig.LLM_CACHE_DISTANCE_THRESHOLD if CacheConfig.LLM_CACHE_ENABLED else None
        }
    }
