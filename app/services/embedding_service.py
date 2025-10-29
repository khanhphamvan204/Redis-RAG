# app/services/embedding_service.py
import os
import logging
import gc
from typing import List, Optional, Dict
from functools import lru_cache
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, Num, Text
from app.config import Config
from app.services.file_service import get_file_paths
from app.services.metadata_service import find_document_info
from app.services.document_loader import load_new_documents
import redis
import numpy as np
import json

# HuggingFace Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Global embedding model cache
_embedding_model_cache = None
_embedding_model_lock = None

try:
    import threading
    _embedding_model_lock = threading.Lock()
except ImportError:
    class DummyLock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    _embedding_model_lock = DummyLock()

@lru_cache(maxsize=1)
def _create_embedding_model():
    """Private function to create embedding model with caching"""
    logger.info("Creating new embedding model instance...")
    
    model = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={'device': 'cuda','trust_remote_code':True}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info("Embedding model created successfully")
    return model

def get_embedding_model():
    """Get consistent embedding model with singleton pattern"""
    global _embedding_model_cache
    
    if _embedding_model_cache is None:
        with _embedding_model_lock:
            if _embedding_model_cache is None:
                _embedding_model_cache = _create_embedding_model()
    
    return _embedding_model_cache

def clear_embedding_model_cache():
    """Clear embedding model cache"""
    global _embedding_model_cache
    with _embedding_model_lock:
        if _embedding_model_cache is not None:
            logger.info("Clearing embedding model cache")
            _embedding_model_cache = None
            _create_embedding_model.cache_clear()
            gc.collect()

class EmbeddingModelManager:
    """Context manager for embedding model"""
    def __init__(self):
        self.model = None
    
    def __enter__(self):
        self.model = get_embedding_model()
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def get_redis_client():
    """Get Redis client connection"""
    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=False  # Important for binary data
    )

def get_index_name(file_type: str) -> str:
    """Generate Redis index name based on file type"""
    return f"doc_index_{file_type}"

def create_redis_index(file_type: str, embedding_dim: int = 768):
    """Create Redis search index for a specific file type"""
    index_name = get_index_name(file_type)
    
    schema = {
        "index": {
            "name": index_name,
            "prefix": f"doc:{file_type}:",
            "storage_type": "hash"
        },
        "fields": [
            {
                "name": "content",
                "type": "text"
            },
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": embedding_dim,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32"
                }
            },
            {
                "name": "doc_id",
                "type": "tag"
            },
            {
                "name": "filename",
                "type": "text"
            },
            {
                "name": "file_type",
                "type": "tag"
            },
            {
                "name": "uploaded_by",
                "type": "text"
            },
            {
                "name": "role_user",
                "type": "tag"
            },
            {
                "name": "role_subject",
                "type": "tag"
            },
            {
                "name": "created_at",
                "type": "text"
            },
            {
                "name": "url",
                "type": "text"
            }
        ]
    }
    
    try:
        client = get_redis_client()
        index = SearchIndex.from_dict(schema)
        index.set_client(client)
        index.create(overwrite=True)
        logger.info(f"Created Redis index: {index_name}")
        return index
    except Exception as e:
        logger.error(f"Error creating Redis index: {e}")
        raise

def get_or_create_index(file_type: str, embedding_dim: int = 768):
    """Get existing index or create new one"""
    index_name = get_index_name(file_type)
    client = get_redis_client()
    
    try:
        # Try to get existing index
        index = SearchIndex(name=index_name)
        index.set_client(client)
        # Test if index exists
        index.info()
        logger.info(f"Using existing Redis index: {index_name}")
        return index
    except:
        # Create new index if doesn't exist
        logger.info(f"Index {index_name} not found, creating new one")
        return create_redis_index(file_type, embedding_dim)

def semantic_sliding_window_split(text: str, embedding_model, window_overlap: float = 0.2) -> List[str]:
    """Sliding window with semantic boundaries"""
    try:
        semantic_chunker = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            sentence_split_regex=r'(?<!\b\d\.)(?<!\b\d\d\.)(?<!\b\d\d\d\.)(?<!\b[A-Za-zÀ-ỹ]\.)(?<!\b[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ]\.)(?<!\b[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ][A-Za-zÀ-ỹ]\.)(?<=[.?!…])\s+(?=[A-ZÀ-Ỵ0-9])',
            buffer_size=5,
        )
        
        chunks = semantic_chunker.split_text(text)
        
        if len(chunks) <= 1:
            return chunks
        
        sliding_chunks = []
        
        for i in range(len(chunks)):
            if i == 0:
                sliding_chunks.append(chunks[i])
            else:
                prev_chunk_words = chunks[i-1].split()
                overlap_words_count = int(len(prev_chunk_words) * window_overlap)
                
                if overlap_words_count > 0:
                    overlap_text = ' '.join(prev_chunk_words[-overlap_words_count:])
                    new_chunk = overlap_text + " " + chunks[i]
                else:
                    new_chunk = chunks[i]
                
                sliding_chunks.append(new_chunk)
        
        return sliding_chunks
        
    except Exception as e:
        logger.warning(f"Semantic sliding window failed: {e}")
        return [text]

def get_text_splitter(use_semantic: bool = True, semantic_overlap: float = 0.2, embedding_model=None):
    """Get text splitter"""
    try:
        if use_semantic:
            logger.info("Using SemanticChunker with sliding window overlap")
            
            if embedding_model is None:
                embedding_model = get_embedding_model()
            
            class SemanticSlidingWindowSplitter:
                def __init__(self, embedding_model, window_overlap=0.2):
                    self.embedding_model = embedding_model
                    self.window_overlap = window_overlap
                
                def split_text(self, text: str) -> List[str]:
                    return semantic_sliding_window_split(text, self.embedding_model, self.window_overlap)
                
                def split_documents(self, documents: List[Document]) -> List[Document]:
                    chunks = []
                    for doc in documents:
                        text_chunks = self.split_text(doc.page_content)
                        for chunk_text in text_chunks:
                            chunk_doc = Document(
                                page_content=chunk_text,
                                metadata=doc.metadata.copy()
                            )
                            chunks.append(chunk_doc)
                    return chunks
            
            return SemanticSlidingWindowSplitter(embedding_model, semantic_overlap)
        else:
            logger.info("Using RecursiveCharacterTextSplitter")
            return RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=200
            )
    except Exception as e:
        logger.warning(f"Failed to create SemanticSlidingWindowSplitter: {e}")
        return RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

def add_to_embedding(file_path: str, metadata, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2):
    """Add documents to Redis vector database"""
    try:
        logger.info(f"Starting embedding process for: {file_path}")
        
        documents = load_new_documents(file_path, metadata)
        if not documents:
            logger.warning(f"No documents loaded from {file_path}")
            return False

        with EmbeddingModelManager() as embedding_model:
            # Split documents
            text_splitter = get_text_splitter(
                use_semantic=use_semantic_chunking, 
                semantic_overlap=semantic_overlap,
                embedding_model=embedding_model
            )
            
            try:
                chunks = text_splitter.split_documents(documents)
                logger.info(f"Created {len(chunks)} chunks")
            except Exception as e:
                if use_semantic_chunking:
                    logger.warning(f"Semantic chunking failed: {e}. Falling back")
                    text_splitter = get_text_splitter(use_semantic=False)
                    chunks = text_splitter.split_documents(documents)
                else:
                    raise e
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False
            
            # Get or create Redis index
            index = get_or_create_index(metadata.file_type)
            client = get_redis_client()
            
            # Generate embeddings and store in Redis
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = embedding_model.embed_query(chunk.page_content)
                    
                    # Prepare data for Redis
                    doc_key = f"doc:{metadata.file_type}:{metadata.id}:{i}"
                    
                    # Convert role lists to comma-separated strings for tag fields
                    role_user_str = ",".join(chunk.metadata.get('role', {}).get('user', []))
                    role_subject_str = ",".join(chunk.metadata.get('role', {}).get('subject', []))
                    
                    data = {
                        "content": chunk.page_content,
                        "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                        "doc_id": metadata.id,
                        "filename": metadata.filename,
                        "file_type": metadata.file_type,
                        "uploaded_by": chunk.metadata.get('uploaded_by', ''),
                        "role_user": role_user_str,
                        "role_subject": role_subject_str,
                        "created_at": chunk.metadata.get('createdAt', ''),
                        "url": chunk.metadata.get('url', '')
                    }
                    
                    # Store in Redis
                    client.hset(doc_key, mapping=data)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            logger.info(f"Successfully added {len(chunks)} chunks to Redis")
            return True
            
    except Exception as e:
        logger.error(f"Error in add_to_embedding: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        gc.collect()

def delete_from_redis_index(file_type: str, doc_id: str) -> bool:
    """Delete documents from Redis index"""
    try:
        client = get_redis_client()
        index_name = get_index_name(file_type)
        
        # Find all keys matching the doc_id
        pattern = f"doc:{file_type}:{doc_id}:*"
        keys = client.keys(pattern)
        
        if keys:
            client.delete(*keys)
            logger.info(f"Deleted {len(keys)} chunks for doc_id: {doc_id}")
        else:
            logger.warning(f"No documents found with doc_id: {doc_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting from Redis: {str(e)}")
        return False

def update_document_metadata_in_vector_store(doc_id: str, old_metadata: dict, new_metadata, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2) -> bool:
    """Update document by re-embedding"""
    try:
        old_file_type = old_metadata.get('file_type')
        
        # Delete old document
        success = delete_from_redis_index(old_file_type, doc_id)
        if not success:
            return False
        
        # Re-embed with new metadata
        file_path = new_metadata.url
        if os.path.exists(file_path):
            return add_to_embedding(file_path, new_metadata, use_semantic_chunking, semantic_overlap)
        else:
            logger.error(f"File not found for re-embedding: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata in vector store: {str(e)}")
        return False

def update_metadata_only(doc_id: str, new_metadata) -> bool:
    """Update only metadata without re-embedding"""
    try:
        client = get_redis_client()
        pattern = f"doc:{new_metadata.file_type}:{doc_id}:*"
        keys = client.keys(pattern)
        
        if not keys:
            logger.warning(f"No documents found for doc_id: {doc_id}")
            return False
        
        updated_count = 0
        role_user_str = ",".join(new_metadata.role.get('user', []))
        role_subject_str = ",".join(new_metadata.role.get('subject', []))
        
        for key in keys:
            try:
                client.hset(key, mapping={
                    "filename": new_metadata.filename,
                    "uploaded_by": new_metadata.uploaded_by,
                    "role_user": role_user_str,
                    "role_subject": role_subject_str,
                    "url": new_metadata.url
                })
                updated_count += 1
            except Exception as e:
                logger.error(f"Error updating key {key}: {e}")
                continue
        
        if updated_count > 0:
            logger.info(f"Updated metadata for {updated_count} chunks")
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata only: {str(e)}")
        return False

def smart_metadata_update(doc_id: str, old_metadata: dict, new_metadata, force_re_embed: bool = False, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2) -> bool:
    """Smart metadata update with fallback logic"""
    try:
        file_type_changed = old_metadata.get('file_type') != new_metadata.file_type
        filename_changed = old_metadata.get('filename') != new_metadata.filename
        
        if file_type_changed or filename_changed or force_re_embed:
            return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking, semantic_overlap)
        else:
            success = update_metadata_only(doc_id, new_metadata)
            if not success:
                logger.info("Metadata-only update failed, attempting re-embedding")
                return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking, semantic_overlap)
            return success
            
    except Exception as e:
        logger.error(f"Error in smart metadata update: {str(e)}")
        return False

def get_embedding_model_info():
    """Get information about current embedding model"""
    global _embedding_model_cache
    return {
        "is_cached": _embedding_model_cache is not None,
        "cache_info": _create_embedding_model.cache_info() if hasattr(_create_embedding_model, 'cache_info') else None
    }

def cleanup_embedding_resources():
    """Cleanup embedding resources on shutdown"""
    logger.info("Cleaning up embedding resources...")
    clear_embedding_model_cache()