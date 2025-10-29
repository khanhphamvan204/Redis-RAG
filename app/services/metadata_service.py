import os
import json
import logging
from contextlib import contextmanager
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from fastapi import HTTPException
from app.config import Config
from app.services.file_service import get_file_paths
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Connection manager singleton
class MongoManager:
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self) -> Optional[MongoClient]:
        if self._client is None:
            try:
                # Log the connection URL for debugging (without password)
                connection_url = Config.DATABASE_URL
                if '@' in connection_url:
                    masked_url = connection_url.split('@')[0].split('//')[0] + '//***:***@' + connection_url.split('@')[1]
                    logger.info(f"Attempting MongoDB connection to: {masked_url}")
                else:
                    logger.info(f"Attempting MongoDB connection to: {connection_url}")
                
                self._client = MongoClient(
                    Config.DATABASE_URL,
                    maxPoolSize=10,
                    minPoolSize=2,
                    maxIdleTimeMS=30000,
                    waitQueueTimeoutMS=5000,
                    serverSelectionTimeoutMS=10000,
                    connectTimeoutMS=10000,
                    socketTimeoutMS=30000,
                    retryWrites=True,
                    retryReads=True,
                    heartbeatFrequencyMS=30000,
                )
                # Test connection
                self._client.admin.command('ping')
                logger.info("MongoDB connection established successfully")
                
                # Log server info
                server_info = self._client.server_info()
                logger.info(f"Connected to MongoDB version: {server_info.get('version', 'unknown')}")
                
            except ServerSelectionTimeoutError as e:
                logger.warning(f"MongoDB server not available: {e}")
                self._client = None
                return None
            except Exception as e:
                logger.error(f"Failed to establish MongoDB connection: {e}")
                self._client = None
                return None
        return self._client
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected and available"""
        try:
            if self._client is None:
                return False
            self._client.admin.command('ping')
            return True
        except:
            return False
    
    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            logger.info("MongoDB connection closed")

# Global instance
mongo_manager = MongoManager()

@contextmanager
def get_mongo_connection():
    """Context manager for MongoDB connections"""
    client = None
    try:
        client = mongo_manager.get_client()
        yield client
    except Exception as e:
        logger.error(f"MongoDB operation failed: {e}")
        yield None
    finally:
        if client:
            logger.debug("MongoDB connection context closed")

def save_metadata_to_json(metadata) -> bool:
    """Save metadata to JSON file"""
    try:
        file_type = metadata.file_type
        _, vector_db_path = get_file_paths(file_type, metadata.filename)
        metadata_file = os.path.join(vector_db_path, "metadata.json")
        
        existing_metadata = []
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"No existing metadata.json found at {metadata_file}, creating new")
        
        metadata_dict = metadata.dict(by_alias=True)
        existing_metadata.append(metadata_dict)
        
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully saved metadata to JSON file: {metadata_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save to JSON file: {e}")
        return False

def save_metadata(metadata) -> bool:
    """Save metadata to MongoDB first, fallback to JSON if MongoDB fails"""
    try:
        with get_mongo_connection() as client:
            if client is not None:
                db = client["faiss_db"]
                collection = db["metadata"]
                
                metadata_dict = metadata.dict(by_alias=True)
                result = collection.insert_one(metadata_dict)
                
                if result.inserted_id:
                    logger.info(f"Successfully saved metadata to MongoDB for _id: {metadata.id}")
                    return True
                else:
                    logger.warning("MongoDB insert operation returned no inserted_id")
                    
    except Exception as e:
        logger.warning(f"MongoDB save failed: {str(e)}")
    
    # Fallback to JSON file
    logger.info("MongoDB save failed, falling back to JSON file storage")
    return save_metadata_to_json(metadata)

def delete_metadata(doc_id: str) -> bool:
    """Delete metadata with proper connection handling"""
    mongodb_success = False
    json_success = False
    
    # Try MongoDB first
    try:
        with get_mongo_connection() as client:
            if client is not None:
                db = client["faiss_db"]
                collection = db["metadata"]
                
                result = collection.delete_one({"_id": doc_id})
                if result.deleted_count > 0:
                    logger.info(f"Successfully deleted metadata from MongoDB for _id: {doc_id}")
                    mongodb_success = True
                else:
                    logger.info(f"Document not found in MongoDB: {doc_id}")
    except Exception as e:
        logger.warning(f"MongoDB delete failed: {str(e)}")
    
    # Try JSON files
    file_type_paths = Config.get_file_type_paths()
    for role in file_type_paths:
        try:
            metadata_file = os.path.join(
                Config.DATA_PATH,
                file_type_paths[role]['vector_folder'],
                "metadata.json"
            )
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                
                original_length = len(metadata_list)
                metadata_list = [item for item in metadata_list if item.get('_id') != doc_id]
                
                if len(metadata_list) < original_length:
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                    logger.info(f"Successfully deleted metadata from {metadata_file}")
                    json_success = True
        except Exception as e:
            logger.error(f"Error deleting from {metadata_file}: {str(e)}")
    
    success = mongodb_success or json_success
    if not success:
        logger.error(f"Failed to delete metadata from both MongoDB and JSON files: {doc_id}")
    
    return success

def find_document_info(doc_id: str) -> Optional[Dict]:
    """Find document info with proper connection handling"""
    # Try MongoDB first
    try:
        with get_mongo_connection() as client:
            if client is not None:
                db = client["faiss_db"]
                collection = db["metadata"]
                
                doc_info = collection.find_one({"_id": doc_id})
                
                if doc_info:
                    logger.info(f"Found document in MongoDB: {doc_id}")
                    return doc_info
                else:
                    logger.info(f"Document not found in MongoDB: {doc_id}")
    except Exception as e:
        logger.warning(f"MongoDB search failed: {str(e)}")
    
    # Fallback to JSON files
    logger.info("Searching in JSON files as fallback")
    file_type_paths = Config.get_file_type_paths()
    for role in file_type_paths:
        metadata_file = os.path.join(
            Config.DATA_PATH,
            file_type_paths[role]['vector_folder'],
            "metadata.json"
        )
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                
                for item in metadata_list:
                    if item.get('_id') == doc_id:
                        logger.info(f"Found document in JSON file: {metadata_file}")
                        return item
        except Exception as e:
            logger.error(f"Error reading {metadata_file}: {str(e)}")
    
    logger.warning(f"Document not found in both MongoDB and JSON files: {doc_id}")
    return None

def get_all_metadata() -> List[Dict]:
    """Get all metadata from MongoDB with JSON fallback"""
    # Try MongoDB first
    try:
        with get_mongo_connection() as client:
            if client is not None:
                db = client["faiss_db"]
                collection = db["metadata"]
                
                metadata_list = list(collection.find({}))
                if metadata_list:
                    logger.info(f"Retrieved {len(metadata_list)} documents from MongoDB")
                    return metadata_list
                else:
                    logger.info("No documents found in MongoDB")
                    
    except Exception as e:
        logger.warning(f"MongoDB get_all failed: {str(e)}")
    
    # Fallback to JSON files
    logger.info("Retrieving metadata from JSON files as fallback")
    all_metadata = []
    file_type_paths = Config.get_file_type_paths()
    for role in file_type_paths:
        metadata_file = os.path.join(
            Config.DATA_PATH,
            file_type_paths[role]['vector_folder'],
            "metadata.json"
        )
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                    all_metadata.extend(metadata_list)
        except Exception as e:
            logger.error(f"Error reading {metadata_file}: {str(e)}")
    
    logger.info(f"Retrieved {len(all_metadata)} documents from JSON files")
    return all_metadata

def get_connection_status() -> Dict:
    """Get current MongoDB connection status"""
    is_connected = mongo_manager.is_connected()
    status = {
        "mongodb_connected": is_connected,
        "connection_url": Config.DATABASE_URL.split('@')[1] if '@' in Config.DATABASE_URL else Config.DATABASE_URL,
        "fallback_mode": not is_connected
    }
    return status

def initialize_mongo():
    """Initialize MongoDB connection at startup"""
    try:
        client = mongo_manager.get_client()
        if client:
            logger.info("MongoDB initialized successfully")
        else:
            logger.warning("MongoDB not available - using JSON fallback mode")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")

def close_mongo():
    """Close MongoDB connection at shutdown"""
    mongo_manager.close()