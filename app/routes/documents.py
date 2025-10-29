from fastapi import APIRouter, HTTPException, Depends
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.config import Config
import os
import json
from app.services.auth_service import verify_token_v2
from bson import ObjectId
from typing import Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

# Sử dụng connection pooling
_mongo_client = None

def get_mongo_client():
    """Get MongoDB client with connection pooling"""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(
                Config.DATABASE_URL,
                maxPoolSize=10,
                minPoolSize=2,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000
            )
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info("MongoDB connection established in documents route")
        except Exception as e:
            logger.error(f"Failed to establish MongoDB connection: {e}")
            _mongo_client = None
            raise
    return _mongo_client

@router.get("/types", response_model=dict)
async def get_file_types(current_user: dict = Depends(verify_token_v2)):
    """Lấy danh sách các loại file được hỗ trợ."""
    return {
        "file_types": [
            {"value": "public", "label": "Thông báo chung (Public)", "description": "Tài liệu công khai cho tất cả người dùng"},
            {"value": "student", "label": "Sinh viên (Student)", "description": "Tài liệu dành cho sinh viên"},
            {"value": "teacher", "label": "Giảng viên (Teacher)", "description": "Tài liệu dành cho giảng viên"},
            {"value": "admin", "label": "Quản trị viên (Admin)", "description": "Tài liệu dành cho quản trị viên"}
        ]
    }

@router.get("/list", response_model=dict)
async def list_documents(
    file_type: str = None, 
    q: str = None,            
    limit: int = 100, 
    skip: int = 0,
    current_user: dict = Depends(verify_token_v2)
):
    """Lấy danh sách tài liệu (có phân trang & tìm kiếm)."""
    try:
        documents = []
        total = 0

        # Thử lấy từ MongoDB
        try:
            client = get_mongo_client()
            db = client["faiss_db"]
            collection = db["metadata"]

            filter_dict = {}
            if file_type:
                filter_dict["file_type"] = file_type
            if q:
                filter_dict["$or"] = [
                    {"filename": {"$regex": q, "$options": "i"}},
                    {"uploaded_by": {"$regex": q, "$options": "i"}},
                ]

            # Tính tổng
            total = collection.count_documents(filter_dict)

            # Lấy dữ liệu phân trang
            documents = list(
                collection.find(filter_dict)
                .skip(skip)
                .limit(limit)
                .sort("createdAt", -1)
            )

            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                doc["_id"] = str(doc["_id"])

            return {
                "documents": documents,
                "total": total,
                "source": "mongodb",
                "showing": len(documents),
            }
        
        except PyMongoError as e:
            logger.error(f"Failed to retrieve documents from MongoDB: {str(e)}")
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")

        # Fallback: JSON
        logger.info("Falling back to JSON files")
        base_path = Config.DATA_PATH
        file_type_paths = Config.get_file_type_paths()
        metadata_paths = [
            os.path.join(base_path, file_type_paths[role]['vector_folder'], "metadata.json")
            for role in file_type_paths
        ]

        all_documents = []
        for metadata_file in metadata_paths:
            try:
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata_list = json.load(f)

                    if file_type:
                        metadata_list = [item for item in metadata_list if item.get("file_type") == file_type]
                    if q:
                        metadata_list = [
                            item for item in metadata_list
                            if q.lower() in item.get("filename", "").lower()
                            or q.lower() in item.get("uploaded_by", "").lower()
                        ]

                    all_documents.extend(metadata_list)
            except Exception as e:
                logger.error(f"Error reading {metadata_file}: {str(e)}")

        # Sort + paginate
        all_documents.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
        total = len(all_documents)
        documents = all_documents[skip: skip + limit]

        return {
            "documents": documents,
            "total": total,
            "source": "json",
            "showing": len(documents),
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")
def convert_to_str(d):
    """Recursively convert all values in a dictionary to strings."""
    if isinstance(d, dict):
        return {k: convert_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_to_str(i) for i in d]
    else:
        # Nếu là ObjectId, chuyển đổi sang str
        if isinstance(d, (ObjectId)):
            return str(d)
        return d

@router.get("/list/details/{document_id}", response_model=dict)
async def get_document_details(
    document_id: str,
    current_user: dict = Depends(verify_token_v2)
) -> Dict[str, Any]:
    """Lấy chi tiết tài liệu theo ID trực tiếp từ MongoDB."""
    try:
        # Kết nối MongoDB
        client = get_mongo_client()
        db = client["faiss_db"]
        collection = db["metadata"]

        # Tìm tài liệu theo document_id
        document = collection.find_one({"_id": document_id})

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Chuyển đổi tất cả ID và ObjectId thành chuỗi
        document = convert_to_str(document)
        
        # Ẩn đường dẫn nội bộ trong url
        if "url" in document:
            document["url"] = document["url"].replace(Config.DATA_PATH, "/files")

        # Lấy thông tin người dùng và phòng ban từ trường 'role'
        allowed_users = document.get("role", {}).get("user", [])
        allowed_subjects = document.get("role", {}).get("subject", [])

        return {
            "document": document
        }

    except PyMongoError as e:
        logger.error(f"Failed to retrieve document details from MongoDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving document details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Optional: Add cleanup function for app shutdown
def close_documents_mongo():
    """Close MongoDB connection for documents route"""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None