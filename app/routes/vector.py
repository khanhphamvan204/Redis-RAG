# app/routes/vector.py
import os
import time
import gc
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from app.services.auth_service import verify_token_v2, filter_accessible_files
from app.services.embedding_service import (
    get_redis_client,
    get_index_name,
    get_embedding_model,
    add_to_embedding,
    delete_from_redis_index,
    smart_metadata_update
)
from app.services.file_service import get_file_paths
from app.services.metadata_service import (
    save_metadata,
    find_document_info,
    delete_metadata
)
from app.services.cache_service import (
    get_embedding_cache,
    get_llm_cache
)
from app.config import Config
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
import logging
import traceback
import json
import uuid
from datetime import datetime, timezone, timedelta
import shutil
import logging
import time
import numpy as np
from redisvl.extensions.message_history import SemanticMessageHistory 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from typing import List, Optional, Dict, Any
from langchain.prompts import PromptTemplate
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer
from app.services.query_rewriter import QueryRewriter
from app.services.hdfs_service import hdfs_service
from app.services.query_tracker import get_query_tracker
from pymongo import MongoClient

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================
# USER METADATA LOOKUP HELPER
# ============================================

def get_user_metadata_from_db(user_id: int) -> Dict[str, Any]:
    """
    Lookup comprehensive user metadata from MongoDB for query tracking
    
    Args:
        user_id: User ID from JWT token
        
    Returns:
        Dictionary with UserInfo fields:
        - user_id: int
        - user_type: str
        - department_id: int | None
        - department_name: str | None
        - code: str | None (student_code or teacher_code)
        - years: int | None (calculated from enrollment_date or hire_date)
    """
    try:
        client = MongoClient(Config.DATABASE_URL, serverSelectionTimeoutMS=3000)
        db = client["faiss_db"]
        
        # Fetch user document with all relevant fields
        user_doc = db.users.find_one(
            {"user_id": user_id},
            {
                "_id": 0, 
                "user_id": 1,
                "user_type": 1, 
                "department_id": 1, 
                "student_info": 1,
                "teacher_info": 1
            }
        )
        
        if not user_doc:
            logger.warning(f"User {user_id} not found in database")
            return {
                "user_id": user_id,
                "user_type": "Unknown",
                "department_id": None,
                "department_name": None,
                "code": None,
                "years": None
            }
        
        user_type = user_doc.get("user_type", "Unknown")
        department_id = user_doc.get("department_id")
        department_name = None
        code = None
        years = None
        
        # Lookup department_name from departments collection
        if department_id is not None:
            dept_doc = db.departments.find_one(
                {"department_id": department_id},
                {"_id": 0, "department_name": 1}
            )
            if dept_doc:
                department_name = dept_doc.get("department_name")
            else:
                logger.warning(f"Department {department_id} not found in departments collection")
        
        # Calculate years based on user type
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        if user_type == "Học sinh" or user_type == "Sinh viên":
            # Student: calculate student year from enrollment_date
            student_info = user_doc.get("student_info", {})
            if student_info and isinstance(student_info, dict):
                code = student_info.get("student_code")
                enrollment_date = student_info.get("enrollment_date")
                
                if enrollment_date:
                    try:
                        # Parse enrollment date (format: "YYYY-MM-DD")
                        enrollment_year = int(enrollment_date[:4])
                        enrollment_month = int(enrollment_date[5:7])
                        
                        # Calculate academic year
                        # If before September, subtract 1 from difference
                        years_diff = current_year - enrollment_year
                        if current_month < 9:  # Before September
                            years_diff -= 1
                        if enrollment_month >= 9:  # Enrolled in Sept or later
                            years_diff += 1
                        
                        years = max(1, min(years_diff, 6))  # Clamp to 1-6
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse enrollment_date '{enrollment_date}': {e}")
        
        elif user_type == "Giáo viên":
            # Teacher: calculate teaching years from hire_date
            teacher_info = user_doc.get("teacher_info", {})
            if teacher_info and isinstance(teacher_info, dict):
                code = teacher_info.get("teacher_code")
                hire_date = teacher_info.get("hire_date")
                
                if hire_date:
                    try:
                        # Parse hire date (format: "YYYY-MM-DD")
                        hire_year = int(hire_date[:4])
                        hire_month = int(hire_date[5:7])
                        
                        # Calculate years of service
                        years = current_year - hire_year
                        if current_month < hire_month:
                            years -= 1
                        
                        years = max(0, years)  # At least 0
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse hire_date '{hire_date}': {e}")
        
        client.close()
        
        return {
            "user_id": user_id,
            "user_type": user_type,
            "department_id": department_id,
            "department_name": department_name,
            "code": code,
            "years": years
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch user metadata: {e}")
        return {
            "user_id": user_id,
            "user_type": "Unknown",
            "department_id": None,
            "department_name": None,
            "code": None,
            "years": None
        }



class AddVectorRequest(BaseModel):
    id: str = Field(alias="_id")
    filename: str
    url: str
    uploaded_by: str
    role: dict
    file_type: str
    createdAt: str

class SearchResult(BaseModel):
    content: str
    metadata: dict

class VectorSearchRequest(BaseModel):
    query: str = Field(..., description="Câu truy vấn tìm kiếm")
    k: int = Field(default=5, ge=1, le=100, description="Số lượng kết quả trả về (1-100)")
    file_type: str = Field(..., description="Loại tài liệu (public, student, teacher, admin)")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Ngưỡng độ tương quan (0.0-1.0)")

class VectorSearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
    k_requested: int
    file_type: str
    similarity_threshold: float
    search_time_ms: float

@router.post("/add", response_model=dict)
async def add_vector_document(
    file: UploadFile = File(...),
    uploaded_by: str = Form(...),
    file_type: str = Form(...),
    role_user: str = Form(default="[]"),
    role_subject: str = Form(default="[]"),
    current_user: dict = Depends(verify_token_v2)
):
    try:
        file_type_paths_data = Config.get_file_type_paths() 
        valid_file_types = list(file_type_paths_data.keys())
        
        if file_type not in valid_file_types:
            raise HTTPException(status_code=400, detail=f"Invalid file_type. Must be one of: {valid_file_types}")
        
        file_name = file.filename
        file_path, _ = get_file_paths(file_type, file_name)
        
        if os.path.exists(file_path):
            raise HTTPException(status_code=409, detail=f"File already exists at path: {file_path}")
        
        # Validate file extension
        supported_extensions = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls'}
        file_extension = os.path.splitext(file_name.lower())[1]
        if file_extension not in supported_extensions:
            raise HTTPException(status_code=400, detail=f"File format {file_extension} not supported")
        
        # Generate metadata
        generated_id = str(uuid.uuid4())
        vietnam_tz = timezone(timedelta(hours=7))
        created_at = datetime.now(vietnam_tz).isoformat()
        
        role = {
            "user": json.loads(role_user),
            "subject": json.loads(role_subject)
        }
        
        metadata = AddVectorRequest(
            _id=generated_id,
            filename=file_name,
            url=file_path,
            uploaded_by=uploaded_by,
            role=role,
            file_type=file_type,
            createdAt=created_at
        )
        
        # Save file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # ========================================
        # HDFS AUTO-UPLOAD (Phase 3 Integration)
        # ========================================
        hdfs_upload_success = False
        hdfs_path = None
        try:
            # Create HDFS directory structure: /data/documents/{file_type}/{date}
            hdfs_dir = f"/data/documents/{file_type}/{datetime.now().strftime('%Y-%m-%d')}"
            hdfs_service.create_directory(hdfs_dir)
            
            # Upload file to HDFS
            hdfs_path = f"{hdfs_dir}/{file_name}"
            hdfs_upload_success = hdfs_service.upload_file(
                local_path=file_path,
                hdfs_path=hdfs_path,
                overwrite=True
            )
            
            if hdfs_upload_success:
                # Save metadata to HDFS
                hdfs_metadata = {
                    "doc_id": generated_id,
                    "filename": file_name,
                    "file_type": file_type,
                    "uploaded_by": uploaded_by,
                    "role": role,
                    "created_at": created_at,
                    "local_path": file_path,
                    "hdfs_path": hdfs_path,
                    "size_bytes": os.path.getsize(file_path)
                }
                
                meta_path = f"{hdfs_dir}/{file_name}_meta.json"
                hdfs_service.write_file(
                    hdfs_path=meta_path,
                    content=json.dumps(hdfs_metadata, indent=2),
                    overwrite=True
                )
                
                logger.info(f"Uploaded to HDFS: {hdfs_path}")
            else:
                logger.warning(f"HDFS upload failed for {file_name}")
                
        except Exception as hdfs_error:
            logger.error(f"HDFS upload error: {str(hdfs_error)}")
            # Continue anyway - không block main flow
        
        # Save metadata and add to Redis
        try:
            save_metadata(metadata)
            add_to_embedding(file_path, metadata)
        except Exception as embed_error:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Failed to process embeddings: {str(embed_error)}")
        
        return {
            "message": "Vector added successfully",
            "_id": generated_id,
            "filename": file_name,
            "file_type": file_type,
            "file_path": file_path,
            "hdfs_uploaded": hdfs_upload_success,
            "hdfs_path": hdfs_path if hdfs_upload_success else None,
            "status": "created"
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError as json_error:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in role fields: {str(json_error)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.delete("/{doc_id}", response_model=dict)
async def delete_vector_document(
    doc_id: str,
    current_user: dict = Depends(verify_token_v2)
):
    try:
        doc_info = find_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
        
        file_type = doc_info.get('file_type')
        filename = doc_info.get('filename')
        file_path = doc_info.get('url')
        
        deletion_results = {
            "file_deleted": False,
            "metadata_deleted": False,
            "vector_deleted": False
        }
        
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            deletion_results["file_deleted"] = True
        
        deletion_results["vector_deleted"] = delete_from_redis_index(file_type, doc_id)
        deletion_results["metadata_deleted"] = delete_metadata(doc_id)
        
        message = "Document deleted successfully" if all(deletion_results.values()) else "Document partially deleted"
        response = {
            "message": message,
            "_id": doc_id,
            "file_type": file_type,
            "filename": filename,
            "deletion_results": deletion_results
        }
        
        if not all(deletion_results.values()):
            response["warning"] = "Some components could not be deleted"
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.get("/{doc_id}", response_model=dict)
async def get_vector_document(
    doc_id: str,
    current_user: dict = Depends(verify_token_v2)
):
    try:
        doc_info = find_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
        
        file_path = doc_info.get('url')
        file_exists = os.path.exists(file_path) if file_path else False
        
        # Check Redis vectors
        client = get_redis_client()
        file_type = doc_info.get('file_type')
        pattern = f"doc:{file_type}:{doc_id}:*"
        vector_keys = client.keys(pattern)
        vector_exists = len(vector_keys) > 0
        
        file_size = os.path.getsize(file_path) if file_exists else None
        
        return {
            **doc_info,
            "file_exists": file_exists,
            "vector_exists": vector_exists,
            "vector_count": len(vector_keys),
            "file_size": file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@router.put("/{doc_id}", response_model=dict)
async def update_vector_document(
    doc_id: str,
    current_user: dict = Depends(verify_token_v2),
    filename: str = Form(None),
    uploaded_by: str = Form(None),
    file_type: str = Form(None),
    role_user: str = Form(None),
    role_subject: str = Form(None),
    force_re_embed: bool = Form(False)
):
    try:
        current_doc = find_document_info(doc_id)
        if not current_doc:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
        
        old_metadata = current_doc.copy()
        current_file_type = current_doc.get('file_type')
        current_filename = current_doc.get('filename')
        current_file_path = current_doc.get('url')
        
        # Handle filename validation
        final_filename = current_filename
        if filename:
            current_name, current_extension = os.path.splitext(current_filename)
            input_name, input_extension = os.path.splitext(filename)
            
            if input_extension:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Please provide filename without extension. Current file extension '{current_extension}' will be preserved automatically."
                )
            
            final_filename = filename + current_extension
            
            supported_extensions = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls'}
            if current_extension.lower() not in supported_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Current file extension '{current_extension}' is not supported"
                )
        
        # Check for duplicate
        if filename and final_filename != current_filename:
            target_file_type = file_type if file_type else current_file_type
            target_file_path, _ = get_file_paths(target_file_type, final_filename)
            if os.path.exists(target_file_path):
                raise HTTPException(
                    status_code=409,
                    detail=f"File '{final_filename}' already exists in {target_file_type} category"
                )

        file_type_paths_data = Config.get_file_type_paths() 
        valid_file_types = list(file_type_paths_data.keys())
        
        if file_type and file_type not in valid_file_types:
            raise HTTPException(status_code=400, detail=f"Invalid file_type")
        
        new_filename = final_filename
        new_file_type = file_type or current_file_type
        new_uploaded_by = uploaded_by or current_doc.get('uploaded_by')
        
        current_role = current_doc.get('role', {'user': [], 'subject': []})
        new_role = {
            'user': json.loads(role_user) if role_user else current_role.get('user', []),
            'subject': json.loads(role_subject) if role_subject else current_role.get('subject', [])
        } if role_user or role_subject else current_role
        
        filename_changed = filename and new_filename != current_filename
        file_type_changed = file_type and file_type != current_file_type
        
        operations = {
            "file_renamed": False,
            "file_moved": False,
            "vector_updated": False,
            "metadata_updated": False,
            "update_method": "none"
        }
        
        # Handle file operations
        final_file_path = current_file_path
        if filename_changed and not file_type_changed:
            new_file_path, _ = get_file_paths(current_file_type, new_filename)
            if os.path.exists(current_file_path):
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                shutil.move(current_file_path, new_file_path)
                operations["file_renamed"] = True
                final_file_path = new_file_path
        elif file_type_changed and not filename_changed:
            new_file_path, _ = get_file_paths(new_file_type, current_filename)
            if os.path.exists(current_file_path):
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                shutil.move(current_file_path, new_file_path)
                operations["file_moved"] = True
                final_file_path = new_file_path
        elif filename_changed and file_type_changed:
            temp_file_path, _ = get_file_paths(current_file_type, new_filename)
            if os.path.exists(current_file_path):
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                shutil.move(current_file_path, temp_file_path)
                operations["file_renamed"] = True
                new_file_path, _ = get_file_paths(new_file_type, new_filename)
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                shutil.move(temp_file_path, new_file_path)
                operations["file_moved"] = True
                final_file_path = new_file_path
        
        new_metadata = AddVectorRequest(
            _id=doc_id,
            filename=new_filename,
            url=final_file_path,
            uploaded_by=new_uploaded_by,
            role=new_role,
            file_type=new_file_type,
            createdAt=current_doc.get('createdAt')
        )
        
        operations["vector_updated"] = smart_metadata_update(doc_id, old_metadata, new_metadata, force_re_embed)
        operations["update_method"] = "full_re_embed" if (filename_changed or file_type_changed or force_re_embed) else "metadata_only"
        
        delete_metadata(doc_id)
        save_metadata(new_metadata)
        operations["metadata_updated"] = True
        
        response = {
            "message": "Document updated successfully" if operations["vector_updated"] and operations["metadata_updated"] else "Document partially updated",
            "_id": doc_id,
            "success": operations["vector_updated"] and operations["metadata_updated"],
            "updated_fields": {
                "filename": {"old": current_filename, "new": new_filename, "changed": filename_changed},
                "uploaded_by": {"old": current_doc.get('uploaded_by'), "new": new_uploaded_by, "changed": new_uploaded_by != current_doc.get('uploaded_by')},
                "file_type": {"old": current_file_type, "new": new_file_type, "changed": file_type_changed},
                "role": {"old": current_role, "new": new_role, "changed": new_role != current_role}
            },
            "operations": operations,
            "paths": {
                "old_file_path": current_file_path,
                "new_file_path": final_file_path
            },
            "updatedAt": datetime.now(timezone(timedelta(hours=7))).isoformat(),
            "force_re_embed": force_re_embed
        }
        
        if not operations["vector_updated"] or not operations["metadata_updated"]:
            response["warnings"] = []
            if not operations["vector_updated"]:
                response["warnings"].append("Vector embeddings update failed")
            if not operations["metadata_updated"]:
                response["warnings"].append("Metadata database update failed")
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating document: {str(e)}")


def standardization(distance: float) -> float:
    """Convert cosine distance to similarity score [0, 1]"""
    # Redis cosine distance is in range [0, 2]
    # 0 = identical, 2 = opposite
    # Convert to similarity: 1 - (distance / 2)
    if distance < 0:
        return 1.0
    elif distance > 2:
        return 0.0
    else:
        return 1.0 - (distance / 2.0)


@router.post("/search", response_model=VectorSearchResponse)
async def search_vector_documents(
    request: VectorSearchRequest,
    current_user: dict = Depends(verify_token_v2) 
):
    start_time = time.time()
    
    try:
        file_type_paths_data = Config.get_file_type_paths() 
        valid_file_types = list(file_type_paths_data.keys())
        
        if request.file_type not in valid_file_types:
            raise HTTPException(status_code=400, detail=f"Invalid file_type. Valid types: {valid_file_types}")
        
        # Get Redis index - FIXED
        index_name = get_index_name(request.file_type)
        client = get_redis_client()
        
        try:
            # Use from_existing instead of __init__
            index = SearchIndex.from_existing(index_name, redis_client=client)
        except Exception as e:
            logger.warning(f"No index found for {request.file_type}: {str(e)}")
            # No index exists for this file_type
            return VectorSearchResponse(
                query=request.query,
                results=[],
                total_found=0,
                k_requested=request.k,
                file_type=request.file_type,
                similarity_threshold=request.similarity_threshold,
                search_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        # Generate query embedding
        try:
            from app.services.embedding_service import get_embedding_model
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(request.query)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {str(e)}")
        
        # Perform vector search with RedisVL
        try:
            # Create vector query
            vector_query = VectorQuery(
                vector=query_embedding,
                vector_field_name="embedding",
                return_fields=["content", "doc_id", "filename", "file_type", "uploaded_by", "role_user", "role_subject", "created_at", "url"],
                num_results=request.k * 3  # Get more results for filtering
            )
            
            # Execute search
            results = index.query(vector_query)
            
            # Process results
            search_results = []
            for result in results:
                # Extract distance score
                distance = float(result.get('vector_distance', 2.0))
                similarity = standardization(distance)
                
                # Filter by threshold
                if similarity < request.similarity_threshold:
                    continue
                
                # Parse role fields back to lists
                role_user = result.get('role_user', '').split(',') if result.get('role_user') else []
                role_subject = result.get('role_subject', '').split(',') if result.get('role_subject') else []
                
                # Build metadata
                metadata = {
                    '_id': result.get('doc_id', ''),
                    'filename': result.get('filename', ''),
                    'file_type': result.get('file_type', ''),
                    'uploaded_by': result.get('uploaded_by', ''),
                    'role': {
                        'user': [u for u in role_user if u],  # Remove empty strings
                        'subject': [s for s in role_subject if s]
                    },
                    'createdAt': result.get('created_at', ''),
                    'url': result.get('url', ''),
                    'similarity_score': float(similarity)
                }
                
                search_results.append({
                    "content": result.get('content', ''),
                    "metadata": metadata
                })
            
            # Apply access control filtering
            accessible_results = filter_accessible_files(current_user, search_results)
            
            if not accessible_results:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied. You don't have permission to view any documents matching your search query."
                )
            
            # Take only top k results after permission filtering
            top_results = accessible_results[:request.k]
            
            results = [
                SearchResult(
                    content=result["content"], 
                    metadata=result["metadata"]
                )
                for result in top_results
            ]
            
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            logger.error(f"Search execution failed: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Search execution failed: {str(e)}")
        
        search_time_ms = round((time.time() - start_time) * 1000, 2)
        return VectorSearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            k_requested=request.k,
            file_type=request.file_type,
            similarity_threshold=request.similarity_threshold,
            search_time_ms=search_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/search-with-llm")
async def search_with_llm(request: VectorSearchRequest, current_user: dict = Depends(verify_token_v2)):
    start_time = time.time()

    try:
        # Get Redis index - FIXED
        index_name = get_index_name(request.file_type)
        client = get_redis_client()
        
        try:
            # Use from_existing instead of __init__
            index = SearchIndex.from_existing(index_name, redis_client=client)
        except Exception as e:
            logger.warning(f"No index found for {request.file_type}: {str(e)}")
            return {
                "llm_response": "Không tìm thấy tài liệu với thông tin được cung cấp.",
                "contexts": []
            }

        # Generate query embedding
        try:
            from app.services.embedding_service import get_embedding_model
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(request.query)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Không thể tạo embedding cho truy vấn: {str(e)}")

        # Perform vector search
        try:
            vector_query = VectorQuery(
                vector=query_embedding,
                vector_field_name="embedding",
                return_fields=["content", "doc_id", "filename", "file_type", "uploaded_by", "role_user", "role_subject", "created_at", "url"],
                num_results=request.k * 2
            )
            
            results = index.query(vector_query)
            
            # Process and filter results
            search_results = []
            for result in results:
                distance = float(result.get('vector_distance', 2.0))
                similarity = standardization(distance)
                
                if similarity < request.similarity_threshold:
                    continue
                
                role_user = result.get('role_user', '').split(',') if result.get('role_user') else []
                role_subject = result.get('role_subject', '').split(',') if result.get('role_subject') else []
                
                metadata = {
                    '_id': result.get('doc_id', ''),
                    'filename': result.get('filename', ''),
                    'file_type': result.get('file_type', ''),
                    'uploaded_by': result.get('uploaded_by', ''),
                    'role': {
                        'user': [u for u in role_user if u],
                        'subject': [s for s in role_subject if s]
                    },
                    'createdAt': result.get('created_at', ''),
                    'url': result.get('url', ''),
                    'similarity_score': float(similarity)
                }
                
                search_results.append({
                    "content": result.get('content', ''),
                    "metadata": metadata
                })
            
            # Apply access control filtering
            accessible_results = filter_accessible_files(current_user, search_results)
            
            # Take top k after permission filtering
            top_results = accessible_results[:request.k]

            # Generate LLM response
            llm_response = "Không tìm thấy tài liệu với thông tin được cung cấp."
            contexts = top_results

            if top_results:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    from langchain.prompts import PromptTemplate
                    
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-robotics-er-1.5-preview"
                    )

                    context = "\n\n".join(
                        [f"Tài liệu {i+1}:\n{result['content']}" for i, result in enumerate(top_results)]
                    )

                    prompt_template = PromptTemplate(
                        input_variables=["query", "context"],
                        template="""
🎯 Vai trò:
Bạn là một trợ lý AI chuyên nghiệp, chỉ trả lời dựa trên thông tin từ **tài liệu được cung cấp**.

📋 Nguyên tắc:
- Chỉ sử dụng thông tin từ tài liệu
- Không thêm kiến thức bên ngoài
- Không suy đoán hoặc giả định
- Nếu không có thông tin: "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu."

📝 Cấu trúc trả lời:
1. **Câu mở đầu**: Tóm tắt ngắn gọn (1-2 câu)
2. **Nội dung chính**: Trình bày bằng danh sách có số thứ tự hoặc gạch đầu dòng
3. **Kết luận** (nếu cần): Tóm lược hoặc lời khuyên

✨ **YÊU CẦU FORMAT MARKDOWN (QUAN TRỌNG)**:
- **BẮT BUỘC** sử dụng Markdown format để trả lời
- Dùng **số thứ tự** (1., 2., 3.) cho các bước hoặc quy trình
- Dùng **gạch đầu dòng** (-, *) cho danh sách các ý
- Dùng **bold** (** **) cho từ khóa quan trọng
- Dùng # ## ### cho tiêu đề phần (nếu cần)
- Dùng > cho trích dẫn từ tài liệu (nếu cần)
- Dùng ``` cho code hoặc ví dụ (nếu cần)
- Dùng bảng (| |) nếu có dữ liệu dạng bảng

❓ Câu hỏi của người dùng:
{query}

📂 Tài liệu tham khảo:
{context}

Hãy trả lời câu hỏi dựa trên tài liệu trên với format Markdown đẹp mắt và dễ đọc.
"""
                    )

                    prompt = prompt_template.format(query=request.query, context=context)
                    result = llm.invoke(prompt)
                    llm_response = result.content

                except Exception as e:
                    logger.error(f"Tạo phản hồi LLM thất bại: {str(e)}")
                    llm_response = "Không thể tạo phản hồi từ LLM."
                    contexts = []

            return {
                "llm_response": llm_response,
                "contexts": contexts
            }

        except Exception as e:
            logger.error(f"Tìm kiếm thất bại: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Tìm kiếm thất bại: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")


# === Pydantic Model mở rộng ===
class SearchWithContextRequest(VectorSearchRequest):
    session_id: Optional[str] = None
    disable_query_rewrite: bool = False  # Tùy chọn tắt rewriting


class SearchWithContextResponse(BaseModel):
    llm_response: str
    contexts: List[Dict[str, Any]]
    session_id: str
    history_used: bool = False
    history_count: int = 0
    query_rewritten: bool = False
    original_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    # Cache metrics
    embedding_cache_hit: bool = False
    llm_cache_hit: bool = False
    cache_distance: Optional[float] = None


# === Hàm chuẩn hóa similarity (giữ nguyên từ code cũ) ===
def standardization(distance: float) -> float:
    """Chuyển distance (0-2) thành similarity (0-1)"""
    return 1 - (distance / 2)


# app/routes/vector.py - Phần được cải tiến

# ... (giữ nguyên imports và các hàm trước đó)

# ============================================
# SESSION MANAGEMENT - SIMPLIFIED
# ============================================

_session_cache = {}
_session_lock = None
_query_rewriter = None

try:
    import threading
    _session_lock = threading.Lock()
except ImportError:
    class DummyLock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    _session_lock = DummyLock()

def get_query_rewriter() -> QueryRewriter:
    """Singleton cho QueryRewriter"""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter

async def ensure_message_history_index():
    """
    Tạo index CHUNG cho tất cả message history
    CHỈ GỌI 1 LẦN khi khởi động app!
    """
    redis_client = get_redis_client()
    index_name = "msg_history_idx"  # Index chung cho tất cả sessions
    
    try:
        index = SearchIndex.from_existing(index_name, redis_client=redis_client)
        logger.info("Đã tìm thấy msg_history_idx")
        return index
    except Exception:
        logger.info("Đang tạo msg_history_idx mới...")
    
    # Schema cho message history - dùng session_tag để phân biệt
    msg_schema = {
        "index": {
            "name": index_name,
            "prefix": "msg:",  # Prefix chung
            "storage_type": "json"
        },
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "embedding", "type": "vector", 
             "attrs": {
                 "dims": 768,
                 "distance_metric": "cosine",
                 "algorithm": "flat",
                 "datatype": "float32"
             }},
            {"name": "role", "type": "tag"},
            {"name": "session_tag", "type": "tag"},  # Phân biệt session qua tag
            {"name": "timestamp", "type": "numeric", "attrs": {"sortable": True}}
        ]
    }
    
    index = SearchIndex.from_dict(msg_schema)
    index.set_client(redis_client)
    index.create(overwrite=False)
    logger.info("Tạo msg_history_idx thành công")
    return index


def get_session_history(session_id: str) -> SemanticMessageHistory:
    """
    Lấy hoặc tạo session history
    Tất cả sessions dùng CHUNG 1 index, phân biệt nhau bằng session_tag
    """
    if session_id not in _session_cache:
        with _session_lock:
            if session_id not in _session_cache:
                vectorizer = HFTextVectorizer(
                    model="dangvantuan/vietnamese-document-embedding",
                    device="cuda",
                    trust_remote_code=True
                )
                
                # name = index name (dùng chung)
                # session_tag = unique session identifier
                _session_cache[session_id] = SemanticMessageHistory(
                    name="msg_history_idx",  # ← Index chung cho tất cả
                    session_tag=session_id,   # ← Phân biệt session
                    redis_client=get_redis_client(),
                    vectorizer=vectorizer,
                    ttl=60*60*24*7,  # 7 ngày
                    prefix="msg:",    # ← Prefix chung
                    overwrite=True    # ← Fix schema mismatch
                )
                logger.info(f"Tạo history mới cho session: {session_id}")
    return _session_cache[session_id]


# ============================================
# CONTEXT BUILDER - SIMPLIFIED & SMART
# ============================================

class ContextBuilder:
    """Xây dựng context thông minh cho LLM"""
    
    @staticmethod
    def build_document_context(results: List[Dict], max_tokens: int = None) -> str:
        """
        Tạo context từ documents (không giới hạn)
        
        Args:
            results: List documents từ vector search
            max_tokens: Deprecated - không sử dụng nữa
        """
        if not results:
            return "Không có tài liệu liên quan."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            content = result['content']
            filename = result['metadata']['filename']
            
            # Format document snippet (không cắt content)
            doc_snippet = f"**Tài liệu {i}: {filename}**\n{content}\n"
            context_parts.append(doc_snippet)
        
        return "\n".join(context_parts)
    
    @staticmethod
    def build_history_context(
        history: SemanticMessageHistory,
        current_query: str,
        max_messages: int = 3
    ) -> tuple[str, int]:
        """
        Lấy lịch sử chat liên quan (semantic search) - không giới hạn content
        
        Returns:
            (context_string, số_messages_sử_dụng)
        """
        try:
            # Tìm messages liên quan semantic
            relevant_msgs = history.get_relevant(
                prompt=current_query,
                top_k=max_messages,
                as_text=False
            )
            
            if not relevant_msgs:
                return "", 0
            
            # Format lịch sử (không cắt content)
            history_parts = ["**Lịch sử chat liên quan:**"]
            for msg in relevant_msgs[-max_messages:]:
                role = "Bạn" if msg.get('role') == 'user' else "AI"
                content = msg.get('content', '')  # Bỏ giới hạn [:300]
                history_parts.append(f"{role}: {content}")
            
            return "\n".join(history_parts), len(relevant_msgs)
        
        except Exception as e:
            logger.warning(f"Không lấy được lịch sử: {e}")
            return "", 0
    



@router.post("/search-with-llm-context", response_model=SearchWithContextResponse)
async def search_with_llm_context(
    request: SearchWithContextRequest,
    current_user: dict = Depends(verify_token_v2)
):
    """
    Tìm kiếm documents + tích hợp lịch sử chat + Query Rewriting
    
    Flow:
    1. Lấy lịch sử chat (nếu có session)
    2. **Query Rewriting: Viết lại câu hỏi dựa trên context**
    3. Vector search với query đã được cải thiện
    4. Build context
    5. Gọi LLM
    6. Lưu vào session
    """
    start_time = time.time()
    user_id = str(current_user.get("id"))
    original_query = request.query 
    
    try:
        # ========================================
        # 1. QUẢN LÝ SESSION
        # ========================================
        session_id = request.session_id
        if not session_id:
            session_id = f"usr:{user_id}:sess:{int(time.time())}:{str(uuid.uuid4())[:8]}"
            logger.info(f"Session mới: {session_id}")
        else:
            logger.info(f"Tiếp tục session: {session_id}")
        
        # ========================================
        # 2. QUERY REWRITING & HISTORY MANAGEMENT
        # ========================================
        query_rewritten = False
        rewritten_query = original_query
        history = None  # Initialize history variable
        history_context = ""
        history_count = 0
        
        if not request.disable_query_rewrite:
            try:
                # Lấy lịch sử 1 LẦN DUY NHẤT - dùng cho cả rewriting và LLM context
                history = get_session_history(session_id)
                logger.info("Đã lấy history (sẽ dùng cho rewriting + LLM context)")
                
                # FIXED: Dùng API của SemanticMessageHistory để lấy messages
                try:
                    # Chỉ lấy 3 câu hỏi gần nhất (role='user') để giảm context
                    all_messages = history.get_recent(top_k=10, as_text=False)  # Lấy 10 để filter ra 3 user messages
                    
                    # Convert và chỉ giữ lại user messages
                    history_messages = []
                    for msg in all_messages:
                        try:
                            role = None
                            content = None
                            
                            # Case 1: Dict format
                            if isinstance(msg, dict):
                                role = msg.get('role', 'user')
                                content = msg.get('content', '')
                            # Case 2: LangChain BaseMessage
                            elif hasattr(msg, 'type') and hasattr(msg, 'content'):
                                role = 'user' if msg.type == 'human' else 'assistant'
                                content = msg.content
                            # Case 3: String (fallback)
                            elif isinstance(msg, str):
                                role = 'user'
                                content = msg
                            else:
                                logger.warning(f"Unknown message type: {type(msg)}")
                                continue
                            
                            # Chỉ lấy user messages (câu hỏi)
                            if role == 'user' and content:
                                history_messages.append({
                                    'role': 'user',
                                    'content': content
                                })
                                
                                # Dừng khi đủ 3 câu hỏi
                                if len(history_messages) >= 5:
                                    break
                                    
                        except Exception as parse_error:
                            logger.warning(f"Không parse được message: {parse_error}")
                            continue
                    
                    logger.info(f"  Lấy được {len(history_messages)} câu hỏi cho rewriting")
                    
                except Exception as e:
                    logger.warning(f"Không lấy được history: {e}")
                    history_messages = []
                
                # Thực hiện rewriting
                if history_messages:
                    rewriter = get_query_rewriter()
                    rewritten_query, query_rewritten = rewriter.rewrite(
                        original_query, 
                        history_messages
                    )
                    
                    if query_rewritten:
                        logger.info(f"   Query rewriting:")
                        logger.info(f"   Original: {original_query}")
                        logger.info(f"   Rewritten: {rewritten_query}")
                else:
                    logger.info("Không có lịch sử, bỏ qua rewriting")
                    
            except Exception as e:
                logger.warning(f"Query rewriting thất bại: {e}, dùng query gốc")
                rewritten_query = original_query
                query_rewritten = False
        else:
            logger.info("Query rewriting bị tắt → BỎ QUA history context")
        
        # ========================================
        # 3. VECTOR SEARCH (dùng rewritten query!)
        # ========================================
        index_name = get_index_name(request.file_type)
        client = get_redis_client()
        
        try:
            index = SearchIndex.from_existing(index_name, redis_client=client)
        except Exception as e:
            logger.warning(f"Không tìm thấy index {request.file_type}: {e}")
            return SearchWithContextResponse(
                llm_response="Không tìm thấy tài liệu nào.",
                contexts=[],
                session_id=session_id,
                history_used=False,
                history_count=0,
                query_rewritten=query_rewritten,
                original_query=original_query if query_rewritten else None,
                rewritten_query=rewritten_query if query_rewritten else None
            )
        
        # ========================================
        # EMBEDDING CACHE CHECK
        # ========================================
        embedding_cache = get_embedding_cache()
        embedding_cache_hit = False
        query_embedding = None
        
        # Try cache first
        model_name = "dangvantuan/vietnamese-document-embedding"
        cached_embedding = embedding_cache.get(rewritten_query, model_name)
        
        if cached_embedding is not None:
            query_embedding = cached_embedding
            embedding_cache_hit = True
            logger.info(f"  Embedding cache HIT")
        else:
            # Generate embedding and cache it
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(rewritten_query)
            
            # Store in cache
            embedding_cache.store(
                text=rewritten_query,
                model_name=model_name,
                embedding=query_embedding,
                metadata={"timestamp": time.time()}
            )
            embedding_cache_hit = False
            logger.info(f"  Embedding cache MISS, stored new embedding") 
        
        # Search
        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="embedding",
            return_fields=[
                "content", "doc_id", "filename", "file_type",
                "uploaded_by", "role_user", "role_subject",
                "created_at", "url"
            ],
            # num_results=request.k * 2
            num_results=request.k
        )
        results = index.query(vector_query)
        
        # Process & filter results (giữ nguyên phần này)
        search_results = []
        for result in results:
            distance = float(result.get('vector_distance', 2.0))
            similarity = standardization(distance)
            
            if similarity < request.similarity_threshold:
                continue
            
            role_user = result.get('role_user', '').split(',') if result.get('role_user') else []
            role_subject = result.get('role_subject', '').split(',') if result.get('role_subject') else []
            
            metadata = {
                '_id': result.get('doc_id', ''),
                'filename': result.get('filename', ''),
                'file_type': result.get('file_type', ''),
                'uploaded_by': result.get('uploaded_by', ''),
                'role': {
                    'user': [u.strip() for u in role_user if u.strip()],
                    'subject': [s.strip() for s in role_subject if s.strip()]
                },
                'createdAt': result.get('created_at', ''),
                'url': result.get('url', ''),
                'similarity_score': round(float(similarity), 4)
            }
            
            search_results.append({
                "content": result.get('content', ''),
                "metadata": metadata
            })
        
        # Access control
        accessible_results = filter_accessible_files(current_user, search_results)
        top_results = accessible_results[:request.k]
        
        # ========================================
        # 4. BUILD CONTEXT
        # ========================================
        builder = ContextBuilder()
        
        # Document context
        doc_context = builder.build_document_context(top_results, max_tokens=3000)
        
        # History context - CHỈ KHI có history (query rewrite enabled)
        if history is not None:
            history_context, history_count = builder.build_history_context(
                history,
                rewritten_query,  # Dùng rewritten query cho semantic search
                max_messages=3
            )
            logger.info(f"Đã build history context: {history_count} messages")
        else:
            logger.info("Bỏ qua history context (query rewrite disabled)")
        
        history_used = history_count > 0
        
        # Combine contexts
        if history_context:
            full_context = f"{history_context}\n\n---\n\n{doc_context}"
        else:
            full_context = doc_context
        
        # ========================================
        # 5. GỌI LLM (với LLM CACHE)
        # ========================================
        llm_response = "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
        llm_cache_hit = False
        cache_distance = None
        
        if top_results or history_used:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain.prompts import PromptTemplate
                
                # Build the full prompt
                prompt_template = PromptTemplate(
                    input_variables=["query", "context"],
                    template="""Bạn là trợ lý AI chuyên nghiệp.

**Nguyên tắc:**
- Chỉ trả lời dựa trên ngữ cảnh bên dưới và khi trả lời không cần nhắc đến ngữ cảnh
- Không thêm thông tin bên ngoài
- Trả lời tự nhiên, dễ hiểu
- Có thể tham khảo lịch sử chat nếu câu hỏi liên quan

**YÊU CẦU FORMAT MARKDOWN (QUAN TRỌNG):**
- **BẮT BUỘC** sử dụng Markdown format để trả lời
- Dùng **số thứ tự** (1., 2., 3.) cho các bước hoặc quy trình
- Dùng **gạch đầu dòng** (-, *) cho danh sách các ý
- Dùng **bold** (** **) cho từ khóa quan trọng
- Dùng # ## ### cho tiêu đề phần (nếu cần)
- Dùng > cho trích dẫn (nếu cần)
- Dùng ``` cho code hoặc ví dụ (nếu cần)
- Dùng bảng (| |) nếu có dữ liệu dạng bảng

**Câu hỏi của người dùng:**
{query}

**Ngữ cảnh:**
{context}

---
Hãy trả lời câu hỏi dựa trên ngữ cảnh trên với format Markdown đẹp mắt và dễ đọc."""
                )
                
                # Use ORIGINAL query for natural interaction
                full_prompt = prompt_template.format(
                    query=original_query,
                    context=full_context
                )
                
                # ========================================
                # LLM CACHE CHECK
                # ========================================
                llm_cache = get_llm_cache()
                cached_result = llm_cache.check(full_prompt)
                
                if cached_result:
                    # Cache HIT - use cached response
                    llm_response = cached_result.get('response')
                    cache_distance = cached_result.get('distance', 0.0)
                    llm_cache_hit = True
                    logger.info(f"  LLM cache HIT (distance: {cache_distance:.4f})")
                else:
                    # Cache MISS - call LLM and cache result
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-robotics-er-1.5-preview",
                        temperature=0.3
                    )
                    
                    result = llm.invoke(full_prompt)
                    llm_response = result.content
                    
                    # Store in cache
                    llm_cache.store(
                        prompt=full_prompt,
                        response=llm_response,
                        metadata={
                            "model": "gemini-robotics-er-1.5-preview",
                            "timestamp": time.time(),
                            "original_query": original_query
                        }
                    )
                    llm_cache_hit = False
                    logger.info(f"  LLM cache MISS, called LLM and cached response")
                
            except Exception as e:
                logger.error(f"LLM thất bại: {e}")
                llm_response = "Không thể tạo phản hồi từ AI."
        
        # ========================================
        # 6. LƯU VÀO SESSION (lưu original query!)
        # ========================================
        try:
            history = get_session_history(session_id)
            # Lưu query gốc vào lịch sử
            history.add_message({"role": "user", "content": original_query})
            history.add_message({"role": "assistant", "content": llm_response})
            logger.info(f"Đã lưu vào session {session_id}")
        except Exception as e:
            logger.warning(f"Lưu session thất bại: {e}")
        
        # ========================================
        # 7. TRACKING QUERY (MongoDB + Kafka)
        # ========================================
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        
        try:
            # Extract user_id from JWT token
            user_id_int = current_user.get("user_id")
            
            # Lookup comprehensive user metadata from MongoDB
            user_info = get_user_metadata_from_db(user_id_int)
            
            logger.info(f"Query tracking metadata - "
                       f"User: {user_info.get('user_type')} | "
                       f"Code: {user_info.get('code')} | "
                       f"Years: {user_info.get('years')}")
            
            # Log query to MongoDB (async, non-blocking)
            tracker = get_query_tracker()
            query_id = await tracker.log_query(
                user_info=user_info,
                session_id=session_id,
                query_text=original_query,
                rewritten_query=rewritten_query if query_rewritten else None,
                k=request.k,
                similarity_threshold=request.similarity_threshold,
                context_found=len(top_results),
                response_time_ms=response_time_ms,
                llm_response=llm_response,
                model_used="gemini-robotics-er-1.5-preview",
                query_rewritten=query_rewritten,
                history_used=history_used,
                history_count=history_count
            )
            
            if query_id:
                logger.info(f"Query tracked: {query_id[:8]}...")
            else:
                logger.warning("Query tracking failed (non-critical)")
                
        except Exception as e:
            # Don't fail the request if tracking fails
            logger.error(f"Query tracking error: {e}")
        return SearchWithContextResponse(
            llm_response=llm_response,
            contexts=top_results,
            session_id=session_id,
            history_used=history_used,
            history_count=history_count,
            query_rewritten=query_rewritten,
            original_query=original_query if query_rewritten else None,
            rewritten_query=rewritten_query if query_rewritten else None,
            embedding_cache_hit=embedding_cache_hit,
            llm_cache_hit=llm_cache_hit,
            cache_distance=cache_distance
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")


# ============================================
# HELPER ENDPOINT: XÓA LỊCH SỬ SESSION
# ============================================

@router.delete("/session/{session_id}")
async def clear_session_history(
    session_id: str,
    current_user: dict = Depends(verify_token_v2)
):
    """Xóa toàn bộ lịch sử của 1 session"""
    try:
        if session_id in _session_cache:
            history = _session_cache[session_id]
            # Clear all messages
            redis_client = get_redis_client()
            pattern = f"msg:{session_id}:*"
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
            
            # Remove from cache
            del _session_cache[session_id]
            
            return {
                "message": f"Đã xóa {len(keys)} messages từ session {session_id}",
                "session_id": session_id,
                "deleted_count": len(keys)
            }
        else:
            raise HTTPException(status_code=404, detail="Session không tồn tại")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xóa session: {str(e)}")