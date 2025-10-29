# app/routes/vector.py
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from app.services.embedding_service import (
    add_to_embedding, delete_from_redis_index, get_embedding_model, 
    smart_metadata_update, get_redis_client, get_index_name
)
from app.services.metadata_service import save_metadata, delete_metadata, find_document_info
from app.services.file_service import get_file_paths
from app.services.auth_service import verify_token_v2, filter_accessible_files
from app.config import Config
from pydantic import BaseModel, Field
from redisvl.query import VectorQuery
from redisvl.index import SearchIndex
import os
import json
import uuid
from datetime import datetime, timezone, timedelta
import shutil
import logging
import time
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

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
            
            # Take top k
            top_results = search_results[:request.k]

            # Generate LLM response
            llm_response = "Không tìm thấy tài liệu với thông tin được cung cấp."
            contexts = top_results

            if top_results:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    from langchain.prompts import PromptTemplate
                    
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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

💡 Format markdown:
- Dùng **số thứ tự** (1., 2., 3.) cho các bước hoặc quy trình
- Dùng **gạch đầu dòng** (-, *, •) cho danh sách các ý
- Dùng **bold** cho từ khóa quan trọng
- Dùng > cho trích dẫn từ tài liệu (nếu cần)

❓ Câu hỏi của người dùng:
{query}

📂 Tài liệu tham khảo:
{context}

Hãy trả lời câu hỏi dựa trên tài liệu trên.
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