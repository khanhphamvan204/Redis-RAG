# app/services/file_service.py
import os
from fastapi import HTTPException
from app.config import Config
import logging

logger = logging.getLogger(__name__)

def get_file_paths(file_type: str, filename: str) -> tuple[str, str]:
    current_file_type_paths = Config.get_file_type_paths()
    
    if file_type not in current_file_type_paths:
        raise HTTPException(
            status_code=400,
            # Lấy danh sách keys từ biến đã lưu
            detail=f"Invalid file_type: {file_type}. Must be one of: {list(current_file_type_paths.keys())}"
        )
    
    base_path = Config.DATA_PATH
    
    path_info = current_file_type_paths[file_type]
    file_folder = path_info['file_folder']
    vector_folder = path_info['vector_folder']
    
    file_path = os.path.join(base_path, file_folder, filename).replace("\\", "/")
    vector_db_path = os.path.join(base_path, vector_folder).replace("\\", "/")
    
    return file_path, vector_db_path