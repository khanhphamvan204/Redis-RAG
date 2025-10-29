
import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    # Các biến cấu hình tĩnh vẫn giữ nguyên
    DATA_PATH = os.getenv("DATA_PATH", "data")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore/db_faiss")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
    
    @staticmethod
    def get_file_type_paths():
        """
        Phương thức này sẽ gọi API để lấy danh sách thư mục mỗi khi nó được gọi.
        Nếu API lỗi, nó sẽ sử dụng một danh sách mặc định để dự phòng.
        """
        API_URL = "https://ai-api.bitech.vn/agent-list/api/folders/list"
        DEFAULT_FOLDERS = ['admin', 'teacher', 'student', 'public']
        
        folder_names = []
        
        try:
            response = requests.get(API_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
            api_folders = data.get('folders')
            
            if api_folders and isinstance(api_folders, list):
                folder_names = api_folders
                logging.info(f"Đã tải lại cấu hình thư mục từ API: {folder_names}")
            else:
                raise ValueError("Định dạng response từ API không hợp lệ.")
                
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.warning(f"Không thể lấy cấu hình từ API ({e}). Sử dụng danh sách mặc định.")
            folder_names = DEFAULT_FOLDERS

        # Tạo dictionary chứa các đường dẫn
        generated_paths = {}
        for name in folder_names:
            base_path_name = f"{name.capitalize()}_Rag_Info"
            env_file_key = f"{name.upper()}_FILE_FOLDER"
            env_vector_key = f"{name.upper()}_VECTOR_FOLDER"
            
            generated_paths[name] = {
                'file_folder': os.getenv(env_file_key, f"{base_path_name}/File_Folder"),
                'vector_folder': os.getenv(env_vector_key, f"{base_path_name}/Faiss_Folder")
            }
            
        return generated_paths

# import os
# import requests
# import logging
# import shutil
# from dotenv import load_dotenv

# load_dotenv()

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class Config:
#     DATA_PATH = os.getenv("DATA_PATH", "data")
#     VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore/db_faiss")
#     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#     DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
    
#     @staticmethod
#     def get_file_type_paths():
#         """
#         Phương thức này sẽ gọi API để lấy danh sách thư mục mỗi khi nó được gọi.
#         Nếu API lỗi, nó sẽ sử dụng một danh sách mặc định để dự phòng.
#         Xóa các thư mục trong rootfolder nếu không tồn tại trong danh sách từ API.
#         """
#         API_URL = "https://ai-api.bitech.vn/agent-list/api/folders/list"
#         DEFAULT_FOLDERS = ['admin', 'teacher', 'student', 'public']
#         ROOT_FOLDER = Config.DATA_PATH

#         folder_names = []
        
#         try:
#             response = requests.get(API_URL, timeout=5)
#             response.raise_for_status()
#             data = response.json()
#             api_folders = data.get('folders')
            
#             if api_folders and isinstance(api_folders, list):
#                 # Chuẩn hóa tên thư mục: thêm hậu tố _Rag_Info
#                 folder_names = [f"{name.capitalize()}_Rag_Info" for name in api_folders]
#                 logging.info(f"Đã tải lại cấu hình thư mục từ API: {folder_names}")
#             else:
#                 raise ValueError("Định dạng response từ API không hợp lệ.")
                
#         except (requests.exceptions.RequestException, ValueError) as e:
#             logging.warning(f"Không thể lấy cấu hình từ API ({e}). Sử dụng danh sách mặc định.")
#             folder_names = [f"{name.capitalize()}_Rag_Info" for name in DEFAULT_FOLDERS]

#         # Kiểm tra và xóa các thư mục không còn trong danh sách từ API
#         try:
#             existing_folders = [f for f in os.listdir(ROOT_FOLDER) if os.path.isdir(os.path.join(ROOT_FOLDER, f))]
#             logging.info(f"Các thư mục hiện có trong {ROOT_FOLDER}: {existing_folders}")

#             for folder in existing_folders:
#                 if folder not in folder_names:
#                     folder_path = os.path.join(ROOT_FOLDER, folder)
#                     if os.path.exists(folder_path):
#                         shutil.rmtree(folder_path)
#                         logging.info(f"Đã xóa thư mục không còn trong danh sách API: {folder_path}")

#         except Exception as e:
#             logging.error(f"Lỗi khi kiểm tra/xóa thư mục trong {ROOT_FOLDER}: {str(e)}")

#         # Tạo dictionary chứa các đường dẫn
#         generated_paths = {}
#         for name in [f.replace('_Rag_Info', '').lower() for f in folder_names]:  # Lấy tên gốc để làm key
#             base_path_name = f"{name.capitalize()}_Rag_Info"
#             env_file_key = f"{name.upper()}_FILE_FOLDER"
#             env_vector_key = f"{name.upper()}_VECTOR_FOLDER"
            
#             generated_paths[name] = {
#                 'file_folder': os.getenv(env_file_key, os.path.join(ROOT_FOLDER, f"{base_path_name}/File_Folder")),
#                 'vector_folder': os.getenv(env_vector_key, os.path.join(ROOT_FOLDER, f"{base_path_name}/Faiss_Folder"))
#             }
            
#         return generated_paths