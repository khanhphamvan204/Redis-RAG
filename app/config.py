import os
import logging
from dotenv import load_dotenv
import pymongo # <-- THAY ĐỔI: Import pymongo
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError # <-- THAY ĐỔI

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    # Các biến cấu hình tĩnh vẫn giữ nguyên
    DATA_PATH = os.getenv("DATA_PATH", "data")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore/db_faiss")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
    
    # Lấy hằng số từ file folder.py để đảm bảo nhất quán
    _AGENT_DB_NAME = "faiss_db"
    _FOLDER_COLLECTION = "folders"

    @staticmethod
    def get_file_type_paths():
        """
        Phương thức này sẽ KẾT NỐI TRỰC TIẾP TỚI MONGODB để lấy danh sách thư mục.
        Nếu lỗi, nó sẽ sử dụng một danh sách mặc định để dự phòng.
        """
        DEFAULT_FOLDERS = ['admin', 'teacher', 'student', 'public']
        folder_names = []
        client = None # Khởi tạo client bên ngoài để có thể đóng trong 'finally'
        
        try:
            # 1. Kết nối trực tiếp đến MongoDB
            # Đảm bảo DATABASE_URL trong .env là:
            # mongodb://admin:123@mongo:27017/faiss_db?authSource=admin
            db_url = Config.DATABASE_URL
            
            # Sử dụng pymongo (client đồng bộ)
            client = pymongo.MongoClient(
                db_url, 
                serverSelectionTimeoutMS=5000 # 5 giây timeout kết nối
            )
            
            # Kiểm tra kết nối
            client.admin.command('ping')
            
            # 2. Truy vấn collection 'folders'
            db = client[Config._AGENT_DB_NAME]
            collection = db[Config._FOLDER_COLLECTION]
            
            cursor = collection.find({}, {"folder_name": 1, "_id": 0})
            folder_names = [doc["folder_name"] for doc in cursor]
            
            if folder_names:
                logging.info(f"Đã tải lại cấu hình thư mục từ DB: {folder_names}")
            else:
                # Vẫn kết nối được nhưng không có dữ liệu
                logging.warning("Truy vấn DB thành công nhưng không tìm thấy folder nào. Dùng mặc định.")
                folder_names = DEFAULT_FOLDERS
                
        except (PyMongoError, ServerSelectionTimeoutError, Exception) as e:
            # 3. Xử lý lỗi nếu không kết nối được
            logging.warning(f"Không thể lấy cấu hình từ MongoDB ({e}). Sử dụng danh sách mặc định.")
            folder_names = DEFAULT_FOLDERS
        
        finally:
            # 4. Luôn đóng kết nối
            if client:
                client.close()

        # Tạo dictionary chứa các đường dẫn (phần này giữ nguyên)
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