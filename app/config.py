import os
import logging
from dotenv import load_dotenv
import pymongo
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    DATA_PATH = os.getenv("DATA_PATH", "data")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore/db_faiss")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
    
    _AGENT_DB_NAME = "faiss_db"
    _FOLDER_COLLECTION = "folders"

    @staticmethod
    def get_file_type_paths():
        """
        Lấy danh sách file_type → đường dẫn thư mục từ MongoDB.
        Nếu lỗi → dùng mặc định.
        KHÔNG THÊM _Rag_Info NỮA → DÙNG TÊN THẬT TRONG DB
        """
        DEFAULT_FOLDERS = ['admin', 'teacher', 'student', 'public']
        folder_names = []
        client = None
        
        try:
            db_url = Config.DATABASE_URL
            client = pymongo.MongoClient(db_url, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            
            db = client[Config._AGENT_DB_NAME]
            collection = db[Config._FOLDER_COLLECTION]
            
            cursor = collection.find({}, {"folder_name": 1, "_id": 0})
            folder_names = [doc["folder_name"] for doc in cursor]
            
            if folder_names:
                logging.info(f"Đã tải lại cấu hình thư mục từ DB: {folder_names}")
            else:
                logging.warning("Không tìm thấy folder nào trong DB. Dùng mặc định.")
                folder_names = DEFAULT_FOLDERS
                
        except (PyMongoError, ServerSelectionTimeoutError) as e:
            logging.warning(f"Lỗi kết nối MongoDB ({type(e).__name__}): {e}. Dùng mặc định.")
            folder_names = DEFAULT_FOLDERS
        except Exception as e:
            logging.error(f"Lỗi không xác định khi lấy folder: {e}")
            folder_names = DEFAULT_FOLDERS
        finally:
            if client:
                client.close()

        # TẠO ĐƯỜNG DẪN DỰA TRÊN folder_name THẬT (KHÔNG THÊM _Rag_Info)
        generated_paths = {}
        for name in folder_names:
            # DÙNG TÊN THẬT TRONG DB → THƯ MỤC THẬT TRÊN SERVER
            file_folder = f"{name}/File_Folder"
            vector_folder = f"vector_db_{name.split('_')[-1].lower()}"  # Giữ logic cũ nếu cần

            # Cho phép override bằng .env (tùy chọn)
            env_file_key = f"{name.upper()}_FILE_FOLDER"
            env_vector_key = f"{name.upper()}_VECTOR_FOLDER"
            
            generated_paths[name] = {
                'file_folder': os.getenv(env_file_key, file_folder),
                'vector_folder': os.getenv(env_vector_key, vector_folder)
            }
            
        return generated_paths