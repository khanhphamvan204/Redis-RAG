# main.py
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import các router
from app.routes import health, documents, vector, folder 
from app.routes.folder import startup_db_client, shutdown_db_client
from app.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    """
    Hàm này sẽ được gọi khi ứng dụng FastAPI khởi động.
    """
    logger.info("Sự kiện startup: Đang khởi tạo kết nối CSDL...")
    await startup_db_client(app)

@app.on_event("shutdown")
async def on_shutdown():
    """
    Hàm này sẽ được gọi khi ứng dụng FastAPI tắt.
    """
    logger.info("Sự kiện shutdown: Đang đóng kết nối CSDL...")
    await shutdown_db_client(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure environment
try:
    os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY    
except AttributeError as e:
    logger.error(f"Config error: {str(e)}")
    raise Exception("Configuration error: Missing GEMINI_API_KEY")

# Include routes
app.include_router(health.router, prefix="/health")
app.include_router(documents.router, prefix="/documents")
app.include_router(vector.router, prefix="/documents/vector")
app.include_router(folder.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)