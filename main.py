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
from app.api.routes import analytics
from app.routes.folder import startup_db_client, shutdown_db_client
from app.config import Config
from app.services.kafka_service import initialize_kafka, shutdown_kafka
from app.routes.vector import ensure_message_history_index

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
    logger.info("Sự kiện startup: Đang khởi tạo các services...")
    
    # 1. Database
    await startup_db_client(app)
    
    # 2. Message history index
    try:
        await ensure_message_history_index()
        logger.info("✓ Message history index initialized")
    except Exception as e:
        logger.warning(f"Message history init failed: {e}")
    
    # 3. Kafka producer
    try:
        await initialize_kafka()
        logger.info("✓ Kafka producer initialized")
    except Exception as e:
        logger.warning(f"Kafka init failed (non-critical): {e}")

@app.on_event("shutdown")
async def on_shutdown():
    """
    Hàm này sẽ được gọi khi ứng dụng FastAPI tắt.
    """
    logger.info("Sự kiện shutdown: Đang đóng các kết nối...")
    
    # Shutdown Kafka
    try:
        await shutdown_kafka()
    except Exception as e:
        logger.error(f"Kafka shutdown error: {e}")
    
    # Shutdown database
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
# Changed from /api to /analytics to avoid conflict with folder.py's /api routes
app.include_router(analytics.router)

# WebSocket endpoint for real-time analytics
from fastapi import WebSocket, WebSocketDisconnect
from app.services.websocket_manager import ws_manager
from app.services.redis_analytics_service import (
    get_faculty_analytics,
    get_year_analytics,
    get_heatmap_analytics,
    subscribe_to_updates,
    get_popular_questions,
    get_department_analytics
)
import asyncio
import json


@app.websocket("/ws/analytics")
async def websocket_analytics_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time analytics updates
    Sends initial data and then pushes updates via Redis Pub/Sub
    """
    await ws_manager.connect(websocket)
    
    try:
        # Send initial data
        initial_data = {
            "type": "initial",
            "data": {
                "faculty": get_faculty_analytics(days=30),
                "year": get_year_analytics(days=30),
                "heatmap": get_heatmap_analytics(days=30),
                "popularQuestions": get_popular_questions(days=30, limit=10),
                "departments": get_department_analytics(days=30)
            }
        }
        await ws_manager.send_personal_message(initial_data, websocket)
        logger.info("Sent initial analytics data to WebSocket client")
        
        # Subscribe to Redis Pub/Sub for updates
        pubsub = subscribe_to_updates()
        
        if pubsub:
            # Listen for updates
            while True:
                message = pubsub.get_message()
                if message and message['type'] == 'message':
                    # Parse update notification
                    update_data = json.loads(message['data'])
                    
                    # Fetch fresh data for the updated analytics type
                    analytics_type = update_data.get('type')
                    fresh_data = {}
                    
                    if analytics_type == 'faculty':
                        fresh_data['faculty'] = get_faculty_analytics(days=30)
                    elif analytics_type == 'year':
                        fresh_data['year'] = get_year_analytics(days=30)
                    elif analytics_type == 'heatmap':
                        fresh_data['heatmap'] = get_heatmap_analytics(days=30)
                    elif analytics_type == 'popular_queries':
                        fresh_data['popularQuestions'] = get_popular_questions(days=30, limit=10)
                    
                    # Send update to client
                    await ws_manager.send_personal_message({
                        "type": "update",
                        "analytics_type": analytics_type,
                        "data": fresh_data,
                        "timestamp": update_data.get('timestamp')
                    }, websocket)
                    
                await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting
        else:
            # Fallback: Keep connection alive with heartbeats
            while True:
                await asyncio.sleep(30)
                await ws_manager.send_personal_message({"type": "heartbeat"}, websocket)
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)