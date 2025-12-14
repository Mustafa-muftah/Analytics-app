from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import os
import logging

from core import settings, init_db, init_async_db, logger
from services.video import camera_manager
from api.routes import cameras_router, analytics_router, reports_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def realtime_processing_task(camera_id: str):
    """Background task for real-time camera processing"""
    from core import async_session, PeopleCount, ZoneStat
    
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        return
    
    logger.info(f"[{camera_id}] Starting real-time processing")
    
    while True:
        try:
            count, frame, stats = camera.process_frame()
            
            async with async_session() as session:
                session.add(PeopleCount(
                    camera_id=camera_id,
                    timestamp=datetime.utcnow(),
                    count=count,
                    hour=datetime.utcnow().hour
                ))
                session.add(ZoneStat(
                    camera_id=camera_id,
                    zone=camera.config.zone,
                    timestamp=datetime.utcnow(),
                    hour=datetime.utcnow().hour,
                    total_count=stats.get("unique_visitors", count),
                    avg_dwell_time=stats.get("avg_dwell_time", 0)
                ))
                await session.commit()
            
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"[{camera_id}] Processing error: {e}")
            await asyncio.sleep(10)


async def heatmap_generation_task():
    """Background task to generate heatmaps periodically"""
    from core import async_session, Heatmap
    
    await asyncio.sleep(60)
    
    while True:
        try:
            for camera in camera_manager.get_all_cameras():
                if not camera.config.is_batch_mode():
                    url, proc_time = camera.generate_heatmap()
                    if url:
                        async with async_session() as session:
                            session.add(Heatmap(
                                camera_id=camera.camera_id,
                                timestamp=datetime.utcnow(),
                                s3_url=url,
                                processing_time=proc_time
                            ))
                            await session.commit()
            
            await asyncio.sleep(settings.heatmap_interval)
        except Exception as e:
            logger.error(f"Heatmap error: {e}")
            await asyncio.sleep(settings.heatmap_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Retail Analytics API (Modular Architecture)...")
    
    # Initialize database
    init_db()
    await init_async_db()
    
    # Create directories
    os.makedirs("heatmaps", exist_ok=True)
    
    # Initialize camera manager
    camera_manager.initialize()
    camera_manager.load_from_settings()
    
    # Start background tasks for real-time cameras
    for camera in camera_manager.get_all_cameras():
        if not camera.config.is_batch_mode():
            asyncio.create_task(realtime_processing_task(camera.camera_id))
    
    # Start heatmap generation
    asyncio.create_task(heatmap_generation_task())
    
    logger.info(f"Started with {len(camera_manager.cameras)} cameras")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    camera_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Retail Video Analytics API",
    description="Modular architecture for retail analytics",
    version="2.1.0",
    lifespan=lifespan
)

# Static files
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cameras_router)
app.include_router(analytics_router)
app.include_router(reports_router)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Retail Video Analytics API",
        "version": "2.1.0",
        "architecture": "modular",
        "timestamp": datetime.utcnow().isoformat(),
        "cameras": len(camera_manager.cameras)
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "cameras": len(camera_manager.cameras),
        "services": {
            "video": "running",
            "analytics": "running",
            "reports": "running",
            "notifications": "disabled"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)