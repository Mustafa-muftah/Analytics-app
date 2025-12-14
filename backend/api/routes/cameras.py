from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio
import logging

from core import (
    get_db, async_session, 
    PeopleCount, GenderStat, ZoneStat, DwellTimeStats, Heatmap, ProcessingJob
)
from services.video import camera_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["cameras"])

# Track batch tasks
batch_tasks: Dict[str, asyncio.Task] = {}


async def clear_camera_data(camera_id: str):
    """Clear all data for a camera before batch processing"""
    async with async_session() as session:
        await session.execute(delete(PeopleCount).where(PeopleCount.camera_id == camera_id))
        await session.execute(delete(GenderStat).where(GenderStat.camera_id == camera_id))
        await session.execute(delete(ZoneStat).where(ZoneStat.camera_id == camera_id))
        await session.execute(delete(DwellTimeStats).where(DwellTimeStats.camera_id == camera_id))
        await session.execute(delete(Heatmap).where(Heatmap.camera_id == camera_id))
        await session.execute(delete(ProcessingJob).where(ProcessingJob.camera_id == camera_id))
        await session.commit()
    logger.info(f"[{camera_id}] Cleared previous data")


async def batch_processing_task(camera_id: str, job_id: int):
    """Background task for batch video processing"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        return
    
    logger.info(f"[{camera_id}] Starting batch processing (Job #{job_id})")
    
    async with async_session() as session:
        job = await session.get(ProcessingJob, job_id)
        if job:
            job.status = "processing"
            job.started_at = datetime.utcnow()
            await session.commit()
    
    processed = 0
    
    try:
        while True:
            count, frame, stats = camera.process_frame()
            
            if stats.get("skipped"):
                continue
            
            if count == -1:
                break
            
            processed += 1
            
            # Save stats every 10 frames
            if processed % 10 == 0:
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
                    
                    job = await session.get(ProcessingJob, job_id)
                    if job:
                        job.processed_frames = camera.frame_count
                    
                    await session.commit()
            
            await asyncio.sleep(0.01)
        
        # Save final stats
        visitor_stats = camera.processor.get_visitor_stats()
        
        async with async_session() as session:
            session.add(DwellTimeStats(
                camera_id=camera_id,
                zone=camera.config.zone,
                timestamp=datetime.utcnow(),
                hour=datetime.utcnow().hour,
                avg_dwell_time=visitor_stats["avg_dwell_time"],
                min_dwell_time=visitor_stats["min_dwell_time"],
                max_dwell_time=visitor_stats["max_dwell_time"],
                visitor_count=visitor_stats["unique_visitors"]
            ))
            
            gender = visitor_stats.get("gender_breakdown", {})
            session.add(GenderStat(
                camera_id=camera_id,
                timestamp=datetime.utcnow(),
                hour=datetime.utcnow().hour,
                male_count=gender.get("male", 0),
                female_count=gender.get("female", 0)
            ))
            
            await session.commit()
        
        # Generate heatmap
        heatmap_url, proc_time = camera.generate_heatmap()
        if heatmap_url:
            async with async_session() as session:
                session.add(Heatmap(
                    camera_id=camera_id,
                    timestamp=datetime.utcnow(),
                    s3_url=heatmap_url,
                    processing_time=proc_time
                ))
                await session.commit()
        
        # Mark complete
        async with async_session() as session:
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.processed_frames = camera.frame_count
                await session.commit()
        
        logger.info(f"[{camera_id}] Batch completed. Visitors: {visitor_stats['unique_visitors']}")
        
    except Exception as e:
        logger.error(f"[{camera_id}] Batch error: {e}")
        async with async_session() as session:
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = str(e)
                await session.commit()


@router.get("/cameras")
async def get_cameras() -> List[Dict]:
    """Get all cameras"""
    return camera_manager.get_camera_configs()


@router.get("/cameras/{camera_id}")
async def get_camera(camera_id: str) -> Dict:
    """Get camera details"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return {
        "id": camera.camera_id,
        "name": camera.config.name,
        "zone": camera.config.zone,
        "source": camera.config.source,
        "mode": "batch" if camera.config.is_batch_mode() else "realtime",
        "progress": camera.get_progress()
    }


@router.post("/batch/start/{camera_id}")
async def start_batch(camera_id: str, background_tasks: BackgroundTasks) -> Dict:
    """Start batch processing"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    if not camera.config.is_batch_mode():
        raise HTTPException(status_code=400, detail="Camera is in real-time mode")
    
    await clear_camera_data(camera_id)
    camera.reset()
    
    if not camera.connect():
        raise HTTPException(status_code=500, detail="Failed to connect to video source")
    
    async with async_session() as session:
        job = ProcessingJob(
            camera_id=camera_id,
            source=camera.config.source,
            mode="batch",
            total_frames=camera.total_frames
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = job.id
    
    task = asyncio.create_task(batch_processing_task(camera_id, job_id))
    batch_tasks[camera_id] = task
    
    return {
        "message": "Batch processing started",
        "job_id": job_id,
        "camera_id": camera_id,
        "total_frames": camera.total_frames
    }


@router.get("/batch/progress/{camera_id}")
async def get_progress(camera_id: str) -> Dict:
    """Get batch processing progress"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return camera.get_progress()


@router.post("/reset-count/{camera_id}")
async def reset_count(camera_id: str) -> Dict:
    """Reset count after batch processing"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    async with async_session() as session:
        recent = datetime.utcnow() - timedelta(seconds=30)
        await session.execute(
            delete(PeopleCount).where(
                PeopleCount.camera_id == camera_id,
                PeopleCount.timestamp >= recent
            )
        )
        await session.commit()
    
    return {"status": "reset", "camera_id": camera_id}