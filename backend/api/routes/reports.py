from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from core import get_db, PeopleCount, Heatmap
from services.reports import ReportGenerator
from services.video import camera_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["reports"])


@router.get("/report/summary")
async def get_summary(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Generate summary report"""
    generator = ReportGenerator(db)
    return await generator.generate_summary(camera_id)


@router.get("/report/daily")
async def get_daily_report(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Generate daily report"""
    generator = ReportGenerator(db)
    return await generator.generate_daily_report(camera_id)


@router.get("/report/weekly")
async def get_weekly_report(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Generate weekly report"""
    generator = ReportGenerator(db)
    return await generator.generate_weekly_report(camera_id)


@router.get("/stats")
async def get_system_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get system statistics"""
    total_counts = (await db.execute(select(func.count(PeopleCount.id)))).scalar()
    total_heatmaps = (await db.execute(select(func.count(Heatmap.id)))).scalar()
    
    active_cameras = [
        cam.camera_id for cam in camera_manager.get_all_cameras()
        if cam.cap and cam.cap.isOpened()
    ]
    
    return {
        "total_counts": total_counts,
        "total_heatmaps": total_heatmaps,
        "total_cameras": len(camera_manager.cameras),
        "active_cameras": active_cameras,
        "database": "healthy",
        "camera_status": "connected" if active_cameras else "disconnected"
    }


@router.get("/batch/jobs")
async def get_jobs(
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> List[Dict]:
    """Get processing jobs"""
    from core import ProcessingJob
    from sqlalchemy import select
    
    stmt = select(ProcessingJob)
    if camera_id:
        stmt = stmt.where(ProcessingJob.camera_id == camera_id)
    if status:
        stmt = stmt.where(ProcessingJob.status == status)
    stmt = stmt.order_by(ProcessingJob.id.desc()).limit(20)
    
    result = await db.execute(stmt)
    jobs = result.scalars().all()
    
    return [
        {
            "id": job.id,
            "camera_id": job.camera_id,
            "source": job.source,
            "status": job.status,
            "progress": round((job.processed_frames / job.total_frames * 100), 1) if job.total_frames > 0 else 0,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
        for job in jobs
    ]