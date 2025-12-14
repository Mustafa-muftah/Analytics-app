from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
import logging

from core import get_db
from services.analytics import AnalyticsCalculator
from services.video import camera_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/count")
async def get_count(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get current people count"""
    if camera_id:
        camera = camera_manager.get_camera(camera_id)
        if camera and camera.batch_completed:
            return {
                "count": 0,
                "timestamp": "",
                "by_camera": {camera_id: 0},
                "batch_completed": True
            }
    
    calc = AnalyticsCalculator(db)
    return await calc.get_current_count(camera_id)


@router.get("/peak-hours")
async def get_peak_hours(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> List[Dict]:
    """Get peak hours analysis"""
    calc = AnalyticsCalculator(db)
    return await calc.get_peak_hours(camera_id)


@router.get("/gender-stats")
async def get_gender_stats(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get gender distribution"""
    calc = AnalyticsCalculator(db)
    return await calc.get_gender_stats(camera_id)


@router.get("/zone-stats")
async def get_zone_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get zone statistics"""
    calc = AnalyticsCalculator(db)
    return await calc.get_zone_stats()


@router.get("/dwell-time")
async def get_dwell_time(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get dwell time statistics"""
    calc = AnalyticsCalculator(db)
    return await calc.get_dwell_time_stats(camera_id)


@router.get("/heatmap")
async def get_heatmap(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get latest heatmap"""
    calc = AnalyticsCalculator(db)
    result = await calc.get_latest_heatmap(camera_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="No heatmap available")
    
    return result


@router.get("/visitor-stats")
async def get_visitor_stats(camera_id: Optional[str] = None) -> Dict[str, Any]:
    """Get visitor statistics from live tracking"""
    if camera_id:
        camera = camera_manager.get_camera(camera_id)
        if camera:
            return camera.processor.get_visitor_stats()
    
    # Aggregate all cameras
    all_stats = {
        "unique_visitors": 0,
        "active_visitors": 0,
        "avg_dwell_time": 0,
        "gender_breakdown": {"male": 0, "female": 0, "unknown": 0},
        "by_camera": {}
    }
    
    total_dwell = 0
    total_visitors = 0
    
    for cam in camera_manager.get_all_cameras():
        stats = cam.processor.get_visitor_stats()
        all_stats["by_camera"][cam.camera_id] = stats
        all_stats["unique_visitors"] += stats["unique_visitors"]
        all_stats["active_visitors"] += stats["active_visitors"]
        
        if stats["unique_visitors"] > 0:
            total_dwell += stats["avg_dwell_time"] * stats["unique_visitors"]
            total_visitors += stats["unique_visitors"]
        
        for gender in ["male", "female", "unknown"]:
            all_stats["gender_breakdown"][gender] += stats["gender_breakdown"].get(gender, 0)
    
    if total_visitors > 0:
        all_stats["avg_dwell_time"] = round(total_dwell / total_visitors, 1)
    
    return all_stats