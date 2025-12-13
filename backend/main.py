from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from sqlalchemy import func, select, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional

from config import settings
from models import Base, PeopleCount, Heatmap, GenderStat, ZoneStat, Camera, ProcessingJob, UniqueVisitor, DwellTimeStats, init_db
from analytics import camera_manager, video_analytics

# Logging setup
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Async database engine
engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Track batch processing tasks
batch_tasks: Dict[str, asyncio.Task] = {}


async def clear_camera_data(camera_id: str):
    """Clear all data for a specific camera before batch processing"""
    async with async_session() as session:
        await session.execute(delete(PeopleCount).where(PeopleCount.camera_id == camera_id))
        await session.execute(delete(GenderStat).where(GenderStat.camera_id == camera_id))
        await session.execute(delete(ZoneStat).where(ZoneStat.camera_id == camera_id))
        await session.execute(delete(DwellTimeStats).where(DwellTimeStats.camera_id == camera_id))
        await session.execute(delete(Heatmap).where(Heatmap.camera_id == camera_id))
        await session.execute(delete(ProcessingJob).where(ProcessingJob.camera_id == camera_id))
        await session.commit()
        logger.info(f"[{camera_id}] Cleared all previous data")


async def realtime_processing_task(camera_id: str):
    """Background task for real-time camera processing"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        logger.error(f"Camera {camera_id} not found")
        return
    
    logger.info(f"[{camera_id}] Starting real-time processing")
    
    while True:
        try:
            count, frame, stats = camera.process_frame()
            current_time = datetime.utcnow()
            
            async with async_session() as session:
                people_count = PeopleCount(
                    camera_id=camera_id,
                    timestamp=current_time,
                    count=count,
                    hour=current_time.hour
                )
                session.add(people_count)
                
                zone_stat = ZoneStat(
                    camera_id=camera_id,
                    zone=camera.config.zone,
                    timestamp=current_time,
                    hour=current_time.hour,
                    total_count=stats.get("unique_visitors", count),
                    avg_dwell_time=stats.get("avg_dwell_time", 0)
                )
                session.add(zone_stat)
                
                await session.commit()
            
            logger.debug(f"[{camera_id}] Count: {count}, Unique: {stats.get('unique_visitors', 0)}, Dwell: {stats.get('avg_dwell_time', 0):.1f}s")
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"[{camera_id}] Real-time processing error: {e}")
            await asyncio.sleep(10)


async def batch_processing_task(camera_id: str, job_id: int):
    """Background task for batch video processing"""
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        logger.error(f"Camera {camera_id} not found")
        return
    
    logger.info(f"[{camera_id}] Starting batch processing (Job #{job_id})")
    
    async with async_session() as session:
        job = await session.get(ProcessingJob, job_id)
        if job:
            job.status = "processing"
            job.started_at = datetime.utcnow()
            await session.commit()
    
    processed_count = 0
    
    try:
        while True:
            count, frame, stats = camera.process_frame()
            
            if stats.get("skipped"):
                continue
            
            if count == -1:
                logger.info(f"[{camera_id}] Batch processing completed")
                break
            
            current_time = datetime.utcnow()
            processed_count += 1
            
            # Save people count and zone stats every 10 frames (NOT gender - that's saved at end only)
            if processed_count % 10 == 0:
                async with async_session() as session:
                    people_count = PeopleCount(
                        camera_id=camera_id,
                        timestamp=current_time,
                        count=count,
                        hour=current_time.hour
                    )
                    session.add(people_count)
                    
                    zone_stat = ZoneStat(
                        camera_id=camera_id,
                        zone=camera.config.zone,
                        timestamp=current_time,
                        hour=current_time.hour,
                        total_count=stats.get("unique_visitors", count),
                        avg_dwell_time=stats.get("avg_dwell_time", 0)
                    )
                    session.add(zone_stat)
                    
                    job = await session.get(ProcessingJob, job_id)
                    if job:
                        job.processed_frames = camera.frame_count
                    
                    await session.commit()
                
                progress = camera.get_progress()
                logger.info(f"[{camera_id}] Progress: {progress['progress_percent']}%, Unique visitors: {stats.get('unique_visitors', 0)}")
            
            await asyncio.sleep(0.01)
        
        # Get final visitor stats
        visitor_stats = camera.get_visitor_stats()
        
        # Save final stats at end of batch
        async with async_session() as session:
            dwell_stats = DwellTimeStats(
                camera_id=camera_id,
                zone=camera.config.zone,
                timestamp=datetime.utcnow(),
                hour=datetime.utcnow().hour,
                avg_dwell_time=visitor_stats["avg_dwell_time"],
                min_dwell_time=visitor_stats["min_dwell_time"],
                max_dwell_time=visitor_stats["max_dwell_time"],
                visitor_count=visitor_stats["unique_visitors"]
            )
            session.add(dwell_stats)
            
            # Save final gender stats (unique counts only)
            gender_breakdown = visitor_stats.get("gender_breakdown", {})
            final_gender = GenderStat(
                camera_id=camera_id,
                timestamp=datetime.utcnow(),
                hour=datetime.utcnow().hour,
                male_count=gender_breakdown.get("male", 0),
                female_count=gender_breakdown.get("female", 0)
            )
            session.add(final_gender)
            
            await session.commit()
        
        # Generate final heatmap
        heatmap_url, processing_time = camera.generate_heatmap_image()
        if heatmap_url:
            async with async_session() as session:
                heatmap = Heatmap(
                    camera_id=camera_id,
                    timestamp=datetime.utcnow(),
                    s3_url=heatmap_url,
                    processing_time=processing_time
                )
                session.add(heatmap)
                await session.commit()
        
        # Update job as completed
        async with async_session() as session:
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.processed_frames = camera.frame_count
                await session.commit()
        
        logger.info(f"[{camera_id}] Batch job #{job_id} completed. Unique visitors: {visitor_stats['unique_visitors']}, Avg dwell: {visitor_stats['avg_dwell_time']}s")
        
    except Exception as e:
        logger.error(f"[{camera_id}] Batch processing error: {e}")
        async with async_session() as session:
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = str(e)
                await session.commit()


async def heatmap_generation_task():
    """Background task to periodically generate heatmaps for all cameras"""
    await asyncio.sleep(60)
    
    while True:
        try:
            for camera in camera_manager.get_all_cameras():
                if not camera.config.is_batch_mode():
                    heatmap_url, processing_time = camera.generate_heatmap_image()
                    
                    if heatmap_url:
                        async with async_session() as session:
                            heatmap = Heatmap(
                                camera_id=camera.camera_id,
                                timestamp=datetime.utcnow(),
                                s3_url=heatmap_url,
                                processing_time=processing_time
                            )
                            session.add(heatmap)
                            await session.commit()
            
            await asyncio.sleep(settings.heatmap_interval)
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            await asyncio.sleep(settings.heatmap_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting Retail Analytics API...")
    init_db()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    os.makedirs("heatmaps", exist_ok=True)
    
    camera_manager.initialize()
    camera_manager.load_cameras_from_config()
    
    for camera in camera_manager.get_all_cameras():
        if not camera.config.is_batch_mode():
            asyncio.create_task(realtime_processing_task(camera.camera_id))
    
    asyncio.create_task(heatmap_generation_task())
    
    yield
    
    logger.info("Shutting down...")
    camera_manager.cleanup()


app = FastAPI(
    title="Retail Video Analytics API",
    version="2.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Retail Video Analytics API",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "cameras": len(camera_manager.cameras)
    }


@app.get("/api/cameras")
async def get_cameras() -> List[Dict]:
    return camera_manager.get_camera_configs()


@app.get("/api/cameras/{camera_id}")
async def get_camera(camera_id: str) -> Dict:
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


@app.post("/api/batch/start/{camera_id}")
async def start_batch_processing(camera_id: str, background_tasks: BackgroundTasks) -> Dict:
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


@app.get("/api/batch/progress/{camera_id}")
async def get_batch_progress(camera_id: str) -> Dict:
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return camera.get_progress()


@app.get("/api/batch/jobs")
async def get_processing_jobs(
    camera_id: Optional[str] = None,
    status: Optional[str] = None
) -> List[Dict]:
    async with async_session() as session:
        stmt = select(ProcessingJob)
        if camera_id:
            stmt = stmt.where(ProcessingJob.camera_id == camera_id)
        if status:
            stmt = stmt.where(ProcessingJob.status == status)
        stmt = stmt.order_by(ProcessingJob.id.desc()).limit(20)
        
        result = await session.execute(stmt)
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


@app.get("/api/count")
async def get_current_count(camera_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        async with async_session() as session:
            recent_time = datetime.utcnow() - timedelta(seconds=30)
            stmt = select(PeopleCount).where(PeopleCount.timestamp >= recent_time)
            
            if camera_id:
                stmt = stmt.where(PeopleCount.camera_id == camera_id)
            
            stmt = stmt.order_by(PeopleCount.timestamp.desc()).limit(10)
            result = await session.execute(stmt)
            counts = result.scalars().all()
            
            if counts:
                total = sum(c.count for c in counts) // len(counts)
                by_camera = {}
                for c in counts:
                    if c.camera_id not in by_camera:
                        by_camera[c.camera_id] = c.count
                
                return {
                    "count": total,
                    "timestamp": counts[0].timestamp.isoformat(),
                    "by_camera": by_camera
                }
            else:
                return {
                    "count": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "by_camera": {}
                }
    
    except Exception as e:
        logger.error(f"Error getting current count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/peak-hours")
async def get_peak_hours(camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        async with async_session() as session:
            time_threshold = datetime.utcnow() - timedelta(hours=24)
            
            stmt = select(
                PeopleCount.hour,
                func.avg(PeopleCount.count).label('avg_count'),
                func.max(PeopleCount.count).label('max_count'),
                func.count(PeopleCount.id).label('sample_count')
            ).where(PeopleCount.timestamp >= time_threshold)
            
            if camera_id:
                stmt = stmt.where(PeopleCount.camera_id == camera_id)
            
            stmt = stmt.group_by(PeopleCount.hour).order_by(PeopleCount.hour)
            
            result = await session.execute(stmt)
            rows = result.all()
            
            peak_hours = [
                {
                    "hour": row.hour,
                    "avg_count": int(round(row.avg_count)),
                    "max_count": row.max_count,
                    "samples": row.sample_count
                }
                for row in rows
            ]
            
            existing_hours = {item["hour"] for item in peak_hours}
            for hour in range(24):
                if hour not in existing_hours:
                    peak_hours.append({"hour": hour, "avg_count": 0, "max_count": 0, "samples": 0})
            
            peak_hours.sort(key=lambda x: x["hour"])
            return peak_hours
    
    except Exception as e:
        logger.error(f"Error getting peak hours: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gender-stats")
async def get_gender_stats(camera_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        async with async_session() as session:
            total_stmt = select(
                func.sum(GenderStat.male_count).label('total_male'),
                func.sum(GenderStat.female_count).label('total_female')
            )
            if camera_id:
                total_stmt = total_stmt.where(GenderStat.camera_id == camera_id)
            
            result = await session.execute(total_stmt)
            totals = result.one()
            
            total_male = totals.total_male or 0
            total_female = totals.total_female or 0
            total_detected = total_male + total_female
            
            male_pct = round((total_male / total_detected * 100), 1) if total_detected > 0 else 0
            female_pct = round((total_female / total_detected * 100), 1) if total_detected > 0 else 0
            
            time_threshold = datetime.utcnow() - timedelta(hours=24)
            hourly_stmt = select(
                GenderStat.hour,
                func.sum(GenderStat.male_count).label('male'),
                func.sum(GenderStat.female_count).label('female')
            ).where(GenderStat.timestamp >= time_threshold)
            
            if camera_id:
                hourly_stmt = hourly_stmt.where(GenderStat.camera_id == camera_id)
            
            hourly_stmt = hourly_stmt.group_by(GenderStat.hour).order_by(GenderStat.hour)
            
            hourly_result = await session.execute(hourly_stmt)
            hourly_data = [
                {"hour": row.hour, "male": row.male or 0, "female": row.female or 0}
                for row in hourly_result.all()
            ]
            
            existing_hours = {item["hour"] for item in hourly_data}
            for hour in range(24):
                if hour not in existing_hours:
                    hourly_data.append({"hour": hour, "male": 0, "female": 0})
            hourly_data.sort(key=lambda x: x["hour"])
            
            return {
                "total": {"male": total_male, "female": total_female, "total_detected": total_detected},
                "percentage": {"male": male_pct, "female": female_pct},
                "by_hour": hourly_data
            }
    
    except Exception as e:
        logger.error(f"Error getting gender stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/zone-stats")
async def get_zone_stats() -> Dict[str, Any]:
    try:
        async with async_session() as session:
            time_threshold = datetime.utcnow() - timedelta(hours=24)
            
            stmt = select(
                ZoneStat.camera_id,
                ZoneStat.zone,
                func.sum(ZoneStat.total_count).label('total_visitors'),
                func.avg(ZoneStat.total_count).label('avg_count'),
                func.avg(ZoneStat.avg_dwell_time).label('avg_dwell')
            ).where(
                ZoneStat.timestamp >= time_threshold
            ).group_by(ZoneStat.camera_id, ZoneStat.zone)
            
            result = await session.execute(stmt)
            rows = result.all()
            
            zones = [
                {
                    "camera_id": row.camera_id,
                    "zone": row.zone,
                    "total_visitors": int(row.total_visitors or 0),
                    "avg_count": int(round(row.avg_count)) if row.avg_count else 0,
                    "avg_dwell_time": round(row.avg_dwell or 0, 1)
                }
                for row in rows
            ]
            
            total_all = sum(z["total_visitors"] for z in zones)
            for zone in zones:
                zone["percentage"] = round((zone["total_visitors"] / total_all * 100), 1) if total_all > 0 else 0
            
            return {
                "zones": zones,
                "total_visitors": total_all
            }
    
    except Exception as e:
        logger.error(f"Error getting zone stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visitor-stats")
async def get_visitor_stats(camera_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        camera = camera_manager.get_camera(camera_id) if camera_id else None
        
        if camera:
            return camera.get_visitor_stats()
        
        all_stats = {
            "unique_visitors": 0,
            "active_visitors": 0,
            "completed_visitors": 0,
            "avg_dwell_time": 0,
            "gender_breakdown": {"male": 0, "female": 0, "unknown": 0},
            "by_camera": {}
        }
        
        total_dwell = 0
        total_visitors = 0
        
        for cam in camera_manager.get_all_cameras():
            cam_stats = cam.get_visitor_stats()
            all_stats["by_camera"][cam.camera_id] = cam_stats
            all_stats["unique_visitors"] += cam_stats["unique_visitors"]
            all_stats["active_visitors"] += cam_stats["active_visitors"]
            all_stats["completed_visitors"] += cam_stats.get("completed_visitors", 0)
            
            if cam_stats["unique_visitors"] > 0:
                total_dwell += cam_stats["avg_dwell_time"] * cam_stats["unique_visitors"]
                total_visitors += cam_stats["unique_visitors"]
            
            for gender in ["male", "female", "unknown"]:
                all_stats["gender_breakdown"][gender] += cam_stats["gender_breakdown"].get(gender, 0)
        
        if total_visitors > 0:
            all_stats["avg_dwell_time"] = round(total_dwell / total_visitors, 1)
        
        return all_stats
    
    except Exception as e:
        logger.error(f"Error getting visitor stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dwell-time")
async def get_dwell_time_stats(camera_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        async with async_session() as session:
            time_threshold = datetime.utcnow() - timedelta(hours=24)
            
            stmt = select(
                DwellTimeStats.camera_id,
                DwellTimeStats.zone,
                func.avg(DwellTimeStats.avg_dwell_time).label('avg_dwell'),
                func.min(DwellTimeStats.min_dwell_time).label('min_dwell'),
                func.max(DwellTimeStats.max_dwell_time).label('max_dwell'),
                func.sum(DwellTimeStats.visitor_count).label('total_visitors')
            ).where(DwellTimeStats.timestamp >= time_threshold)
            
            if camera_id:
                stmt = stmt.where(DwellTimeStats.camera_id == camera_id)
            
            stmt = stmt.group_by(DwellTimeStats.camera_id, DwellTimeStats.zone)
            
            result = await session.execute(stmt)
            rows = result.all()
            
            stats = [
                {
                    "camera_id": row.camera_id,
                    "zone": row.zone,
                    "avg_dwell_time": round(row.avg_dwell or 0, 1),
                    "min_dwell_time": round(row.min_dwell or 0, 1),
                    "max_dwell_time": round(row.max_dwell or 0, 1),
                    "total_visitors": int(row.total_visitors or 0)
                }
                for row in rows
            ]
            
            total_visitors = sum(s["total_visitors"] for s in stats)
            overall_avg = 0
            if total_visitors > 0:
                overall_avg = sum(s["avg_dwell_time"] * s["total_visitors"] for s in stats) / total_visitors
            
            return {
                "by_zone": stats,
                "overall_avg_dwell_time": round(overall_avg, 1),
                "total_visitors": total_visitors
            }
    
    except Exception as e:
        logger.error(f"Error getting dwell time stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/heatmap")
async def get_latest_heatmap(camera_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        async with async_session() as session:
            stmt = select(Heatmap)
            if camera_id:
                stmt = stmt.where(Heatmap.camera_id == camera_id)
            stmt = stmt.order_by(Heatmap.timestamp.desc()).limit(1)
            
            result = await session.execute(stmt)
            latest = result.scalar_one_or_none()
            
            if not latest:
                raise HTTPException(status_code=404, detail="No heatmap available yet")
            
            return {
                "url": latest.s3_url,
                "camera_id": latest.camera_id,
                "timestamp": latest.timestamp.isoformat(),
                "processing_time": latest.processing_time
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics() -> Dict[str, Any]:
    try:
        async with async_session() as session:
            total_counts = (await session.execute(select(func.count(PeopleCount.id)))).scalar()
            total_heatmaps = (await session.execute(select(func.count(Heatmap.id)))).scalar()
            total_cameras = len(camera_manager.cameras)
            
            active_cameras = [
                cam.camera_id for cam in camera_manager.get_all_cameras()
                if cam.cap and cam.cap.isOpened()
            ]
            
            return {
                "total_counts": total_counts,
                "total_heatmaps": total_heatmaps,
                "total_cameras": total_cameras,
                "active_cameras": active_cameras,
                "database": "healthy",
                "camera_status": "connected" if active_cameras else "disconnected"
            }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/summary")
async def get_report_summary(camera_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        async with async_session() as session:
            time_threshold = datetime.utcnow() - timedelta(hours=24)
            
            count_filter = PeopleCount.timestamp >= time_threshold
            gender_filter = GenderStat.timestamp >= time_threshold
            zone_filter = ZoneStat.timestamp >= time_threshold
            dwell_filter = DwellTimeStats.timestamp >= time_threshold
            
            if camera_id:
                count_filter = count_filter & (PeopleCount.camera_id == camera_id)
                gender_filter = gender_filter & (GenderStat.camera_id == camera_id)
                zone_filter = zone_filter & (ZoneStat.camera_id == camera_id)
                dwell_filter = dwell_filter & (DwellTimeStats.camera_id == camera_id)
            
            visitor_stmt = select(
                func.sum(ZoneStat.total_count).label('total'),
                func.avg(ZoneStat.total_count).label('avg'),
                func.max(ZoneStat.total_count).label('max')
            ).where(zone_filter)
            visitor_result = await session.execute(visitor_stmt)
            visitor_data = visitor_result.one()
            
            gender_stmt = select(
                func.sum(GenderStat.male_count).label('male'),
                func.sum(GenderStat.female_count).label('female')
            ).where(gender_filter)
            gender_result = await session.execute(gender_stmt)
            gender_data = gender_result.one()
            
            dwell_stmt = select(
                func.avg(DwellTimeStats.avg_dwell_time).label('avg_dwell'),
                func.min(DwellTimeStats.min_dwell_time).label('min_dwell'),
                func.max(DwellTimeStats.max_dwell_time).label('max_dwell'),
                func.sum(DwellTimeStats.visitor_count).label('total_visitors')
            ).where(dwell_filter)
            dwell_result = await session.execute(dwell_stmt)
            dwell_data = dwell_result.one()
            
            peak_stmt = select(
                PeopleCount.hour,
                func.avg(PeopleCount.count).label('avg_count')
            ).where(count_filter).group_by(PeopleCount.hour).order_by(func.avg(PeopleCount.count).desc()).limit(5)
            peak_result = await session.execute(peak_stmt)
            peak_hours = [{"hour": r.hour, "avg_count": int(round(r.avg_count))} for r in peak_result.all()]
            
            heatmap_stmt = select(Heatmap).order_by(Heatmap.timestamp.desc()).limit(1)
            if camera_id:
                heatmap_stmt = heatmap_stmt.where(Heatmap.camera_id == camera_id)
            heatmap_result = await session.execute(heatmap_stmt)
            latest_heatmap = heatmap_result.scalar_one_or_none()
            
            total_male = gender_data.male or 0
            total_female = gender_data.female or 0
            total_gender = total_male + total_female
            
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "period": "Last 24 hours",
                "camera_id": camera_id or "All Cameras",
                "visitors": {
                    "total": int(visitor_data.total or 0),
                    "average_per_hour": int(round(visitor_data.avg or 0)),
                    "peak": int(visitor_data.max or 0),
                    "unique_tracked": int(dwell_data.total_visitors or 0)
                },
                "gender": {
                    "male": int(total_male),
                    "female": int(total_female),
                    "male_percentage": round(total_male / total_gender * 100, 1) if total_gender > 0 else 0,
                    "female_percentage": round(total_female / total_gender * 100, 1) if total_gender > 0 else 0
                },
                "dwell_time": {
                    "average_seconds": round(dwell_data.avg_dwell or 0, 1),
                    "min_seconds": round(dwell_data.min_dwell or 0, 1),
                    "max_seconds": round(dwell_data.max_dwell or 0, 1),
                    "average_formatted": f"{int((dwell_data.avg_dwell or 0) // 60)}m {int((dwell_data.avg_dwell or 0) % 60)}s"
                },
                "peak_hours": peak_hours,
                "heatmap_url": latest_heatmap.s3_url if latest_heatmap else None
            }
    
    except Exception as e:
        logger.error(f"Error generating report summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)