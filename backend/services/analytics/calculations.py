from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from core import (
    PeopleCount, GenderStat, ZoneStat, DwellTimeStats, Heatmap
)

logger = logging.getLogger(__name__)


class AnalyticsCalculator:
    """
    Performs analytics calculations from database
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_current_count(self, camera_id: str = None) -> Dict[str, Any]:
        """Get current people count"""
        recent_time = datetime.utcnow() - timedelta(seconds=30)
        
        stmt = select(PeopleCount).where(PeopleCount.timestamp >= recent_time)
        if camera_id:
            stmt = stmt.where(PeopleCount.camera_id == camera_id)
        stmt = stmt.order_by(PeopleCount.timestamp.desc()).limit(10)
        
        result = await self.session.execute(stmt)
        counts = result.scalars().all()
        
        if counts:
            total = sum(c.count for c in counts) // len(counts)
            by_camera = {c.camera_id: c.count for c in counts}
            return {
                "count": total,
                "timestamp": counts[0].timestamp.isoformat(),
                "by_camera": by_camera
            }
        
        return {"count": 0, "timestamp": datetime.utcnow().isoformat(), "by_camera": {}}
    
    async def get_peak_hours(self, camera_id: str = None, hours: int = 24) -> List[Dict]:
        """Get peak hours analysis"""
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        stmt = select(
            PeopleCount.hour,
            func.avg(PeopleCount.count).label('avg_count'),
            func.max(PeopleCount.count).label('max_count'),
            func.count(PeopleCount.id).label('sample_count')
        ).where(PeopleCount.timestamp >= time_threshold)
        
        if camera_id:
            stmt = stmt.where(PeopleCount.camera_id == camera_id)
        
        stmt = stmt.group_by(PeopleCount.hour).order_by(PeopleCount.hour)
        
        result = await self.session.execute(stmt)
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
        
        # Fill missing hours
        existing = {h["hour"] for h in peak_hours}
        for hour in range(24):
            if hour not in existing:
                peak_hours.append({"hour": hour, "avg_count": 0, "max_count": 0, "samples": 0})
        
        peak_hours.sort(key=lambda x: x["hour"])
        return peak_hours
    
    async def get_gender_stats(self, camera_id: str = None) -> Dict[str, Any]:
        """Get gender distribution statistics"""
        stmt = select(
            func.sum(GenderStat.male_count).label('total_male'),
            func.sum(GenderStat.female_count).label('total_female')
        )
        if camera_id:
            stmt = stmt.where(GenderStat.camera_id == camera_id)
        
        result = await self.session.execute(stmt)
        totals = result.one()
        
        total_male = totals.total_male or 0
        total_female = totals.total_female or 0
        total = total_male + total_female
        
        male_pct = round((total_male / total * 100), 1) if total > 0 else 0
        female_pct = round((total_female / total * 100), 1) if total > 0 else 0
        
        # Hourly breakdown
        time_threshold = datetime.utcnow() - timedelta(hours=24)
        hourly_stmt = select(
            GenderStat.hour,
            func.sum(GenderStat.male_count).label('male'),
            func.sum(GenderStat.female_count).label('female')
        ).where(GenderStat.timestamp >= time_threshold)
        
        if camera_id:
            hourly_stmt = hourly_stmt.where(GenderStat.camera_id == camera_id)
        
        hourly_stmt = hourly_stmt.group_by(GenderStat.hour).order_by(GenderStat.hour)
        
        hourly_result = await self.session.execute(hourly_stmt)
        hourly_data = [
            {"hour": row.hour, "male": row.male or 0, "female": row.female or 0}
            for row in hourly_result.all()
        ]
        
        # Fill missing hours
        existing = {h["hour"] for h in hourly_data}
        for hour in range(24):
            if hour not in existing:
                hourly_data.append({"hour": hour, "male": 0, "female": 0})
        hourly_data.sort(key=lambda x: x["hour"])
        
        return {
            "total": {"male": total_male, "female": total_female, "total_detected": total},
            "percentage": {"male": male_pct, "female": female_pct},
            "by_hour": hourly_data
        }
    
    async def get_zone_stats(self) -> Dict[str, Any]:
        """Get zone-based statistics"""
        time_threshold = datetime.utcnow() - timedelta(hours=24)
        
        stmt = select(
            ZoneStat.camera_id,
            ZoneStat.zone,
            func.sum(ZoneStat.total_count).label('total_visitors'),
            func.avg(ZoneStat.total_count).label('avg_count'),
            func.avg(ZoneStat.avg_dwell_time).label('avg_dwell')
        ).where(ZoneStat.timestamp >= time_threshold).group_by(ZoneStat.camera_id, ZoneStat.zone)
        
        result = await self.session.execute(stmt)
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
        
        return {"zones": zones, "total_visitors": total_all}
    
    async def get_dwell_time_stats(self, camera_id: str = None) -> Dict[str, Any]:
        """Get dwell time statistics"""
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
        
        result = await self.session.execute(stmt)
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
    
    async def get_latest_heatmap(self, camera_id: str = None) -> Optional[Dict]:
        """Get latest heatmap URL"""
        stmt = select(Heatmap)
        if camera_id:
            stmt = stmt.where(Heatmap.camera_id == camera_id)
        stmt = stmt.order_by(Heatmap.timestamp.desc()).limit(1)
        
        result = await self.session.execute(stmt)
        heatmap = result.scalar_one_or_none()
        
        if heatmap:
            return {
                "url": heatmap.s3_url,
                "camera_id": heatmap.camera_id,
                "timestamp": heatmap.timestamp.isoformat(),
                "processing_time": heatmap.processing_time
            }
        return None