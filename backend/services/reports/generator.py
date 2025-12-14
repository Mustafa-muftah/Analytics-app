from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from core import (
    PeopleCount, GenderStat, ZoneStat, DwellTimeStats, Heatmap
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates analytics reports
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def generate_summary(self, camera_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        # Build filters
        count_filter = PeopleCount.timestamp >= time_threshold
        gender_filter = GenderStat.timestamp >= time_threshold
        zone_filter = ZoneStat.timestamp >= time_threshold
        dwell_filter = DwellTimeStats.timestamp >= time_threshold
        
        if camera_id:
            count_filter = count_filter & (PeopleCount.camera_id == camera_id)
            gender_filter = gender_filter & (GenderStat.camera_id == camera_id)
            zone_filter = zone_filter & (ZoneStat.camera_id == camera_id)
            dwell_filter = dwell_filter & (DwellTimeStats.camera_id == camera_id)
        
        # Visitor stats
        visitor_stmt = select(
            func.sum(ZoneStat.total_count).label('total'),
            func.avg(ZoneStat.total_count).label('avg'),
            func.max(ZoneStat.total_count).label('max')
        ).where(zone_filter)
        visitor_result = await self.session.execute(visitor_stmt)
        visitor_data = visitor_result.one()
        
        # Gender stats
        gender_stmt = select(
            func.sum(GenderStat.male_count).label('male'),
            func.sum(GenderStat.female_count).label('female')
        ).where(gender_filter)
        gender_result = await self.session.execute(gender_stmt)
        gender_data = gender_result.one()
        
        # Dwell time stats
        dwell_stmt = select(
            func.avg(DwellTimeStats.avg_dwell_time).label('avg_dwell'),
            func.min(DwellTimeStats.min_dwell_time).label('min_dwell'),
            func.max(DwellTimeStats.max_dwell_time).label('max_dwell'),
            func.sum(DwellTimeStats.visitor_count).label('total_visitors')
        ).where(dwell_filter)
        dwell_result = await self.session.execute(dwell_stmt)
        dwell_data = dwell_result.one()
        
        # Peak hours
        peak_stmt = select(
            PeopleCount.hour,
            func.avg(PeopleCount.count).label('avg_count')
        ).where(count_filter).group_by(PeopleCount.hour).order_by(
            func.avg(PeopleCount.count).desc()
        ).limit(5)
        peak_result = await self.session.execute(peak_stmt)
        peak_hours = [
            {"hour": r.hour, "avg_count": int(round(r.avg_count))} 
            for r in peak_result.all()
        ]
        
        # Latest heatmap
        heatmap_stmt = select(Heatmap).order_by(Heatmap.timestamp.desc()).limit(1)
        if camera_id:
            heatmap_stmt = heatmap_stmt.where(Heatmap.camera_id == camera_id)
        heatmap_result = await self.session.execute(heatmap_stmt)
        latest_heatmap = heatmap_result.scalar_one_or_none()
        
        # Calculate percentages
        total_male = gender_data.male or 0
        total_female = gender_data.female or 0
        total_gender = total_male + total_female
        
        avg_dwell = dwell_data.avg_dwell or 0
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "period": f"Last {hours} hours",
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
                "average_seconds": round(avg_dwell, 1),
                "min_seconds": round(dwell_data.min_dwell or 0, 1),
                "max_seconds": round(dwell_data.max_dwell or 0, 1),
                "average_formatted": f"{int(avg_dwell // 60)}m {int(avg_dwell % 60)}s"
            },
            "peak_hours": peak_hours,
            "heatmap_url": latest_heatmap.s3_url if latest_heatmap else None
        }
    
    async def generate_daily_report(self, camera_id: str = None) -> Dict[str, Any]:
        """Generate daily report (last 24 hours)"""
        return await self.generate_summary(camera_id, hours=24)
    
    async def generate_weekly_report(self, camera_id: str = None) -> Dict[str, Any]:
        """Generate weekly report (last 7 days)"""
        return await self.generate_summary(camera_id, hours=168)