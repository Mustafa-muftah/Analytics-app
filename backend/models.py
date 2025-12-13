from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import settings

Base = declarative_base()


class Camera(Base):
    """Camera/Zone configuration"""
    __tablename__ = "cameras"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    zone = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)
    mode = Column(String, default="auto")  # realtime, batch, auto
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PeopleCount(Base):
    __tablename__ = "people_counts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, default="default", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    count = Column(Integer, nullable=False)
    hour = Column(Integer, index=True)  # 0-23 for aggregation
    

class Heatmap(Base):
    __tablename__ = "heatmaps"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, default="default", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    s3_url = Column(String, nullable=False)
    processing_time = Column(Float)  # seconds


class GenderStat(Base):
    __tablename__ = "gender_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, default="default", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    hour = Column(Integer, index=True)  # 0-23 for aggregation
    male_count = Column(Integer, default=0)
    female_count = Column(Integer, default=0)


class ZoneStat(Base):
    """Aggregated zone statistics"""
    __tablename__ = "zone_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    zone = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    hour = Column(Integer, index=True)
    total_count = Column(Integer, default=0)
    avg_dwell_time = Column(Float, default=0)  # seconds


class ProcessingJob(Base):
    """Track batch processing jobs"""
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    source = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    mode = Column(String, default="batch")
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(String)


class UniqueVisitor(Base):
    """Track unique visitors with their track ID"""
    __tablename__ = "unique_visitors"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    track_id = Column(Integer, index=True)  # DeepSORT track ID
    first_seen = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen = Column(DateTime, default=datetime.utcnow)
    gender = Column(String)  # male, female, unknown
    dwell_time = Column(Float, default=0)  # seconds
    is_active = Column(Boolean, default=True)  # currently in frame


class DwellTimeStats(Base):
    """Aggregated dwell time statistics"""
    __tablename__ = "dwell_time_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    zone = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    hour = Column(Integer, index=True)
    avg_dwell_time = Column(Float, default=0)  # seconds
    min_dwell_time = Column(Float, default=0)
    max_dwell_time = Column(Float, default=0)
    visitor_count = Column(Integer, default=0)


# Database setup
engine = create_engine(
    settings.database_url.replace("+aiosqlite", ""),
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
