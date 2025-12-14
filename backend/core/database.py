from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .config import settings

Base = declarative_base()


# ============ Models ============

class Camera(Base):
    __tablename__ = "cameras"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    zone = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)
    mode = Column(String, default="auto")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PeopleCount(Base):
    __tablename__ = "people_counts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, default="default", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    count = Column(Integer, nullable=False)
    hour = Column(Integer, index=True)


class Heatmap(Base):
    __tablename__ = "heatmaps"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, default="default", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    s3_url = Column(String, nullable=False)
    processing_time = Column(Float)


class GenderStat(Base):
    __tablename__ = "gender_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, default="default", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    hour = Column(Integer, index=True)
    male_count = Column(Integer, default=0)
    female_count = Column(Integer, default=0)


class ZoneStat(Base):
    __tablename__ = "zone_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    zone = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    hour = Column(Integer, index=True)
    total_count = Column(Integer, default=0)
    avg_dwell_time = Column(Float, default=0)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    source = Column(String, nullable=False)
    status = Column(String, default="pending")
    mode = Column(String, default="batch")
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(String)


class UniqueVisitor(Base):
    __tablename__ = "unique_visitors"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    track_id = Column(Integer, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen = Column(DateTime, default=datetime.utcnow)
    gender = Column(String)
    dwell_time = Column(Float, default=0)
    is_active = Column(Boolean, default=True)


class DwellTimeStats(Base):
    __tablename__ = "dwell_time_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    zone = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    hour = Column(Integer, index=True)
    avg_dwell_time = Column(Float, default=0)
    min_dwell_time = Column(Float, default=0)
    max_dwell_time = Column(Float, default=0)
    visitor_count = Column(Integer, default=0)


# ============ Database Connections ============

# Async engine for FastAPI
async_engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for background tasks
sync_engine = create_engine(
    settings.database_url.replace("+aiosqlite", ""),
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=sync_engine)


async def init_async_db():
    """Initialize database tables (async)"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Dependency for FastAPI
async def get_db():
    async with async_session() as session:
        yield session