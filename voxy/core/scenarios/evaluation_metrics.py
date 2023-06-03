from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeMeta, declarative_base
from sqlalchemy.sql import func

# declarative base class
Base: DeclarativeMeta = declarative_base()


class EvaluationData(Base):
    __tablename__ = "evaluation_data"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    cache_key = Column(String)
    run_uuid = Column(String)
    monitor = Column(String)
    kind = Column(String)
    data = Column(JSON)


class EvaluationMetrics(Base):
    __tablename__ = "evaluation_metrics"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    cache_key = Column(String)
    run_uuid = Column(String)
    monitor = Column(String)
    precision = Column(Float)
    recall = Column(Float)
