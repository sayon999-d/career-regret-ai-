from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean, JSON, create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    preferences = Column(JSON, default={})

    decisions = relationship("CareerDecision", back_populates="user")
    analyses = relationship("RegretAnalysis", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")

class CareerDecision(Base):
    __tablename__ = "career_decisions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    decision_type = Column(String(50), index=True)
    description = Column(Text)
    alternatives = Column(JSON, default=[])
    context_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="pending")

    user = relationship("User", back_populates="decisions")
    analyses = relationship("RegretAnalysis", back_populates="decision")

class RegretAnalysis(Base):
    __tablename__ = "regret_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    decision_id = Column(Integer, ForeignKey("career_decisions.id"))
    predicted_regret = Column(Float)
    confidence = Column(Float)
    risk_level = Column(String(20))
    top_factors = Column(JSON, default=[])
    recommendations = Column(JSON, default=[])
    graph_analysis = Column(JSON, default={})
    llm_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50))

    user = relationship("User", back_populates="analyses")
    decision = relationship("CareerDecision", back_populates="analyses")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    analysis_id = Column(Integer, ForeignKey("regret_analyses.id"), nullable=True)
    feedback_type = Column(String(50))
    rating = Column(Integer, nullable=True)
    comment = Column(Text, nullable=True)
    correction_data = Column(JSON, nullable=True)
    actual_outcome = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)

    user = relationship("User", back_populates="feedback")

class DecisionNode(Base):
    __tablename__ = "decision_nodes"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(100), unique=True, index=True)
    node_type = Column(String(50))
    label = Column(String(255))
    weight = Column(Float, default=1.0)
    attributes = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DecisionEdge(Base):
    __tablename__ = "decision_edges"

    id = Column(Integer, primary_key=True, index=True)
    source_node_id = Column(String(100), index=True)
    target_node_id = Column(String(100), index=True)
    edge_type = Column(String(50))
    weight = Column(Float, default=1.0)
    confidence = Column(Float, default=0.5)
    sample_count = Column(Integer, default=1)
    attributes = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

_engine = None
_async_engine = None
_SessionLocal = None

def get_engine(database_url: str = "sqlite:///./career_regret.db"):
    global _engine
    if _engine is None:
        _engine = create_engine(database_url, connect_args={"check_same_thread": False} if "sqlite" in database_url else {})
    return _engine

def init_db(database_url: str = "sqlite:///./career_regret.db"):
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return engine

def get_session(database_url: str = "sqlite:///./career_regret.db"):
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(database_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal()
