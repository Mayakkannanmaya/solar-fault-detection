"""
Database Schema & Data Access Layer using SQLAlchemy for Solar Panel Fault Monitoring
"""

import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, desc
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from config import DB_PATH

Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Panel(Base):
    __tablename__ = "panels"

    id = Column(Integer, primary_key=True, index=True)
    panel_code = Column(String(50), unique=True, index=True, nullable=False) # e.g. "PNL-A1-03"
    array_id = Column(String(50), nullable=False) # e.g. "Array-Alpha"
    string_id = Column(String(50), nullable=False) # e.g. "String-01"
    grid_row = Column(Integer, default=1)
    grid_col = Column(Integer, default=1)
    capacity_watts = Column(Float, default=400.0)
    current_status = Column(String(50), default="Normal") # Normal, Warning, Faulty, Maintenance
    last_inspected = Column(DateTime, default=datetime.datetime.utcnow)
    risk_score = Column(Float, default=0.0) # 0.0 to 100.0
    degradation_pct = Column(Float, default=0.5) # Estimated yearly degradation

    inspections = relationship("InspectionRecord", back_populates="panel", cascade="all, delete-orphan")
    alerts = relationship("AlertRecord", back_populates="panel", cascade="all, delete-orphan")

class InspectionRecord(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    panel_id = Column(Integer, ForeignKey("panels.id"), nullable=False)
    panel_code = Column(String(50), index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    
    # Fault Information
    fault_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False) # Normal, Low, Medium, High, Critical
    confidence = Column(Float, default=0.0) # 0.0 to 1.0
    
    # Localization (6x10 PV cell grid)
    cell_row = Column(Integer, nullable=True) # 1 to 6
    cell_col = Column(Integer, nullable=True) # 1 to 10
    bbox_x1 = Column(Float, default=0.0)
    bbox_y1 = Column(Float, default=0.0)
    bbox_x2 = Column(Float, default=0.0)
    bbox_y2 = Column(Float, default=0.0)
    
    # Thermal Metrics
    ambient_temp_c = Column(Float, default=28.0)
    max_temp_c = Column(Float, default=42.0)
    delta_t_c = Column(Float, default=0.0) # max_temp - ambient_temp
    
    # Image Paths
    rgb_image_path = Column(String(255), nullable=True)
    thermal_image_path = Column(String(255), nullable=True)
    explain_map_path = Column(String(255), nullable=True)
    annotated_image_path = Column(String(255), nullable=True)
    
    # AI Assistant Diagnostics
    possible_cause = Column(Text, nullable=True)
    performance_impact = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
    immediate_inspection_required = Column(Boolean, default=False)
    
    # Maintenance Workflow
    status = Column(String(50), default="Pending") # Pending, In Progress, Resolved
    technician_notes = Column(Text, nullable=True)
    after_maintenance_img_path = Column(String(255), nullable=True)
    maintenance_resolution_date = Column(DateTime, nullable=True)
    resolution_verification = Column(String(100), nullable=True)

    panel = relationship("Panel", back_populates="inspections")

class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    panel_id = Column(Integer, ForeignKey("panels.id"), nullable=False)
    panel_code = Column(String(50), index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    fault_type = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.0)
    cell_location = Column(String(50), nullable=True)
    message = Column(Text, nullable=False)
    channel = Column(String(50), default="System") # Email, SMS, WhatsApp, IoT, Webhook
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)

    panel = relationship("Panel", back_populates="alerts")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
