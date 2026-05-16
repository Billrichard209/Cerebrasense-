"""
CerebraSense Clinical Database Layer
Handles patient records, session history, and biomarker persistence.
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from pathlib import Path

Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    external_id = Column(String, unique=True, index=True) # e.g. OAS2_0001
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON, default=dict)
    
    sessions = relationship("ClinicalSession", back_populates="patient", cascade="all, delete-orphan")

class ClinicalSession(Base):
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    session_id = Column(String) # e.g. d0001
    
    # Model Outputs
    probability = Column(Float)
    label = Column(String)
    
    # Longitudinal Intelligence
    velocity = Column(Float, default=0.0)
    is_rapid_decline = Column(Boolean, default=False)
    
    # Biomarkers (Hippocampal Volumetrics)
    hippo_vol_mm3 = Column(Float, nullable=True)
    tiv_mm3 = Column(Float, nullable=True)
    normalized_hippo = Column(Float, nullable=True)
    
    # Metadata
    mri_path = Column(String)
    clinical_meta = Column(JSON, default=dict) # Age, Sex, MMSE
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="sessions")

# --- Engine Setup ---

def get_db_engine(db_path: str = "cerebrasense.db"):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine

def get_db_session(engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

# --- Migration Helpers ---

def migrate_csv_to_db(csv_path: Path, db_session: Session):
    """Import longitudinal data from CSV into the structured database."""
    import pandas as pd
    import json
    
    if not csv_path.exists():
        return
    
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        subj_id = str(row.get("meta_subject_id", "Unknown"))
        sess_id = str(row.get("meta_session_id", "Unknown"))
        
        # 1. Get or Create Patient
        patient = db_session.query(Patient).filter(Patient.external_id == subj_id).first()
        if not patient:
            patient = Patient(external_id=subj_id)
            db_session.add(patient)
            db_session.flush()
        
        # 2. Add Session if not exists
        existing = db_session.query(ClinicalSession).filter(
            ClinicalSession.patient_id == patient.id,
            ClinicalSession.session_id == sess_id
        ).first()
        
        if not existing:
            # Extract metadata
            clinical = {}
            try:
                meta_str = str(row.get("meta", "{}")).replace("'", "\"")
                m = json.loads(meta_str)
                clinical = m.get("oasis2_metadata", {})
            except: pass
            
            session = ClinicalSession(
                patient_id=patient.id,
                session_id=sess_id,
                probability=float(row.get("probability_class_1", 0.0)),
                label=str(row.get("predicted_label", "Unknown")),
                clinical_meta=clinical
            )
            db_session.add(session)
    
    db_session.commit()
