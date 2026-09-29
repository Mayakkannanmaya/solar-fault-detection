"""
FastAPI Enterprise REST API Server for Solar Panel Fault Monitoring
Exposes endpoints for:
- Drone/IoT Image Ingestion & Real-Time AI Inference
- Solar Farm Digital Twin & Panel Telemetry
- Automated Multi-Channel Alerting
- Predictive Maintenance & Health Index Forecasting
- Before/After Maintenance Resolution Verification
- IEC 62446 PDF and CSV Audit Report Generation
- Deep Learning Benchmark & Metrics
"""

import os
import cv2
import numpy as np
import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any

from config import STATIC_DIR, REPORTS_DIR, FAULT_CLASSES, WEIGHTS_DIR
from src.database import SessionLocal, Panel, InspectionRecord, AlertRecord
from src.inference_engine import SolarFaultInferenceEngine
from src.ai_assistant import AISolarFaultAssistant
from src.alert_system import AlertSystem
from src.predictive_maintenance import PredictiveMaintenanceEngine
from src.maintenance_comparator import MaintenanceComparator
from src.report_generator import ReportGenerator

app = FastAPI(
    title="Solar PV AI Fault Detection & Monitoring API",
    description="Deep Learning, Thermal Radiometry & Predictive Maintenance REST API",
    version="2.0.0"
)

# Enable CORS for external dashboards & web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static artifacts directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize master inference engine
weights_file = WEIGHTS_DIR / "dual_modal_pvnet.pth"
inference_engine = SolarFaultInferenceEngine(
    weights_path=str(weights_file) if weights_file.exists() else None
)

@app.get("/")
def root():
    return {
        "system": "AI Solar Panel Fault Detection, Localization & Predictive Maintenance System",
        "version": "2.0.0",
        "status": "ONLINE",
        "documentation": "/docs",
        "endpoints": [
            "/api/farm", "/api/panels", "/api/inspect",
            "/api/alerts", "/api/predictive", "/api/maintenance/compare",
            "/api/reports/pdf", "/api/reports/csv", "/api/metrics"
        ]
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(inference_engine.device),
        "classes_supported": len(FAULT_CLASSES),
        "database_connected": True
    }

@app.get("/api/farm")
def get_farm_overview():
    """Returns digital twin overview of the entire PV array layout"""
    db = SessionLocal()
    try:
        panels = db.query(Panel).all()
        summary = {
            "total_panels": len(panels),
            "normal_count": sum(1 for p in panels if p.current_status == "Normal"),
            "warning_count": sum(1 for p in panels if p.current_status in ["Medium", "Low"]),
            "faulty_count": sum(1 for p in panels if p.current_status in ["High", "Critical"]),
            "panels": [
                {
                    "id": p.id,
                    "code": p.panel_code,
                    "array": p.array_id,
                    "string": p.string_id,
                    "row": p.grid_row,
                    "col": p.grid_col,
                    "status": p.current_status,
                    "risk_score": p.risk_score,
                    "degradation_pct": p.degradation_pct,
                    "last_inspected": p.last_inspected.isoformat() if p.last_inspected else None
                }
                for p in panels
            ]
        }
        return summary
    finally:
        db.close()

@app.get("/api/panels/{panel_code}")
def get_panel_details(panel_code: str):
    """Retrieves full panel telemetry, historical audits, and risk assessment"""
    db = SessionLocal()
    try:
        panel = db.query(Panel).filter(Panel.panel_code == panel_code).first()
        if not panel:
            raise HTTPException(status_code=404, detail="Panel code not found")

        inspections = db.query(InspectionRecord).filter(
            InspectionRecord.panel_id == panel.id
        ).order_by(InspectionRecord.timestamp.desc()).all()

        predictive_health = PredictiveMaintenanceEngine.assess_panel_health(panel.id)

        return {
            "panel": {
                "id": panel.id,
                "code": panel.panel_code,
                "array": panel.array_id,
                "string": panel.string_id,
                "status": panel.current_status,
                "risk_score": panel.risk_score,
                "degradation_pct": panel.degradation_pct
            },
            "predictive_health": predictive_health,
            "inspections": [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "fault_type": r.fault_type,
                    "severity": r.severity,
                    "confidence": r.confidence,
                    "cell_location": f"R{r.cell_row} C{r.cell_col}" if r.cell_row else "Module",
                    "delta_t_c": r.delta_t_c,
                    "max_temp_c": r.max_temp_c,
                    "rgb_img": f"/static/{os.path.basename(r.rgb_image_path)}" if r.rgb_image_path else None,
                    "thermal_img": f"/static/{os.path.basename(r.thermal_image_path)}" if r.thermal_image_path else None,
                    "quad_img": f"/static/{os.path.basename(r.annotated_image_path)}" if r.annotated_image_path else None,
                    "possible_cause": r.possible_cause,
                    "recommended_action": r.recommended_action,
                    "status": r.status
                }
                for r in inspections
            ]
        }
    finally:
        db.close()

@app.post("/api/inspect")
async def inspect_solar_panel(
    rgb_file: UploadFile = File(...),
    thermal_file: UploadFile = File(...),
    panel_code: str = Form("PNL-A1-03")
):
    """
    Ingests paired RGB and Thermal images, executes deep learning inference,
    localizes faulty cells (Row/Col), assesses severity, creates Grad-CAM,
    and returns AI Assistant recommendations.
    """
    try:
        rgb_bytes = await rgb_file.read()
        thm_bytes = await thermal_file.read()

        nparr_rgb = np.frombuffer(rgb_bytes, np.uint8)
        nparr_thm = np.frombuffer(thm_bytes, np.uint8)

        rgb_img = cv2.imdecode(nparr_rgb, cv2.IMREAD_COLOR)
        thm_img = cv2.imdecode(nparr_thm, cv2.IMREAD_COLOR)

        if rgb_img is None or thm_img is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        # Run pipeline
        result = inference_engine.run_full_inspection(
            rgb_bgr=rgb_img,
            thermal_bgr=thm_img,
            panel_code=panel_code,
            save_visualizations=True
        )

        return {
            "success": True,
            "panel_code": result["panel_code"],
            "fault_type": result["fault_type"],
            "confidence": result["confidence"],
            "severity": result["severity"],
            "cell_location": {
                "row": result["cell_row"],
                "column": result["cell_col"],
                "bbox": result["bbox"]
            },
            "thermal_telemetry": {
                "ambient_temp_c": result["ambient_temp_c"],
                "max_temp_c": result["max_temp_c"],
                "delta_t_c": result["delta_t_c"]
            },
            "ai_assistant": result["assistant"],
            "alert_triggered": result["alert"] is not None,
            "visualizations": {
                "rgb_url": f"/static/{os.path.basename(result['rgb_save_path'])}",
                "thermal_url": f"/static/{os.path.basename(result['thm_save_path'])}",
                "quad_diagnostic_url": f"/static/{os.path.basename(result['quad_save_path'])}"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
def get_alerts(limit: int = 20):
    """Fetches real-time alert logs with acknowledgement status"""
    return AlertSystem.get_recent_alerts(limit=limit)

@app.post("/api/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: int, user: str = "Chief Engineer"):
    """Acknowledges an active alert"""
    success = AlertSystem.acknowledge_alert(alert_id=alert_id, user=user)
    if not success:
        raise HTTPException(status_code=404, detail="Alert ID not found or could not be acknowledged")
    return {"success": True, "alert_id": alert_id, "acknowledged_by": user}

@app.get("/api/predictive")
def get_predictive_maintenance_overview():
    """Returns predictive maintenance risk rankings across the farm"""
    return PredictiveMaintenanceEngine.get_farm_predictive_summary()

@app.post("/api/maintenance/compare")
async def compare_maintenance_images(
    pre_rgb: UploadFile = File(...),
    post_rgb: UploadFile = File(...),
    pre_thermal: UploadFile = File(...),
    post_thermal: UploadFile = File(...),
    pre_delta_t: float = Form(24.5)
):
    """Compares pre-maintenance and post-maintenance images to verify anomaly clearance"""
    try:
        rgb_pre_bytes = await pre_rgb.read()
        rgb_post_bytes = await post_rgb.read()
        thm_pre_bytes = await pre_thermal.read()
        thm_post_bytes = await post_thermal.read()

        pre_img = cv2.imdecode(np.frombuffer(rgb_pre_bytes, np.uint8), cv2.IMREAD_COLOR)
        post_img = cv2.imdecode(np.frombuffer(rgb_post_bytes, np.uint8), cv2.IMREAD_COLOR)
        pre_thm = cv2.imdecode(np.frombuffer(thm_pre_bytes, np.uint8), cv2.IMREAD_COLOR)
        post_thm = cv2.imdecode(np.frombuffer(thm_post_bytes, np.uint8), cv2.IMREAD_COLOR)

        result = MaintenanceComparator.compare_inspections(
            pre_img, post_img, pre_thm, post_thm, pre_delta_t=pre_delta_t
        )

        # Save mosaic
        mosaic_filename = f"maintenance_compare_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
        mosaic_path = STATIC_DIR / mosaic_filename
        cv2.imwrite(str(mosaic_path), result["mosaic_bgr"])

        return {
            "verdict": result["verdict"],
            "summary": result["summary"],
            "clearance_percentage": result["clearance_percentage"],
            "pre_delta_t_c": result["pre_delta_t_c"],
            "post_delta_t_c": result["post_delta_t_c"],
            "thermal_reduction_c": result["thermal_reduction_c"],
            "mosaic_url": f"/static/{mosaic_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/pdf")
def download_pdf_report(farm_name: str = "Helios One 50MW Solar Facility"):
    """Generates and serves IEC-compliant PDF inspection audit report"""
    pdf_path = ReportGenerator.generate_pdf_report(farm_name=farm_name)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(pdf_path)
    )

@app.get("/api/reports/csv")
def download_csv_report():
    """Generates and serves tabular CSV inspection log"""
    csv_path = ReportGenerator.generate_csv_report()
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=os.path.basename(csv_path)
    )

@app.get("/api/metrics")
def get_model_benchmarks():
    """Retrieves deep learning evaluation metrics and confusion matrix"""
    metrics_file = WEIGHTS_DIR.parent / "evaluation_metrics.json"
    if metrics_file.exists():
        import json
        with open(str(metrics_file), "r") as f:
            return json.load(f)
    return {
        "status": "Benchmarks available after training",
        "models": ["DualModalPVNet", "TransferPVNet"]
    }
