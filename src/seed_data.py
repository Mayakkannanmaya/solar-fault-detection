"""
Database Initialization & Solar Farm Digital Twin Seeder
Generates realistic solar arrays, historical inspections, alerts, and baseline telemetry.
"""

import os
import random
import datetime
import cv2
import numpy as np
from pathlib import Path
from src.database import init_db, SessionLocal, Panel, InspectionRecord, AlertRecord
from src.dataset_generator import create_base_solar_panel, inject_fault
from src.ai_assistant import AISolarFaultAssistant
from src.explainable_ai import XAIVisualizer
from config import FAULT_CLASSES, DATASET_RGB, DATASET_THERMAL, STATIC_DIR

def seed_solar_farm(num_panels: int = 24):
    """
    Initializes database and seeds solar farm with:
    - 24 Panels arranged in 4 strings across 2 arrays (Alpha & Beta)
    - Realistic historical inspections covering all fault classes
    - Alerts for high/critical anomalies
    """
    print("Initializing Database schema...")
    init_db()
    db = SessionLocal()

    try:
        # Check if already seeded with inspections
        if db.query(Panel).count() >= num_panels and db.query(InspectionRecord).count() >= num_panels:
            print("Database already contains seeded panels and inspections.")
            return
        
        # Clear existing partial data
        db.query(AlertRecord).delete()
        db.query(InspectionRecord).delete()
        db.query(Panel).delete()
        db.commit()

        print(f"Seeding Solar Farm with {num_panels} PV Modules...")
        panels = []
        arrays = ["Array-Alpha", "Array-Beta"]
        
        # 1. Create Panels
        for i in range(1, num_panels + 1):
            arr = arrays[0] if i <= 12 else arrays[1]
            string_num = ((i - 1) // 6) + 1
            str_id = f"String-0{string_num}"
            code = f"PNL-{arr.split('-')[1][0]}{string_num}-{((i-1)%6)+1:02d}"

            panel = Panel(
                panel_code=code,
                array_id=arr,
                string_id=str_id,
                grid_row=((i - 1) // 6) + 1,
                grid_col=((i - 1) % 6) + 1,
                capacity_watts=400.0,
                current_status="Normal",
                risk_score=float(random.randint(5, 25)),
                degradation_pct=round(random.uniform(0.4, 1.8), 2)
            )
            db.add(panel)
            panels.append(panel)

        db.commit()

        # 2. Assign diverse realistic faults to showcase all categories
        fault_assignment = [
            "Normal", "Hot Spot", "Micro Crack", "Dust & Soiling",
            "Bird Dropping", "Physical Damage", "Discoloration", "Cell Defect",
            "Partial Shading", "Electrical Anomaly", "Normal", "Hot Spot",
            "Dust & Soiling", "Normal", "Micro Crack", "Normal",
            "Bird Dropping", "Partial Shading", "Normal", "Physical Damage",
            "Discoloration", "Cell Defect", "Normal", "Electrical Anomaly"
        ]

        print("Generating dual-modal inspection images and logging historical data...")
        for idx, panel in enumerate(panels):
            fault = fault_assignment[idx % len(fault_assignment)]
            rgb_base, thermal_base, cells = create_base_solar_panel()
            fault_data = inject_fault(rgb_base, thermal_base, cells, fault)

            # File paths
            rgb_path = STATIC_DIR / f"{panel.panel_code}_inspect_rgb.jpg"
            thm_path = STATIC_DIR / f"{panel.panel_code}_inspect_thm.jpg"
            quad_path = STATIC_DIR / f"{panel.panel_code}_inspect_quad.jpg"

            # Assistant diagnosis
            confidence = round(0.92 + random.uniform(-0.05, 0.06), 3) if fault != "Normal" else round(0.97 + random.uniform(-0.02, 0.02), 3)
            diag = AISolarFaultAssistant.analyze_fault(
                fault_type=fault,
                confidence=confidence,
                panel_code=panel.panel_code,
                cell_row=fault_data["cell_row"],
                cell_col=fault_data["cell_col"],
                delta_t=fault_data["delta_t_c"],
                max_temp=fault_data["max_temp_c"]
            )

            # Generate quad view
            h, w = fault_data["rgb"].shape[:2]
            dummy_cam = np.zeros((h, w), dtype=np.float32)
            if fault_data["cell_row"]:
                bx = fault_data["bbox"]
                cx = int((bx[0] + bx[2]) / 2)
                cy = int((bx[1] + bx[3]) / 2)
                cv2.circle(dummy_cam, (cx, cy), 45, 1.0, -1)
                dummy_cam = cv2.GaussianBlur(dummy_cam, (25, 25), 9)

            quad = XAIVisualizer.generate_multimodal_quad_view(
                fault_data["rgb"],
                fault_data["thermal"],
                dummy_cam,
                fault_data["cell_row"],
                fault_data["cell_col"],
                fault,
                confidence,
                fault_data["bbox"]
            )

            cv2.imwrite(str(rgb_path), fault_data["rgb"])
            cv2.imwrite(str(thm_path), fault_data["thermal"])
            cv2.imwrite(str(quad_path), quad)

            # Create Inspection Record
            rec = InspectionRecord(
                panel_id=panel.id,
                panel_code=panel.panel_code,
                timestamp=datetime.datetime.utcnow() - datetime.timedelta(hours=random.randint(1, 72)),
                fault_type=fault,
                severity=diag["severity"],
                confidence=confidence,
                cell_row=fault_data["cell_row"],
                cell_col=fault_data["cell_col"],
                bbox_x1=float(fault_data["bbox"][0]),
                bbox_y1=float(fault_data["bbox"][1]),
                bbox_x2=float(fault_data["bbox"][2]),
                bbox_y2=float(fault_data["bbox"][3]),
                ambient_temp_c=fault_data["ambient_temp_c"],
                max_temp_c=fault_data["max_temp_c"],
                delta_t_c=fault_data["delta_t_c"],
                rgb_image_path=str(rgb_path),
                thermal_image_path=str(thm_path),
                annotated_image_path=str(quad_path),
                possible_cause=diag["primary_cause"],
                performance_impact=diag["performance_impact"],
                recommended_action=diag["recommended_action"],
                immediate_inspection_required=diag["immediate_inspection_required"],
                status="Resolved" if diag["severity"] == "Normal" else "Pending"
            )
            db.add(rec)

            # Update panel status
            panel.current_status = diag["severity"]
            if diag["severity"] in ["Critical", "High"]:
                panel.risk_score = round(random.uniform(65.0, 92.0), 1)
                
                # Add alert record
                alert = AlertRecord(
                    panel_id=panel.id,
                    panel_code=panel.panel_code,
                    timestamp=rec.timestamp,
                    fault_type=fault,
                    severity=diag["severity"],
                    confidence=confidence,
                    cell_location=f"R{fault_data['cell_row']} C{fault_data['cell_col']}" if fault_data['cell_row'] else "Module",
                    message=f"🚨 Immediate action needed: {fault} detected on {panel.panel_code} with peak ΔT +{fault_data['delta_t_c']}°C",
                    channel="Email & SMS",
                    is_acknowledged=False
                )
                db.add(alert)

        db.commit()
        print("Solar farm successfully seeded with panels, inspections, and alerts!")

    except Exception as e:
        db.rollback()
        print(f"Error seeding solar farm: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_solar_farm()
