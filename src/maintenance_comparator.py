"""
Before-and-After Maintenance Comparison Module
Evaluates repair efficacy by comparing pre-maintenance and post-maintenance images,
analyzing thermal anomaly dissipation, and issuing automated resolution certificates.
"""

import cv2
import numpy as np
import datetime
from typing import Dict, Any, Tuple
from src.database import SessionLocal, InspectionRecord, Panel

class MaintenanceComparator:
    """Verifies physical and thermal anomaly resolution after technician servicing"""

    @staticmethod
    def compare_inspections(
        pre_img_bgr: np.ndarray,
        post_img_bgr: np.ndarray,
        pre_thermal_bgr: np.ndarray,
        post_thermal_bgr: np.ndarray,
        pre_delta_t: float = 24.5
    ) -> Dict[str, Any]:
        """
        Performs quantitative multi-modal before/after comparison.
        Returns visual comparison mosaic, clearance percentage, and verification verdict.
        """
        # Resize post images to match pre images if differing
        h, w = pre_img_bgr.shape[:2]
        post_img_resized = cv2.resize(post_img_bgr, (w, h))
        post_thermal_resized = cv2.resize(post_thermal_bgr, (w, h))

        # Optical difference map
        gray_pre = cv2.cvtColor(pre_img_bgr, cv2.COLOR_BGR2GRAY)
        gray_post = cv2.cvtColor(post_img_resized, cv2.COLOR_BGR2GRAY)
        diff_optical = cv2.absdiff(gray_pre, gray_post)
        diff_optical_norm = cv2.normalize(diff_optical, None, 0, 255, cv2.NORM_MINMAX)
        diff_heatmap = cv2.applyColorMap(diff_optical_norm, cv2.COLORMAP_MAGMA)

        # Thermal gradient dissipation calculation
        # Radiometric intensity check
        gray_thm_pre = cv2.cvtColor(pre_thermal_bgr, cv2.COLOR_BGR2GRAY)
        gray_thm_post = cv2.cvtColor(post_thermal_resized, cv2.COLOR_BGR2GRAY)
        
        max_pre_intensity = float(np.max(gray_thm_pre))
        max_post_intensity = float(np.max(gray_thm_post))
        
        # Estimate post Delta-T based on relative radiometric reduction
        intensity_ratio = max_post_intensity / max(1.0, max_pre_intensity)
        post_delta_t = round(pre_delta_t * intensity_ratio, 1)
        post_delta_t = max(1.2, min(post_delta_t, pre_delta_t))

        delta_t_reduction = round(pre_delta_t - post_delta_t, 1)
        clearance_pct = round((delta_t_reduction / max(0.1, pre_delta_t)) * 100.0, 1)
        clearance_pct = max(0.0, min(100.0, clearance_pct))

        # Determination of resolution verdict
        if clearance_pct >= 70.0 or post_delta_t < 6.0:
            verdict = "VERIFIED RESOLVED"
            verdict_color = (0, 200, 0) # Green
            summary = "Anomaly successfully cleared. Panel operating at nominal cell temperature."
        elif clearance_pct >= 40.0:
            verdict = "PARTIAL CLEARANCE"
            verdict_color = (0, 165, 255) # Orange
            summary = "Substantial thermal cooling observed, but residual anomaly persists. Follow-up audit recommended."
        else:
            verdict = "UNRESOLVED"
            verdict_color = (0, 0, 240) # Red
            summary = "Thermal or optical anomaly remains active. Immediate secondary maintenance required."

        # Create 4-panel comparison montage
        # 1. Before RGB | 2. After RGB
        # 3. Before Thermal | 4. After Thermal
        top_before = pre_img_bgr.copy()
        cv2.putText(top_before, "BEFORE REPAIR (RGB)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        top_after = post_img_resized.copy()
        cv2.putText(top_after, "AFTER REPAIR (RGB)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        bot_before = pre_thermal_bgr.copy()
        cv2.putText(bot_before, f"BEFORE IR (dT: +{pre_delta_t}C)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        bot_after = post_thermal_resized.copy()
        cv2.putText(bot_after, f"AFTER IR (dT: +{post_delta_t}C)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        row1 = np.hstack([top_before, top_after])
        row2 = np.hstack([bot_before, bot_after])
        mosaic = np.vstack([row1, row2])

        # Add verdict footer banner
        mh, mw = mosaic.shape[:2]
        banner_h = 50
        banner = np.zeros((banner_h, mw, 3), dtype=np.uint8)
        banner[:] = (25, 28, 35)
        
        status_text = f"RESOLUTION STATUS: {verdict} | THERMAL CLEARANCE: {clearance_pct}% (dT dropped from {pre_delta_t}C to {post_delta_t}C)"
        cv2.putText(banner, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, verdict_color, 2, cv2.LINE_AA)
        
        final_mosaic = np.vstack([mosaic, banner])

        return {
            "verdict": verdict,
            "summary": summary,
            "pre_delta_t_c": pre_delta_t,
            "post_delta_t_c": post_delta_t,
            "thermal_reduction_c": delta_t_reduction,
            "clearance_percentage": clearance_pct,
            "mosaic_bgr": final_mosaic
        }

    @staticmethod
    def resolve_inspection_record(
        inspection_id: int,
        technician_notes: str,
        after_img_path: str,
        verdict: str
    ) -> bool:
        """Updates inspection record in database upon verified completion"""
        db = SessionLocal()
        try:
            record = db.query(InspectionRecord).filter(InspectionRecord.id == inspection_id).first()
            if record:
                record.status = "Resolved" if verdict == "VERIFIED RESOLVED" else "In Progress"
                record.technician_notes = technician_notes
                record.after_maintenance_img_path = after_img_path
                record.maintenance_resolution_date = datetime.datetime.utcnow()
                record.resolution_verification = verdict

                # Update panel status
                if verdict == "VERIFIED RESOLVED":
                    panel = db.query(Panel).filter(Panel.id == record.panel_id).first()
                    if panel:
                        panel.current_status = "Normal"
                        panel.risk_score = max(0.0, panel.risk_score - 20.0)

                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()
