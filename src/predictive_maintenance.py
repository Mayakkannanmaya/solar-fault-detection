"""
Predictive Maintenance Engine for Photovoltaic Arrays
Analyzes historical anomaly patterns, recurrent thermal stress, and degradation trajectories
to forecast failures and recommend preventive interventions before critical breakdown.
"""

import datetime
from typing import Dict, Any, List
from src.database import SessionLocal, Panel, InspectionRecord

class PredictiveMaintenanceEngine:
    """Predictive failure forecasting based on temporal PV telemetry and inspection history"""

    @staticmethod
    def assess_panel_health(panel_id: int) -> Dict[str, Any]:
        """
        Calculates health index, failure probability, and preventive recommendations
        for a specific solar panel based on historical inspections.
        """
        db = SessionLocal()
        try:
            panel = db.query(Panel).filter(Panel.id == panel_id).first()
            if not panel:
                return {}

            inspections = db.query(InspectionRecord).filter(
                InspectionRecord.panel_id == panel_id
            ).order_by(InspectionRecord.timestamp.desc()).all()

            total_inspections = len(inspections)
            fault_counts = {}
            thermal_spikes = 0
            critical_events = 0
            max_observed_delta_t = 0.0

            for insp in inspections:
                f_type = insp.fault_type
                fault_counts[f_type] = fault_counts.get(f_type, 0) + 1
                if insp.delta_t_c and insp.delta_t_c > 15.0:
                    thermal_spikes += 1
                if insp.severity in ["Critical", "High"]:
                    critical_events += 1
                if insp.delta_t_c and insp.delta_t_c > max_observed_delta_t:
                    max_observed_delta_t = insp.delta_t_c

            # Base Health calculation (100 is pristine)
            health_index = 100.0
            health_index -= (thermal_spikes * 12.0)
            health_index -= (critical_events * 15.0)
            health_index -= (total_inspections * 2.0)
            health_index = max(15.0, min(100.0, health_index))

            # Failure Probability calculation (0% to 100%)
            failure_prob = round((100.0 - health_index) * 0.9 + (thermal_spikes * 4.0), 1)
            failure_prob = max(2.0, min(98.0, failure_prob))

            # Predictive Flagging Logic
            preventive_flag = False
            flag_reason = "Operating within normal parameters."
            urgency = "Low"

            if thermal_spikes >= 3:
                preventive_flag = True
                flag_reason = (
                    f"Recurrent Thermal Stress: Panel {panel.panel_code} exhibited {thermal_spikes} "
                    f"thermal anomalies with peak ΔT of {max_observed_delta_t:.1f}°C. "
                    "Pre-failure thermal runaway probable."
                )
                urgency = "Immediate Preventive Replacement"
            elif critical_events >= 2:
                preventive_flag = True
                flag_reason = (
                    f"High Fault Frequency: Panel {panel.panel_code} logged {critical_events} "
                    "severe fault incidents. Potential junction box or bypass diode degradation."
                )
                urgency = "Scheduled Overhaul Required"
            elif fault_counts.get("Dust & Soiling", 0) >= 3:
                preventive_flag = True
                flag_reason = (
                    f"Chronic Soiling: Panel {panel.panel_code} shows repetitive dirt accumulation, "
                    "inducing localized reverse bias heating risks."
                )
                urgency = "Automated Wash Cycle Recommendation"

            # Projected Annual Power Yield Loss
            estimated_power_loss_kwh = round((100.0 - health_index) * 4.8, 1)

            return {
                "panel_id": panel.id,
                "panel_code": panel.panel_code,
                "array_id": panel.array_id,
                "string_id": panel.string_id,
                "total_inspections": total_inspections,
                "fault_distribution": fault_counts,
                "thermal_spikes_count": thermal_spikes,
                "max_delta_t_c": max_observed_delta_t,
                "health_index": round(health_index, 1),
                "failure_risk_pct": failure_prob,
                "predictive_flag": preventive_flag,
                "flag_reason": flag_reason,
                "preventive_urgency": urgency,
                "estimated_annual_loss_kwh": estimated_power_loss_kwh
            }

        finally:
            db.close()

    @staticmethod
    def get_farm_predictive_summary() -> List[Dict[str, Any]]:
        """Returns predictive maintenance reports for all panels in the farm"""
        db = SessionLocal()
        summary = []
        try:
            panels = db.query(Panel).all()
            for p in panels:
                res = PredictiveMaintenanceEngine.assess_panel_health(p.id)
                if res:
                    summary.append(res)
        finally:
            db.close()
        
        # Sort by failure risk descending
        summary.sort(key=lambda x: x["failure_risk_pct"], reverse=True)
        return summary
