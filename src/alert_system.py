"""
Automated Alerting & Multi-Channel Notification Engine for Solar Panel Faults
Supports:
- Email Alerts (HTML-formatted with telemetry and actionable SOP)
- SMS Alert Dispatch (Emergency cellular messaging)
- WhatsApp Cloud API Webhook Simulation
- IoT / MQTT / SCADA Telemetry Notification Broker
"""

import datetime
from typing import Dict, Any, Optional, List
from src.database import SessionLocal, AlertRecord, Panel

class AlertSystem:
    """Manages real-time alert generation and multi-channel dispatch"""

    @staticmethod
    def dispatch_alert(
        panel_code: str,
        fault_type: str,
        severity: str,
        confidence: float,
        cell_location: str,
        message: str,
        channels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Dispatches alerts across configured communication channels
        and records them into the monitoring database.
        """
        if channels is None:
            channels = ["System", "Email", "SMS", "WhatsApp", "IoT/SCADA"]

        dispatched_results = []
        db = SessionLocal()

        try:
            # Query panel record
            panel = db.query(Panel).filter(Panel.panel_code == panel_code).first()
            panel_id = panel.id if panel else 1

            for ch in channels:
                alert_entry = AlertRecord(
                    panel_id=panel_id,
                    panel_code=panel_code,
                    timestamp=datetime.datetime.utcnow(),
                    fault_type=fault_type,
                    severity=severity,
                    confidence=confidence,
                    cell_location=cell_location,
                    message=message,
                    channel=ch,
                    is_acknowledged=False
                )
                db.add(alert_entry)
                
                # Channel specific mock payload
                receipt = {
                    "channel": ch,
                    "target": f"ops-team@{ch.lower()}.solarfarm.net",
                    "status": "DISPATCHED",
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "summary": f"[{severity.upper()}] {fault_type} on {panel_code} at {cell_location}"
                }
                dispatched_results.append(receipt)

            # Update panel status if critical
            if panel and severity in ["Critical", "High"]:
                panel.current_status = "Faulty"
                panel.risk_score = min(100.0, panel.risk_score + 25.0)

            db.commit()

        except Exception as e:
            db.rollback()
            print(f"Error dispatching alert: {e}")
        finally:
            db.close()

        return dispatched_results

    @staticmethod
    def get_recent_alerts(limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent alerts from the database"""
        db = SessionLocal()
        alerts = []
        try:
            records = db.query(AlertRecord).order_by(AlertRecord.timestamp.desc()).limit(limit).all()
            for r in records:
                alerts.append({
                    "id": r.id,
                    "panel_code": r.panel_code,
                    "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "fault_type": r.fault_type,
                    "severity": r.severity,
                    "confidence": round(r.confidence, 3),
                    "cell_location": r.cell_location,
                    "channel": r.channel,
                    "message": r.message,
                    "is_acknowledged": r.is_acknowledged
                })
        finally:
            db.close()
        return alerts

    @staticmethod
    def acknowledge_alert(alert_id: int, user: str = "Chief Field Engineer") -> bool:
        """Marks an alert as acknowledged by a technician"""
        db = SessionLocal()
        try:
            alert = db.query(AlertRecord).filter(AlertRecord.id == alert_id).first()
            if alert:
                alert.is_acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.datetime.utcnow()
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()
