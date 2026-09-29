"""
Comprehensive Test Suite for HeliosAI Photovoltaic Inspection Suite
Tests:
1. Database Models & Schema
2. AI Solar Fault Assistant Diagnostics
3. DualModalPVNet & TransferPVNet PyTorch Models
4. Explainable AI Grad-CAM Engine
5. Predictive Maintenance & Anomaly Forecasting
6. Before/After Maintenance Comparator
7. Alert System Multi-Channel Dispatch
8. Automated PDF & CSV Report Generators
"""

import os
import unittest
import numpy as np
import cv2
import torch
from pathlib import Path

from config import FAULT_CLASSES, PANEL_GRID_ROWS, PANEL_GRID_COLS, STATIC_DIR, REPORTS_DIR
from src.database import init_db, SessionLocal, Panel, InspectionRecord, AlertRecord
from src.dataset_generator import create_base_solar_panel, inject_fault
from src.ai_assistant import AISolarFaultAssistant
from src.models import DualModalPVNet, TransferPVNet, ModelEvaluator
from src.explainable_ai import GradCAM, XAIVisualizer
from src.alert_system import AlertSystem
from src.predictive_maintenance import PredictiveMaintenanceEngine
from src.maintenance_comparator import MaintenanceComparator
from src.report_generator import ReportGenerator
from src.inference_engine import SolarFaultInferenceEngine

class TestSolarMonitoringSystem(unittest.TestCase):

    def setUp(self):
        init_db()

    def test_01_database_operations(self):
        """Test panel and inspection creation and query"""
        db = SessionLocal()
        panel = db.query(Panel).first()
        self.assertIsNotNone(panel, "Database should contain seeded panels")
        self.assertTrue(len(panel.panel_code) > 0)
        inspections = db.query(InspectionRecord).filter(InspectionRecord.panel_id == panel.id).all()
        self.assertTrue(len(inspections) >= 0)
        db.close()

    def test_02_dataset_generator(self):
        """Test base solar panel creation and fault injection"""
        rgb, thm, cells = create_base_solar_panel(600, 360)
        self.assertEqual(rgb.shape, (360, 600, 3))
        self.assertEqual(thm.shape, (360, 600, 3))
        self.assertEqual(len(cells), PANEL_GRID_ROWS * PANEL_GRID_COLS)

        # Inject Hot Spot
        result = inject_fault(rgb.copy(), thm.copy(), cells, "Hot Spot")
        self.assertEqual(result["fault_type"], "Hot Spot")
        self.assertIsNotNone(result["cell_row"])
        self.assertIsNotNone(result["cell_col"])
        self.assertTrue(result["max_temp_c"] > 60.0)
        self.assertTrue(result["delta_t_c"] > 20.0)

    def test_03_ai_assistant_diagnostics(self):
        """Test AI assistant diagnosis for all 10 fault categories"""
        for f in FAULT_CLASSES:
            diag = AISolarFaultAssistant.analyze_fault(
                fault_type=f,
                confidence=0.95,
                panel_code="TEST-01",
                cell_row=2,
                cell_col=4,
                delta_t=25.0 if f == "Hot Spot" else 2.0,
                max_temp=75.0 if f == "Hot Spot" else 35.0
            )
            self.assertEqual(diag["fault_type"], f)
            self.assertIn("recommended_action", diag)
            self.assertIn("possible_causes", diag)
            if f in ["Hot Spot", "Physical Damage", "Electrical Anomaly"]:
                self.assertEqual(diag["severity"], "Critical")
                self.assertTrue(diag["immediate_inspection_required"])

    def test_04_deep_learning_models(self):
        """Test forward pass of DualModalPVNet and TransferPVNet"""
        device = torch.device("cpu")
        dual_model = DualModalPVNet(num_classes=len(FAULT_CLASSES)).to(device)
        transfer_model = TransferPVNet(num_classes=len(FAULT_CLASSES)).to(device)

        dummy_rgb = torch.randn(2, 3, 224, 224)
        dummy_thm = torch.randn(2, 3, 224, 224)

        dual_out = dual_model(dummy_rgb, dummy_thm)
        self.assertIn("logits", dual_out)
        self.assertEqual(dual_out["logits"].shape, (2, len(FAULT_CLASSES)))
        self.assertEqual(dual_out["row_logits"].shape, (2, PANEL_GRID_ROWS))
        self.assertEqual(dual_out["col_logits"].shape, (2, PANEL_GRID_COLS))
        self.assertEqual(dual_out["bbox"].shape, (2, 4))

        transfer_out = transfer_model(dummy_rgb)
        self.assertEqual(transfer_out.shape, (2, len(FAULT_CLASSES)))

    def test_05_explainable_ai_gradcam(self):
        """Test Grad-CAM computation and overlay generation"""
        device = torch.device("cpu")
        dual_model = DualModalPVNet(num_classes=len(FAULT_CLASSES)).to(device)
        grad_cam = GradCAM(dual_model, dual_model.visual_branch.res3)

        dummy_rgb = torch.randn(1, 3, 224, 224)
        dummy_thm = torch.randn(1, 3, 224, 224)

        cam_map = grad_cam.generate(dummy_rgb, dummy_thm, target_class_idx=1)
        self.assertEqual(cam_map.ndim, 2)
        self.assertTrue(np.max(cam_map) <= 1.0)
        self.assertTrue(np.min(cam_map) >= 0.0)

        # Test overlay
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        overlay = XAIVisualizer.overlay_heatmap(test_img, cam_map)
        self.assertEqual(overlay.shape, (224, 224, 3))

    def test_06_predictive_maintenance(self):
        """Test failure risk index and degradation calculation"""
        db = SessionLocal()
        panel = db.query(Panel).first()
        health = PredictiveMaintenanceEngine.assess_panel_health(panel.id)
        self.assertIn("health_index", health)
        self.assertIn("failure_risk_pct", health)
        self.assertTrue(0.0 <= health["health_index"] <= 100.0)
        self.assertTrue(0.0 <= health["failure_risk_pct"] <= 100.0)
        db.close()

    def test_07_maintenance_comparator(self):
        """Test before-and-after maintenance comparison"""
        img1 = np.full((100, 100, 3), 100, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 50, dtype=np.uint8)
        thm1 = np.full((100, 100, 3), 200, dtype=np.uint8)
        thm2 = np.full((100, 100, 3), 50, dtype=np.uint8)

        res = MaintenanceComparator.compare_inspections(img1, img2, thm1, thm2, pre_delta_t=25.0)
        self.assertIn("verdict", res)
        self.assertIn("clearance_percentage", res)
        self.assertTrue(res["clearance_percentage"] > 50.0)
        self.assertIsNotNone(res["mosaic_bgr"])

    def test_08_alert_system(self):
        """Test alert dispatch and acknowledgement"""
        dispatched = AlertSystem.dispatch_alert(
            panel_code="TEST-01",
            fault_type="Hot Spot",
            severity="Critical",
            confidence=0.96,
            cell_location="R2 C4",
            message="Test emergency alert"
        )
        self.assertTrue(len(dispatched) > 0)
        alerts = AlertSystem.get_recent_alerts(limit=5)
        self.assertTrue(len(alerts) > 0)
        
        # Acknowledge
        ack_ok = AlertSystem.acknowledge_alert(alerts[0]["id"])
        self.assertTrue(ack_ok)

    def test_09_report_generation(self):
        """Test PDF and CSV report creation"""
        csv_file = ReportGenerator.generate_csv_report()
        self.assertTrue(os.path.exists(csv_file))

        pdf_file = ReportGenerator.generate_pdf_report(farm_name="Test Solar Facility")
        self.assertTrue(os.path.exists(pdf_file))
        self.assertTrue(os.path.getsize(pdf_file) > 1000)

if __name__ == "__main__":
    unittest.main()
