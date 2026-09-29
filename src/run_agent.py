"""
Solar Panel AI Agent - Run Inference on Synthetic and Real Images
Demonstrates the full end-to-end pipeline:
  1. Generate synthetic panel images
  2. Inject faults
  3. Run DualModalPVNet inference
  4. Display results with explainability
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from config import WEIGHTS_DIR, FAULT_CLASSES
from src.inference_engine import SolarFaultInferenceEngine
from src.dataset_generator import create_base_solar_panel, inject_fault

def run_inference_on_fault(engine, fault_class, panel_code):
    print(f"\n[AGENT] Testing fault: '{fault_class}' on panel '{panel_code}'")
    print("-" * 60)

    # Generate synthetic PV images
    rgb, thermal, cells = create_base_solar_panel()
    fault_data = inject_fault(rgb, thermal, cells, fault_class)
    rgb_img   = fault_data["rgb"]
    thm_img   = fault_data["thermal"]

    # Run full AI inspection
    result = engine.run_full_inspection(rgb_img, thm_img, panel_code=panel_code)

    # Print structured results
    print(f"  Fault Detected     : {result['fault_type']}")
    print(f"  Confidence         : {result['confidence']*100:.1f}%")
    print(f"  Severity           : {result['severity']}")
    print(f"  Cell Location      : Row {result['cell_row']}  Col {result['cell_col']}")
    print(f"  Ambient Temp       : {result['ambient_temp_c']} C")
    print(f"  Max Temp           : {result['max_temp_c']} C")
    print(f"  Delta-T            : {result['delta_t_c']} C")
    print(f"  Primary Cause      : {result['assistant']['primary_cause']}")
    print(f"  Recommended Action : {result['assistant']['recommended_action']}")
    print(f"  Inspection Needed  : {result['assistant']['immediate_inspection_required']}")
    return result

def main():
    weights_path = str(WEIGHTS_DIR / "dual_modal_pvnet.pth")
    print("=" * 60)
    print("  SOLAR PANEL FAULT DETECTION AI AGENT")
    print("=" * 60)
    print(f"  Model Weights: {weights_path}")
    print(f"  Fault Classes: {len(FAULT_CLASSES)}")
    print("=" * 60)

    # Initialize engine
    engine = SolarFaultInferenceEngine(weights_path=weights_path)

    # Run inference for a representative subset of fault classes
    test_faults = [
        ("Hot Spot",         "PNL-A1-01"),
        ("Micro Crack",      "PNL-A1-02"),
        ("Dust & Soiling",   "PNL-B2-05"),
        ("Physical Damage",  "PNL-C3-11"),
        ("Electrical Anomaly", "PNL-D4-07"),
        ("Normal",           "PNL-E1-03"),
    ]

    results = []
    for fault_class, panel_code in test_faults:
        result = run_inference_on_fault(engine, fault_class, panel_code)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("  AGENT SUMMARY")
    print("=" * 60)
    critical = [r for r in results if r["severity"] in ["Critical", "High"]]
    print(f"  Total Panels Inspected : {len(results)}")
    print(f"  Critical/High Alerts   : {len(critical)}")
    for r in critical:
        print(f"    >> {r['panel_code']} | {r['fault_type']} | {r['severity']} | {r['confidence']*100:.1f}%")
    print("=" * 60)
    print("  Agent run complete. Inspection records saved to DB.")

if __name__ == "__main__":
    main()
