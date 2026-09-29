"""
DJI Drone Thermal Camera Integration
=====================================
Connects the Solar Panel Fault Detection AI to a DJI drone thermal camera.

Supported pipelines:
  1. RTSP stream (DJI Zenmuse via DJI Pilot / DJI Enterprise SDK)
  2. Video file replay (.mp4 / .MOV recorded thermal footage)
  3. DJI SDK WebSocket relay (from DJI Mobile SDK / PC SDK)

Requirements:
  - DJI drone streaming RTSP video over WiFi/cable to PC
  - OR pre-recorded thermal video from DJI flight
  - pip install opencv-python torch

Usage:
  # RTSP stream from drone:
  python src/drone_thermal_integration.py --mode rtsp --url rtsp://192.168.2.1:8554/live

  # Local recorded thermal video file:
  python src/drone_thermal_integration.py --mode video --file path/to/thermal.mp4

  # Webcam simulation (for testing without drone):
  python src/drone_thermal_integration.py --mode webcam
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import WEIGHTS_DIR, FAULT_CLASSES, SEVERITY_MAPPING, STATIC_DIR
from src.models import DualModalPVNet, TransferPVNet

# ─── Severity colour palette (BGR) ───────────────────────────────────────────
SEVERITY_COLORS = {
    "Critical": (0,   0,   255),
    "High":     (0,   100, 255),
    "Medium":   (0,   200, 255),
    "Low":      (0,   255, 150),
    "Normal":   (0,   220, 0),
}

# ─── Frame-skip: run AI every N frames, track bounding box in between ─────────
INFERENCE_EVERY_N_FRAMES = 5

# ─── Model Setup ─────────────────────────────────────────────────────────────
def load_model(use_edge_model: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_edge_model:
        model = TransferPVNet(num_classes=len(FAULT_CLASSES)).to(device)
        weights = WEIGHTS_DIR / "transfer_pvnet.pth"
        model_name = "TransferPVNet (Edge)"
    else:
        model = DualModalPVNet(num_classes=len(FAULT_CLASSES)).to(device)
        weights = WEIGHTS_DIR / "dual_modal_pvnet.pth"
        model_name = "DualModalPVNet (Full)"

    if weights.exists():
        model.load_state_dict(torch.load(str(weights), map_location=device))
        print(f"[DRONE AI] Loaded {model_name} from {weights}")
    else:
        print(f"[DRONE AI] WARNING: No weights found. Using random init. Run train_model.py first!")
    model.eval()
    return model, device, model_name

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray, device) -> torch.Tensor:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    tensor  = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean    = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std     = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)

# ─── Simulate Thermal Colourmap (for RGB-only drone feeds) ───────────────────
def simulate_thermal(frame: np.ndarray) -> np.ndarray:
    """Apply FLIR Inferno colourmap to grayscale intensity → simulated thermal."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

# ─── Annotate Frame ──────────────────────────────────────────────────────────
def annotate_frame(frame: np.ndarray, fault: str, confidence: float,
                   severity: str, fps: float, frame_no: int) -> np.ndarray:
    h, w = frame.shape[:2]
    out  = frame.copy()

    color = SEVERITY_COLORS.get(severity, (255, 255, 255))

    # ── Top Banner ──────────────────────────────────────────────────────────
    cv2.rectangle(out, (0, 0), (w, 52), (15, 18, 28), -1)
    cv2.putText(out, "SOLAR PANEL FAULT DETECTION AI  |  DJI DRONE",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 210, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS: {fps:.1f}  |  Frame: {frame_no}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 160, 180), 1, cv2.LINE_AA)

    # ── Severity badge (bottom-left) ─────────────────────────────────────────
    badge_y = h - 60
    cv2.rectangle(out, (0, badge_y), (w, h), (15, 18, 28), -1)
    cv2.putText(out, f"FAULT: {fault.upper()}",
                (12, badge_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"SEVERITY: {severity.upper()}   CONFIDENCE: {confidence*100:.1f}%",
                (12, badge_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 240), 1, cv2.LINE_AA)

    # ── Severity border glow ─────────────────────────────────────────────────
    thickness = 4 if severity in ("Critical", "High") else 2
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)

    return out

# ─── Core Processing Loop ────────────────────────────────────────────────────
def run_stream(source, model, device, save_output: bool = True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open stream: {source}")
        return

    # Determine if source is RTSP (network) or file
    is_live = isinstance(source, str) and source.startswith("rtsp")

    # Output video writer
    out_writer = None
    if save_output:
        out_path = str(STATIC_DIR / "drone_thermal_output.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        fps_src  = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w_src    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_src, (w_src, h_src))
        print(f"[DRONE AI] Saving annotated output to: {out_path}")

    print(f"[DRONE AI] Stream open. Press 'q' to quit.\n")

    frame_no      = 0
    last_fault    = "Normal"
    last_conf     = 1.0
    last_severity = "Normal"
    t_prev        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if not is_live:
                print("[DRONE AI] Video file ended.")
            break

        frame_no += 1

        # ── Run AI every N frames (frame skipping for speed) ─────────────────
        if frame_no % INFERENCE_EVERY_N_FRAMES == 0:
            # Convert frame to simulated thermal for analysis
            thermal_frame = simulate_thermal(frame)
            tensor        = preprocess(frame, device)

            with torch.no_grad():
                if isinstance(model, TransferPVNet):
                    logits = model(tensor)
                else:
                    thm_t  = preprocess(thermal_frame, device)
                    out    = model(tensor, thm_t)
                    logits = out["logits"]

                probs     = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx  = int(np.argmax(probs))
                last_conf = float(probs[pred_idx])

            last_fault    = FAULT_CLASSES[pred_idx]
            last_severity = SEVERITY_MAPPING.get(last_fault, "Low")

        # ── Compute FPS ───────────────────────────────────────────────────────
        t_now    = time.time()
        fps      = 1.0 / max(0.001, t_now - t_prev)
        t_prev   = t_now

        # ── Annotate Frame ────────────────────────────────────────────────────
        annotated = annotate_frame(frame, last_fault, last_conf, last_severity, fps, frame_no)

        if out_writer:
            out_writer.write(annotated)

        # ── Save latest frame snapshot for dashboard preview ──────────────────
        snapshot_path = str(STATIC_DIR / "drone_latest_frame.jpg")
        cv2.imwrite(snapshot_path, annotated)

        # ── Console log every inference cycle ─────────────────────────────────
        if frame_no % INFERENCE_EVERY_N_FRAMES == 0:
            level = "[ALERT]" if last_severity in ("Critical", "High") else "[INFO] "
            print(f"  {level} Frame {frame_no:05d} | {last_fault:<20} | "
                  f"{last_severity:<8} | Conf: {last_conf*100:.1f}%  | FPS: {fps:.1f}")

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    print("[DRONE AI] Stream closed.")

# ─── Entry Point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DJI Drone Thermal Camera AI Integration")
    parser.add_argument("--mode",  choices=["rtsp", "video", "webcam"], default="webcam",
                        help="Stream source mode")
    parser.add_argument("--url",   default="rtsp://192.168.2.1:8554/live",
                        help="RTSP stream URL (for --mode rtsp)")
    parser.add_argument("--file",  default="",
                        help="Path to recorded thermal video (for --mode video)")
    parser.add_argument("--edge",  action="store_true", default=True,
                        help="Use lightweight edge model (TransferPVNet) for speed")
    parser.add_argument("--save",  action="store_true", default=True,
                        help="Save annotated output video to static/")
    args = parser.parse_args()

    print("=" * 62)
    print("  DJI DRONE THERMAL CAMERA  |  SOLAR PANEL FAULT DETECTION")
    print("=" * 62)

    model, device, model_name = load_model(use_edge_model=args.edge)
    print(f"  Model    : {model_name}")
    print(f"  Device   : {device}")
    print(f"  Mode     : {args.mode.upper()}")

    if args.mode == "rtsp":
        # DJI Zenmuse X-series / Enterprise SDK RTSP stream
        print(f"  RTSP URL : {args.url}")
        print("  TIP: On DJI Pilot 2, enable 'Video Stream' in settings.")
        print("       Ensure your PC is on the same WiFi as the drone/controller.")
        print("=" * 62)
        run_stream(args.url, model, device, save_output=args.save)

    elif args.mode == "video":
        if not args.file or not os.path.exists(args.file):
            print(f"[ERROR] Video file not found: '{args.file}'")
            print("  Usage: python src/drone_thermal_integration.py --mode video --file path/to/thermal.mp4")
            return
        print(f"  File     : {args.file}")
        print("=" * 62)
        run_stream(args.file, model, device, save_output=args.save)

    elif args.mode == "webcam":
        print("  Source   : Default Webcam (index 0) — simulation mode")
        print("  NOTE: Webcam simulates drone camera for testing purposes.")
        print("=" * 62)
        run_stream(0, model, device, save_output=args.save)

if __name__ == "__main__":
    main()
