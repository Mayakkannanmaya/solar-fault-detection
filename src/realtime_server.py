import os
import sys
import base64
import cv2
import numpy as np
import torch
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Add project root to sys.path so config can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import TransferPVNet
from config import WEIGHTS_DIR, FAULT_CLASSES

app = FastAPI(title="Real-Time PV Inspection AI")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Lightweight Edge Model for Real-Time processing
model = TransferPVNet(num_classes=len(FAULT_CLASSES)).to(device)
weights_path = WEIGHTS_DIR / "transfer_pvnet.pth"
if weights_path.exists():
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
model.eval()

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)

@app.websocket("/ws/video-stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video stream ingestion.
    Client sends base64 encoded frames, server returns inference results.
    """
    await websocket.accept()
    print("Client connected to real-time video stream.")
    try:
        while True:
            # Receive frame as base64 string
            data = await websocket.receive_text()
            
            # Decode base64 to numpy array image
            img_data = base64.b64decode(data)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # Run fast inference
            tensor = preprocess_frame(frame)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
            
            fault_type = FAULT_CLASSES[pred_idx]
            
            # Send results back
            result = {
                "fault": fault_type,
                "confidence": round(confidence, 3),
                "fps_optimized": True
            }
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error in stream: {e}")

if __name__ == "__main__":
    print("Starting Real-Time Fast API Server on port 8000...")
    uvicorn.run("src.realtime_server:app", host="0.0.0.0", port=8000, reload=True)
