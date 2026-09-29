import os
import sys
# Add project root to sys.path so config can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.onnx
from torch.ao.quantization import quantize_dynamic
from src.models import DualModalPVNet, TransferPVNet
from config import WEIGHTS_DIR, FAULT_CLASSES

def export_and_quantize():
    device = torch.device("cpu") # ONNX/Quantization is usually done on CPU first
    
    # Paths
    dual_weights_path = WEIGHTS_DIR / "dual_modal_pvnet.pth"
    transfer_weights_path = WEIGHTS_DIR / "transfer_pvnet.pth"
    
    dual_onnx_path = WEIGHTS_DIR / "dual_modal_pvnet.onnx"
    transfer_onnx_path = WEIGHTS_DIR / "transfer_pvnet.onnx"
    
    transfer_int8_path = WEIGHTS_DIR / "transfer_pvnet_int8.pth"

    print("--- Real-Time Optimization: ONNX & INT8 Quantization ---")
    
    # 1. Load TransferPVNet (Edge Model)
    transfer_model = TransferPVNet(num_classes=len(FAULT_CLASSES)).to(device)
    if transfer_weights_path.exists():
        transfer_model.load_state_dict(torch.load(str(transfer_weights_path), map_location=device))
        print(f"Loaded TransferPVNet weights from {transfer_weights_path}")
    transfer_model.eval()
    
    # Example input for tracing (Batch Size: 1, Channels: 3, H: 224, W: 224)
    dummy_input_rgb = torch.randn(1, 3, 224, 224).to(device)
    
    # 2. Export TransferPVNet to ONNX
    print(f"Exporting TransferPVNet to ONNX -> {dual_onnx_path}")
    torch.onnx.export(
        transfer_model, 
        dummy_input_rgb, 
        str(transfer_onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['rgb_input'],
        output_names=['logits'],
        dynamic_axes={'rgb_input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )
    print("TransferPVNet ONNX Export Successful!")

    # 3. INT8 Dynamic Quantization for TransferPVNet
    print("Applying INT8 Dynamic Quantization to TransferPVNet...")
    # Dynamic quantization applies to Linear layers. For CNNs, static quantization is better, 
    # but dynamic provides a quick size reduction for FC layers.
    quantized_transfer_model = quantize_dynamic(
        transfer_model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    torch.save(quantized_transfer_model.state_dict(), str(transfer_int8_path))
    print(f"INT8 Quantized Model saved to {transfer_int8_path}")

    # 4. Load DualModalPVNet
    dual_model = DualModalPVNet(num_classes=len(FAULT_CLASSES)).to(device)
    if dual_weights_path.exists():
        dual_model.load_state_dict(torch.load(str(dual_weights_path), map_location=device))
        print(f"Loaded DualModalPVNet weights from {dual_weights_path}")
    dual_model.eval()
    
    dummy_input_thm = torch.randn(1, 3, 224, 224).to(device)
    
    # 5. Export DualModalPVNet to ONNX
    print(f"Exporting DualModalPVNet to ONNX -> {dual_onnx_path}")
    torch.onnx.export(
        dual_model, 
        (dummy_input_rgb, dummy_input_thm), 
        str(dual_onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['rgb_input', 'thermal_input'],
        output_names=['logits', 'row_logits', 'col_logits', 'bbox', 'delta_t'],
        dynamic_axes={
            'rgb_input': {0: 'batch_size'}, 
            'thermal_input': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print("DualModalPVNet ONNX Export Successful!")

if __name__ == "__main__":
    export_and_quantize()
