"""
Explainable AI (XAI) & Localization Visualization Engine for Photovoltaic Inspection
Implements:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Attention & Heatmap generation
- 6x10 PV Cell Grid Coordinate Overlay
- Multi-Panel Visual Diagnostics (RGB, Thermal, Grad-CAM, Grid Localization)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from config import PANEL_GRID_ROWS, PANEL_GRID_COLS

class GradCAM:
    """
    Grad-CAM engine for PyTorch models.
    Generates class-discriminative localization maps.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        rgb_tensor: torch.Tensor,
        thermal_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for the given inputs and target class.
        Returns a normalized 2D numpy array [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(rgb_tensor, thermal_tensor)
        logits = output["logits"]

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()

        # Backward pass for the target class
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        # Calculate channel weights by global average pooling the gradients
        # gradients shape: [batch, channels, H, W]
        gradients = self.gradients[0] # [channels, H, W]
        activations = self.activations[0] # [channels, H, W]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # [channels, 1, 1]
        
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * activations, dim=0) # [H, W]
        cam = F.relu(cam) # keep only positive influences

        # Normalize to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_max = np.max(cam_np)
        if cam_max > 0:
            cam_np = cam_np / cam_max
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

class XAIVisualizer:
    """Creates clear diagnostic overlays for technicians and dashboard views"""

    @staticmethod
    def overlay_heatmap(
        image_bgr: np.ndarray,
        heatmap_2d: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Overlays a 2D heatmap [0, 1] on a BGR image"""
        h, w = image_bgr.shape[:2]
        resized_heatmap = cv2.resize(heatmap_2d, (w, h))
        heatmap_uint8 = np.uint8(255 * resized_heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)
        return blended

    @staticmethod
    def draw_grid_localization(
        image_bgr: np.ndarray,
        cell_row: Optional[int],
        cell_col: Optional[int],
        fault_name: str,
        confidence: float,
        bbox: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Draws the 6x10 PV cell grid and highlights the exact affected cell
        with high-contrast bounding box and technical label.
        """
        output = image_bgr.copy()
        h, w = output.shape[:2]
        
        # Frame boundary
        frame_margin = 16
        inner_w = w - 2 * frame_margin
        inner_h = h - 2 * frame_margin

        cell_w = inner_w / PANEL_GRID_COLS
        cell_h = inner_h / PANEL_GRID_ROWS

        # Draw subtle grid lines
        grid_overlay = output.copy()
        for r in range(PANEL_GRID_ROWS + 1):
            y = int(frame_margin + r * cell_h)
            cv2.line(grid_overlay, (frame_margin, y), (w - frame_margin, y), (80, 120, 160), 1)

        for c in range(PANEL_GRID_COLS + 1):
            x = int(frame_margin + c * cell_w)
            cv2.line(grid_overlay, (x, frame_margin), (x, h - frame_margin), (80, 120, 160), 1)

        cv2.addWeighted(grid_overlay, 0.4, output, 0.6, 0, output)

        # Highlight faulty cell if present
        if cell_row is not None and cell_col is not None and fault_name != "Normal":
            r_idx = cell_row - 1
            c_idx = cell_col - 1
            cx1 = int(frame_margin + c_idx * cell_w)
            cy1 = int(frame_margin + r_idx * cell_h)
            cx2 = int(frame_margin + (c_idx + 1) * cell_w)
            cy2 = int(frame_margin + (r_idx + 1) * cell_h)

            # Pulsing high-contrast highlight
            highlight_color = (0, 140, 255) # Orange-amber BGR
            if fault_name in ["Hot Spot", "Physical Damage", "Electrical Anomaly"]:
                highlight_color = (0, 0, 235) # Bright Red BGR

            # Draw transparent colored cell fill
            fill_overlay = output.copy()
            cv2.rectangle(fill_overlay, (cx1, cy1), (cx2, cy2), highlight_color, -1)
            cv2.addWeighted(fill_overlay, 0.35, output, 0.65, 0, output)

            # Draw cell border
            cv2.rectangle(output, (cx1, cy1), (cx2, cy2), highlight_color, 2)

            # Draw targeted BBox if provided
            if bbox and bbox != [0, 0, 0, 0]:
                bx1, by1, bx2, by2 = bbox
                cv2.rectangle(output, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

            # Technical Label Banner
            label_text = f"R{cell_row}C{cell_col} | {fault_name} ({confidence*100:.1f}%)"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            # Place banner smartly
            by = max(cy1 - 6, th + 6)
            bx = min(max(cx1, 5), w - tw - 10)
            cv2.rectangle(output, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (20, 20, 25), -1)
            cv2.rectangle(output, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), highlight_color, 1)
            cv2.putText(output, label_text, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return output

    @staticmethod
    def generate_multimodal_quad_view(
        rgb_img: np.ndarray,
        thermal_img: np.ndarray,
        heatmap_2d: np.ndarray,
        cell_row: Optional[int],
        cell_col: Optional[int],
        fault_name: str,
        confidence: float,
        bbox: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Combines 4 inspection views into a unified high-resolution diagnostic quad-panel:
        Top-Left: Optical RGB Image
        Top-Right: Infrared Radiometric Thermal Image
        Bottom-Left: Explainable AI Grad-CAM Heatmap Overlay
        Bottom-Right: 6x10 Cell Grid Localization & Bounding Box Map
        """
        h, w = rgb_img.shape[:2]
        
        # Panel 1: RGB
        p1 = rgb_img.copy()
        cv2.putText(p1, "1. OPTICAL (RGB) INSPECTION", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Panel 2: Thermal
        p2 = thermal_img.copy()
        cv2.putText(p2, "2. THERMAL INFRARED (IR) MAP", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Panel 3: Grad-CAM XAI
        p3 = XAIVisualizer.overlay_heatmap(p1.copy(), heatmap_2d, alpha=0.55, colormap=cv2.COLORMAP_JET)
        cv2.putText(p3, "3. EXPLAINABLE AI (GRAD-CAM)", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Panel 4: Cell Grid Localization
        p4 = XAIVisualizer.draw_grid_localization(rgb_img, cell_row, cell_col, fault_name, confidence, bbox)
        cv2.putText(p4, "4. 6x10 CELL GRID LOCALIZATION", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Stitch into 2x2 grid
        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([p3, p4])
        quad_view = np.vstack([top_row, bottom_row])

        return quad_view
