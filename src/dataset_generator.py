"""
Photovoltaic Dual-Modal Dataset Generator
Generates high-fidelity paired RGB (visual) and Thermal (infrared/radiometric) solar panel images
with realistic fault manifestations, cell-level grid coordinates, and thermal metrics.
"""

import os
import sys
# Add project root to sys.path so config can be imported when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from typing import Tuple, Dict, Any, List
from config import (
    DATASET_RGB, DATASET_THERMAL, DATASET_PAIRS,
    FAULT_CLASSES, PANEL_GRID_ROWS, PANEL_GRID_COLS
)

def create_base_solar_panel(width: int = 600, height: int = 360) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    """
    Creates a photorealistic baseline 60-cell PV module in RGB and Thermal.
    Grid: 6 rows x 10 columns = 60 cells.
    Returns: (rgb_img, thermal_img, cell_coords_list)
    """
    # Create canvas
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # Monocrystalline deep blue / dark slate
    rgb[:] = [15, 25, 45] # BGR
    
    # Outer silver/aluminum frame
    frame_thickness = 14
    cv2.rectangle(rgb, (0, 0), (width-1, height-1), (180, 185, 190), thickness=frame_thickness)
    cv2.rectangle(rgb, (frame_thickness, frame_thickness), 
                  (width-frame_thickness, height-frame_thickness), (60, 65, 70), thickness=2)

    # Active solar panel area
    inner_x1 = frame_thickness + 4
    inner_y1 = frame_thickness + 4
    inner_x2 = width - frame_thickness - 4
    inner_y2 = height - frame_thickness - 4

    cell_w = (inner_x2 - inner_x1) / PANEL_GRID_COLS
    cell_h = (inner_y2 - inner_y1) / PANEL_GRID_ROWS

    cell_bboxes = []

    # Draw individual solar cells with anti-reflective coating texture and busbars
    for r in range(PANEL_GRID_ROWS):
        for c in range(PANEL_GRID_COLS):
            cx1 = int(inner_x1 + c * cell_w)
            cy1 = int(inner_y1 + r * cell_h)
            cx2 = int(inner_x1 + (c + 1) * cell_w)
            cy2 = int(inner_y1 + (r + 1) * cell_h)

            cell_bboxes.append({
                "row": r + 1,
                "col": c + 1,
                "x1": cx1 + 2,
                "y1": cy1 + 2,
                "x2": cx2 - 2,
                "y2": cy2 - 2,
                "center_x": int((cx1 + cx2) / 2),
                "center_y": int((cy1 + cy2) / 2)
            })

            # Cell wafer background with subtle monocrystalline wafer diamond chamfers
            cell_color = (
                int(50 + random.randint(-4, 4)), 
                int(38 + random.randint(-4, 4)), 
                int(22 + random.randint(-3, 3))
            ) # Dark Blue-Navy in BGR
            cv2.rectangle(rgb, (cx1 + 2, cy1 + 2), (cx2 - 2, cy2 - 2), cell_color, -1)

            # Busbars (3 vertical silver lines per cell)
            bus_w = max(1, int(cell_w / 4))
            for b in [1, 2, 3]:
                bx = int(cx1 + b * bus_w)
                cv2.line(rgb, (bx, cy1 + 2), (bx, cy2 - 2), (180, 190, 200), 1)

            # Cell isolation boundary gap
            cv2.rectangle(rgb, (cx1, cy1), (cx2, cy2), (8, 12, 20), 1)

    # Base Thermal Map (Uniform normal operating cell temperature ~38-42°C)
    # Thermal normalized: 0 = 20°C, 255 = 90°C
    # 40°C corresponds to approx pixel value 75
    thermal_raw = np.full((height, width), 72, dtype=np.uint8)
    # Add slight natural thermal gradient across panel
    y_grad = np.linspace(0, 10, height, dtype=np.float32)[:, None]
    y_grad_tile = np.repeat(y_grad, width, axis=1)
    thermal_raw = np.clip(thermal_raw.astype(np.float32) + y_grad_tile, 0, 255).astype(np.uint8)
    
    # Cooler aluminum frame
    thermal_raw[:frame_thickness, :] = 35
    thermal_raw[-frame_thickness:, :] = 35
    thermal_raw[:, :frame_thickness] = 35
    thermal_raw[:, -frame_thickness:] = 35

    # Convert to FLIR Ironbow / Inferno Colormap
    thermal_bgr = cv2.applyColorMap(thermal_raw, cv2.COLORMAP_INFERNO)

    return rgb, thermal_bgr, cell_bboxes

def inject_fault(
    rgb: np.ndarray,
    thermal: np.ndarray,
    cell_bboxes: List[Dict[str, int]],
    fault_type: str
) -> Dict[str, Any]:
    """
    Injects realistic visual and radiometric thermal signatures for a specified fault.
    Returns metadata including location, bounding box, temperatures, and descriptions.
    """
    # Convert thermal back to grayscale intensity for radiometric manipulation
    thermal_gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    h, w, _ = rgb.shape

    # Choose a target cell or multi-cell area
    target_cell = random.choice(cell_bboxes)
    r = target_cell["row"]
    c = target_cell["col"]
    cx1, cy1, cx2, cy2 = target_cell["x1"], target_cell["y1"], target_cell["x2"], target_cell["y2"]
    cw = cx2 - cx1
    ch = cy2 - cy1
    center_x, center_y = target_cell["center_x"], target_cell["center_y"]

    ambient_temp = 28.0
    max_temp = 41.5
    bbox = [cx1, cy1, cx2, cy2]

    if fault_type == "Normal":
        # Keep clean, baseline NOCT temperature
        max_temp = round(38.0 + random.uniform(0.5, 3.5), 1)
        delta_t = round(max_temp - ambient_temp, 1)
        thermal_colored = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_INFERNO)
        return {
            "fault_type": "Normal",
            "cell_row": None,
            "cell_col": None,
            "bbox": [0, 0, 0, 0],
            "ambient_temp_c": ambient_temp,
            "max_temp_c": max_temp,
            "delta_t_c": delta_t,
            "rgb": rgb,
            "thermal": thermal_colored
        }

    elif fault_type == "Hot Spot":
        # Intense thermal spike: 68°C - 88°C (pixel values 190 - 250)
        max_temp = round(68.0 + random.uniform(5.0, 20.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)
        
        # Thermal: High intensity Gaussian heat bloom
        heat_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(heat_mask, (center_x, center_y), int(cw * 0.45), 1.0, -1)
        heat_mask = cv2.GaussianBlur(heat_mask, (31, 31), 11)
        
        thermal_boost = (heat_mask * 175).astype(np.uint8)
        thermal_gray = cv2.add(thermal_gray, thermal_boost)

        # Visual (RGB): localized discoloration / burn mark / hotspot ring
        cv2.circle(rgb, (center_x, center_y), int(cw * 0.25), (30, 70, 110), -1)
        cv2.circle(rgb, (center_x, center_y), int(cw * 0.15), (20, 40, 70), -1)
        # Small fissure
        cv2.line(rgb, (center_x - 5, center_y - 8), (center_x + 6, center_y + 7), (120, 140, 160), 1)

    elif fault_type == "Micro Crack":
        # Fractured silicon cell with branching crack
        max_temp = round(48.0 + random.uniform(3.0, 10.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)

        # RGB: Branching light-gray/silver crack
        pt1 = (cx1 + int(cw * 0.15), cy1 + int(ch * 0.2))
        pt2 = (center_x + random.randint(-4, 4), center_y + random.randint(-4, 4))
        pt3 = (cx2 - int(cw * 0.2), cy2 - int(ch * 0.25))
        branch = (center_x + random.randint(5, 12), cy1 + int(ch * 0.7))

        cv2.line(rgb, pt1, pt2, (200, 215, 230), 2)
        cv2.line(rgb, pt2, pt3, (180, 200, 220), 1)
        cv2.line(rgb, pt2, branch, (170, 190, 210), 1)

        # Thermal: Elevated warmth along crack line
        crack_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(crack_mask, pt1, pt3, 80, 5)
        crack_mask = cv2.GaussianBlur(crack_mask, (15, 15), 5)
        thermal_gray = cv2.add(thermal_gray, crack_mask)

    elif fault_type == "Dust & Soiling":
        # Widespread or cell-cluster particulate accumulation
        max_temp = round(43.0 + random.uniform(2.0, 5.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)

        # Multiple dusty cells
        soiling_cols = random.sample(range(1, PANEL_GRID_COLS + 1), k=random.randint(3, 6))
        for cell in cell_bboxes:
            if cell["col"] in soiling_cols:
                # Add sandy / dusty tint (BGR: tan/ochre)
                patch = rgb[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]]
                noise = np.random.randint(0, 40, patch.shape, dtype=np.uint8)
                dust_overlay = np.full(patch.shape, (65, 95, 125), dtype=np.uint8)
                blended = cv2.addWeighted(patch, 0.65, dust_overlay, 0.35, 0)
                blended = cv2.add(blended, noise)
                rgb[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]] = blended

                # Mild thermal insulation
                patch_thm = thermal_gray[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]]
                thermal_gray[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]] = np.clip(patch_thm.astype(np.int16) + 22, 0, 255).astype(np.uint8)
        bbox = [cx1, cy1, cx2, cy2]

    elif fault_type == "Bird Dropping":
        # White / chalky organic splatter blocking cell
        max_temp = round(52.0 + random.uniform(4.0, 12.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)

        # RGB Splatter
        drop_color = (225, 235, 240)
        cv2.ellipse(rgb, (center_x, center_y), (int(cw*0.25), int(ch*0.2)), 30, 0, 360, drop_color, -1)
        for _ in range(6):
            sx = center_x + random.randint(-int(cw*0.35), int(cw*0.35))
            sy = center_y + random.randint(-int(ch*0.3), int(ch*0.3))
            sr = random.randint(2, 5)
            cv2.circle(rgb, (sx, sy), sr, (215, 225, 230), -1)

        # Thermal: Hot sub-cell region caused by optical blockage reverse bias
        t_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(t_mask, (center_x, center_y), int(cw * 0.35), 90, -1)
        t_mask = cv2.GaussianBlur(t_mask, (21, 21), 7)
        thermal_gray = cv2.add(thermal_gray, t_mask)

    elif fault_type == "Physical Damage":
        # Shattered glass, broken wafer, missing shards
        max_temp = round(64.0 + random.uniform(5.0, 15.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)

        # RGB: Shattered spiderweb cracks + dark fractured crater
        cv2.circle(rgb, (center_x, center_y), int(cw * 0.28), (15, 20, 25), -1)
        for angle in range(0, 360, 35):
            rad = math.radians(angle)
            ex = int(center_x + math.cos(rad) * cw * 0.45)
            ey = int(center_y + math.sin(rad) * ch * 0.45)
            cv2.line(rgb, (center_x, center_y), (ex, ey), (190, 210, 225), 2)
            # Web arcs
            cv2.circle(rgb, (center_x, center_y), int(cw * 0.2), (180, 200, 220), 1)

        # Thermal: High localized thermal gradient with cold fractured boundary
        dmg_heat = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(dmg_heat, (center_x, center_y), int(cw * 0.4), 130, -1)
        dmg_heat = cv2.GaussianBlur(dmg_heat, (25, 25), 9)
        thermal_gray = cv2.add(thermal_gray, dmg_heat)

    elif fault_type == "Discoloration":
        # Snail trails or EVA browning
        max_temp = round(46.0 + random.uniform(2.0, 7.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)

        # RGB: Brownish EVA discoloration and snail trails
        eva_patch = rgb[cy1:cy2, cx1:cx2]
        brown_tint = np.full(eva_patch.shape, (25, 60, 95), dtype=np.uint8)
        rgb[cy1:cy2, cx1:cx2] = cv2.addWeighted(eva_patch, 0.6, brown_tint, 0.4, 0)
        # Snail trail squiggles
        cv2.polylines(rgb, [np.array([[cx1+5, cy1+10], [cx1+18, cy1+15], [cx1+25, cy2-12], [cx2-8, cy2-5]], np.int32)], False, (140, 160, 175), 2)

        # Thermal: mild heat accumulation
        thm_sub = thermal_gray[cy1:cy2, cx1:cx2]
        thermal_gray[cy1:cy2, cx1:cx2] = np.clip(thm_sub.astype(np.int16) + 30, 0, 255).astype(np.uint8)

    elif fault_type == "Cell Defect":
        # Dead or shunted wafer sub-region
        max_temp = round(56.0 + random.uniform(4.0, 12.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)

        # RGB: Inactive dark rectangular block or crystallographic impurity
        cv2.rectangle(rgb, (cx1 + 4, cy1 + 4), (cx2 - 4, cy2 - 4), (10, 15, 22), -1)

        # Thermal: Uniformly hot defective cell
        thm_sub2 = thermal_gray[cy1:cy2, cx1:cx2]
        thermal_gray[cy1:cy2, cx1:cx2] = np.clip(thm_sub2.astype(np.int16) + 105, 0, 255).astype(np.uint8)

    elif fault_type == "Partial Shading":
        # Diagonal / horizontal shadow band across 2-3 adjacent cells
        max_temp = round(47.0 + random.uniform(2.0, 6.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)

        # RGB: Shaded polygon
        shadow_pts = np.array([
            [cx1 - 20, cy1 - 10], [cx2 + 40, cy1 - 10],
            [cx2 + 20, cy2 + 20], [cx1 - 40, cy2 + 20]
        ], np.int32)
        shadow_overlay = rgb.copy()
        cv2.fillPoly(shadow_overlay, [shadow_pts], (5, 8, 12))
        rgb = cv2.addWeighted(rgb, 0.5, shadow_overlay, 0.5, 0)

        # Thermal: Bypass diode activation warmth
        t_shade = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(t_shade, [shadow_pts], 45)
        thermal_gray = cv2.add(thermal_gray, t_shade)

    elif fault_type == "Electrical Anomaly":
        # Junction box thermal runaway or ribbon disconnection at panel edge
        max_temp = round(72.0 + random.uniform(6.0, 18.0), 1)
        delta_t = round(max_temp - ambient_temp, 1)

        # RGB: scorched yellow/brown ribbon joint
        cv2.rectangle(rgb, (cx1 + 5, cy1 + 2), (cx2 - 5, cy1 + 12), (20, 50, 90), -1)
        cv2.circle(rgb, (center_x, cy1 + 7), 5, (10, 30, 50), -1)

        # Thermal: High intensity heat source at cell terminal / busbar
        elec_heat = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(elec_heat, (center_x, cy1 + 7), int(cw * 0.45), 160, -1)
        elec_heat = cv2.GaussianBlur(elec_heat, (25, 25), 9)
        thermal_gray = cv2.add(thermal_gray, elec_heat)

    # Render thermal image with Ironbow colormap
    thermal_colored = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_INFERNO)

    return {
        "fault_type": fault_type,
        "cell_row": r,
        "cell_col": c,
        "bbox": bbox,
        "ambient_temp_c": ambient_temp,
        "max_temp_c": max_temp,
        "delta_t_c": delta_t,
        "rgb": rgb,
        "thermal": thermal_colored
    }

def generate_sample_dataset(count_per_class: int = 4) -> List[Dict[str, Any]]:
    """
    Generates a curated dataset of high-quality paired RGB and Thermal PV images
    across all 10 fault categories. Saves images to disk and returns metadata records.
    """
    dataset_records = []
    sample_id = 1

    for fault_class in FAULT_CLASSES:
        for i in range(count_per_class):
            rgb_base, thermal_base, cells = create_base_solar_panel()
            result = inject_fault(rgb_base, thermal_base, cells, fault_class)
            
            clean_name = fault_class.lower().replace(" ", "_").replace("&", "and")
            filename_stem = f"pv_{clean_name}_{sample_id:04d}"
            
            rgb_path = DATASET_RGB / f"{filename_stem}_rgb.jpg"
            thermal_path = DATASET_THERMAL / f"{filename_stem}_thermal.jpg"
            
            # Save images
            cv2.imwrite(str(rgb_path), result["rgb"])
            cv2.imwrite(str(thermal_path), result["thermal"])
            
            record = {
                "sample_id": sample_id,
                "fault_type": fault_class,
                "cell_row": result["cell_row"],
                "cell_col": result["cell_col"],
                "bbox": result["bbox"],
                "ambient_temp_c": result["ambient_temp_c"],
                "max_temp_c": result["max_temp_c"],
                "delta_t_c": result["delta_t_c"],
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path),
            }
            dataset_records.append(record)
            sample_id += 1

    return dataset_records

if __name__ == "__main__":
    print(f"Generating synthetic PV benchmark dataset across {len(FAULT_CLASSES)} fault classes...")
    records = generate_sample_dataset(count_per_class=4)
    print(f"Generated {len(records)} paired samples successfully!")
