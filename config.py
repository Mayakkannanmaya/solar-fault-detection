"""
Project Directory Structure & Configuration for
AI-Powered Solar Panel Fault Detection, Localization and Predictive Maintenance System
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"
TESTS_DIR = BASE_DIR / "tests"

# Sub-directories
DATASET_RGB = DATA_DIR / "rgb"
DATASET_THERMAL = DATA_DIR / "thermal"
DATASET_PAIRS = DATA_DIR / "pairs"
WEIGHTS_DIR = MODELS_DIR / "weights"
DB_PATH = BASE_DIR / "solar_monitoring.db"

# Create directories if not exist
for p in [DATA_DIR, DATASET_RGB, DATASET_THERMAL, DATASET_PAIRS, MODELS_DIR, WEIGHTS_DIR, SRC_DIR, REPORTS_DIR, STATIC_DIR, TESTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Fault Categories
FAULT_CLASSES = [
    "Normal",
    "Hot Spot",
    "Micro Crack",
    "Dust & Soiling",
    "Bird Dropping",
    "Physical Damage",
    "Discoloration",
    "Cell Defect",
    "Partial Shading",
    "Electrical Anomaly"
]

# Severity Thresholds and Mapping
SEVERITY_LEVELS = ["Normal", "Low", "Medium", "High", "Critical"]

SEVERITY_MAPPING = {
    "Normal": "Normal",
    "Dust & Soiling": "Low",
    "Discoloration": "Medium",
    "Partial Shading": "Medium",
    "Bird Dropping": "Medium",
    "Cell Defect": "High",
    "Micro Crack": "High",
    "Electrical Anomaly": "High",
    "Hot Spot": "Critical",
    "Physical Damage": "Critical",
}

# Grid Configuration (Standard 60-cell PV panel: 6 rows x 10 columns)
PANEL_GRID_ROWS = 6
PANEL_GRID_COLS = 10

# API & Server Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
STREAMLIT_PORT = 8501
