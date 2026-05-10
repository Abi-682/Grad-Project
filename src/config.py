"""
Configuration file for Manufacturing Defect Detection CNN Project
Contains all hyperparameters, paths, and crack shape definitions
"""

import os
import torch
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CRACK SHAPES & K1 VALUES (From Thesis)
# ============================================================================
CRACK_SHAPES = {
    0: "C1",
    1: "C2",
    2: "C3",
    3: "C4",
    4: "C5",
    5: "C6",
    6: "C7",
    7: "C8",
    8: "C9",
    9: "C10",
    10: "C11",
    11: "C12",
    12: "C13",
    13: "C14"
}

# SIF Maximum Values for each crack shape (in N/mm^1.5 or appropriate units)
K1_VALUES = {
    "C1": 552.45,
    "C2": 3917.0,
    "C3": 6839.9,
    "C4": 8810.3,
    "C5": 10312.0,
    "C6": 10979.0,
    "C7": 12690.0,
    "C8": 13132.0,
    "C9": 13497.0,
    "C10": 13782.0,
    "C11": 13848.0,
    "C12": 15205.0,
    "C13": 16289.0,
    "C14": 17516.0
}

# Reverse mapping: index to K1 value
IDX_TO_K1 = {i: K1_VALUES[CRACK_SHAPES[i]] for i in range(14)}

# Class names for binary detection (Stage 1)
CLASS_NAMES = {
    0: "No Defect",
    1: "Defect"
}

NUM_CRACK_SHAPES = len(CRACK_SHAPES)
NUM_CLASSES_BINARY = 2

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Image preprocessing
IMAGE_SIZE = 224  # ResNet expects 224x224
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "brightness_range": [0.8, 1.2],
    "zoom_range": [0.9, 1.1],
    "horizontal_flip": False,
    "gaussian_noise_std": 0.01
}

# ============================================================================
# TRAINING CONFIG
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect GPU
NUM_WORKERS = 4
PIN_MEMORY = True
GRADIENT_CLIP = 1.0

# Model save paths
MODEL_BINARY_DETECTOR = MODELS_DIR / "stage1_binary_detector.pth"
MODEL_SHAPE_CLASSIFIER = MODELS_DIR / "stage2_shape_classifier.pth"
MODEL_K1_PREDICTOR = MODELS_DIR / "stage3_k1_predictor.pth"

# ============================================================================
# INFERENCE CONFIG
# ============================================================================
CONFIDENCE_THRESHOLD = 0.85
K1_PREDICTION_CONFIDENCE_THRESHOLD = 0.80

# ============================================================================
# LOGGING & VISUALIZATION
# ============================================================================
LOG_INTERVAL = 10  # Print metrics every N batches
SAVE_INTERVAL = 5  # Save checkpoint every N epochs
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA SPLITS
# ============================================================================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

print("✓ Configuration loaded successfully")
print(f"  Project root: {PROJECT_ROOT}")
print(f"  Data directory: {RAW_DATA_DIR}")
print(f"  Crack shapes: {NUM_CRACK_SHAPES}")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
