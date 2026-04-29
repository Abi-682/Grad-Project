# Manufacturing Defect Detection CNN - Quick Start Guide

## Project Structure

```
Grad-Project/
├── src/
│   ├── config.py              # Configuration & K1 lookup table
│   ├── models.py              # CNN architectures (3 stages)
│   ├── dataset.py             # DataLoader & augmentation
│   ├── train.py               # Training script
│   ├── inference.py           # Inference pipeline
│   └── hello.py               # Original hello world
├── data/
│   ├── raw/
│   │   ├── no_defect/         # Images without cracks
│   │   ├── defect_cracked/    # Images with cracks + issues
│   │   └── crack_shapes/      # C1-C14 crack shapes
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── K1_DATASET_MASTER.csv  # Labels file
├── models/
│   ├── stage1_binary_detector.pth
│   ├── stage2_shape_classifier.pth
│   └── stage3_k1_predictor.pth
├── results/
│   └── [inference results & visualizations]
└── references/
    └── [documentation & specimen images]
```

## CNN Architecture Overview

### Stage 1: Binary Detection
```
ResNet-50 (pretrained ImageNet)
  ↓
Classification Head: 2 classes
  ├─ Class 0: "No Defect"
  └─ Class 1: "Defect"

Output: Binary prediction (is there a crack?)
```

### Stage 2: Shape Classification
```
ResNet-50 (pretrained ImageNet)
  ↓
Classification Head: 14 classes
  ├─ C1, C2, C3, ..., C14

Output: Crack shape type (which crack geometry?)
```

### Stage 3: K1 Regression
```
ResNet-50 (pretrained ImageNet)
  ↓
Regression Head: 1 continuous output

Output: K1 value (Stress Intensity Factor)
```

## K1 Reference Values (from Thesis)

| Shape | K1 Value |
|-------|----------|
| C1    | 552.45   |
| C2    | 3917.00  |
| C3    | 6839.90  |
| C4    | 8810.30  |
| C5    | 10312.00 |
| C6    | 10979.00 |
| C7    | 12690.00 |
| C8    | 13132.00 |
| C9    | 13497.00 |
| C10   | 13782.00 |
| C11   | 13848.00 |
| C12   | 15205.00 |
| C13   | 16289.00 |
| C14   | 17516.00 |

## Installation

### 1. Install Dependencies
```bash
cd c:\Soft\MI\Grad-Project
uv sync
```

### 2. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Preparing Your Data

### Step 1: Organize Images
```
data/raw/
├── no_defect/
│   ├── specimen_001.png
│   ├── specimen_002.png
│   └── ...
├── defect_cracked/
│   ├── defect_001.png
│   └── ...
└── crack_shapes/
    ├── C1_specimen_001.png
    ├── C1_specimen_002.png
    ├── C2_specimen_001.png
    └── ...
```

### Step 2: Create Labels CSV
```
data/K1_DATASET_MASTER.csv

image_name,crack_shape_class,k1_value
no_defect/specimen_001.png,0,0
no_defect/specimen_002.png,0,0
C1_specimen_001.png,0,552.45
C2_specimen_001.png,1,3917.00
...
```

**Format:**
- `image_name`: Relative path from `data/raw/`
- `crack_shape_class`: 0 for no_defect, 1-14 for C1-C14
- `k1_value`: K1 value (0 for no_defect, actual K1 for cracked)

## Training

### Train All Stages
```bash
cd c:\Soft\MI\Grad-Project
python src/train.py \
    --data-dir data/raw \
    --labels-file data/K1_DATASET_MASTER.csv \
    --stage 0 \
    --epochs 50
```

### Train Single Stage
```bash
# Stage 1 only
python src/train.py --data-dir data/raw --labels-file data/K1_DATASET_MASTER.csv --stage 1

# Stage 2 only
python src/train.py --data-dir data/raw --labels-file data/K1_DATASET_MASTER.csv --stage 2

# Stage 3 only
python src/train.py --data-dir data/raw --labels-file data/K1_DATASET_MASTER.csv --stage 3
```

## Inference

### Run on Single Image
```bash
python src/inference.py path/to/specimen_image.png --save-result
```

### Example Output
```
======================================================================
DEFECT DETECTION RESULT
======================================================================
Image: specimen_001.png
Timestamp: 2026-04-28T10:30:45.123456

Stage 1 - Defect Detection:
  Status: Defect
  Confidence: 96.50%

Stage 2 - Crack Shape Classification:
  Shape: C5
  Confidence: 89.20%

Stage 3 - K1 Prediction (Stress Intensity Factor):
  Predicted K1: 10156.34 (N/mm^1.5)
  Reference K1: 10312.00 (N/mm^1.5)
  K1 Ratio: 98.49%

QUALITY ASSESSMENT:
  ⚠ RISK LEVEL: HIGH
  ACTION: REJECT SPECIMEN

======================================================================
```

## Code Usage Examples

### 1. Using the Inference Pipeline

```python
from src.inference import DefectDetectionPipeline
from pathlib import Path

# Initialize
pipeline = DefectDetectionPipeline(device='cuda')

# Single image inference
result = pipeline.infer('path/to/image.png')
print(f"Defect detected: {result['defect_detected']}")
print(f"Risk level: {result['risk_level']}")

# Batch inference
image_paths = list(Path('data/raw/crack_shapes').glob('*.png'))
results = pipeline.infer_batch(image_paths)
```

### 2. Custom Training Loop

```python
from src.models import BinaryDefectDetector
from src.dataset import create_dataloaders
import torch
import torch.optim as optim

# Create model
model = BinaryDefectDetector(pretrained=True, num_classes=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Get dataloaders
train_loader, val_loader = create_dataloaders(
    'data/raw', 
    'data/K1_DATASET_MASTER.csv',
    batch_size=32,
    task='binary'
)

# Training
model.train()
for epoch in range(10):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Key Configuration Values (in config.py)

```python
# Model paths
MODEL_BINARY_DETECTOR = "models/stage1_binary_detector.pth"
MODEL_SHAPE_CLASSIFIER = "models/stage2_shape_classifier.pth"
MODEL_K1_PREDICTOR = "models/stage3_k1_predictor.pth"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
IMAGE_SIZE = 224

# K1 Lookup Table
K1_VALUES = {
    "C1": 552.45,
    "C2": 3917.0,
    ...
    "C14": 17516.0
}
```

## Expected Performance Targets

| Stage | Task | Target Metric |
|-------|------|----------------|
| 1 | Binary Detection | Accuracy ≥ 95% |
| 2 | Shape Classification | Accuracy ≥ 90% |
| 3 | K1 Prediction | RMSE < 0.15, R² > 0.85 |

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `BATCH_SIZE = 16` in config.py
- Use CPU: `DEVICE = "cpu"`

### Poor Accuracy
- Increase training data
- Increase number of epochs
- Adjust learning rate

### Model Not Loading
- Verify path to model weights
- Ensure trained models exist before inference

## File Descriptions

| File | Purpose |
|------|---------|
| config.py | K1 values, paths, hyperparameters |
| models.py | ResNet-50 based CNN architectures |
| dataset.py | Image loading, augmentation, dataloaders |
| train.py | Training scripts for all 3 stages |
| inference.py | Complete inference pipeline |

## Next Steps

1. ✅ Code structure created
2. ⏳ Organize specimen images into data/raw/
3. ⏳ Create K1_DATASET_MASTER.csv labels file
4. ⏳ Run training: `python src/train.py ...`
5. ⏳ Test inference: `python src/inference.py ...`
6. ⏳ Generate results for grad report

## References

- Configuration: [config.py](src/config.py)
- Models: [models.py](src/models.py)
- Dataset: [dataset.py](src/dataset.py)
- Training: [train.py](src/train.py)
- Inference: [inference.py](src/inference.py)
