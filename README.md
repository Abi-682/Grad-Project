# Manufacturing Defect Detection using CNN

**Grad Project: Week 15 Implementation**

A complete 3-stage convolutional neural network pipeline for detecting and classifying manufacturing defects in specimens, with K1 (Stress Intensity Factor) prediction. Integrates directly with thesis research on V-notch crack shapes in 3-point bending specimens.

## Project Overview

```
DEFECT DETECTION PIPELINE
    ↓
┌─────────────────────────────────┐
│ STAGE 1: Binary Detection       │
│ Input: Specimen Image           │
│ Output: Defect/No Defect        │
│ Model: ResNet-50 (2 classes)    │
└─────────────────────────────────┘
         ↓ (if defect)
┌─────────────────────────────────┐
│ STAGE 2: Shape Classification   │
│ Input: Specimen Image           │
│ Output: C1-C14 (14 classes)     │
│ Model: ResNet-50 (14 classes)   │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│ STAGE 3: K1 Regression          │
│ Input: Specimen Image           │
│ Output: K1 Value (severity)     │
│ Model: ResNet-50 (regression)   │
└─────────────────────────────────┘
         ↓
    QUALITY ASSESSMENT
    (Risk Level: LOW/MEDIUM/HIGH/CRITICAL)
```

## Key Features

- **3-Stage Pipeline**: Detection → Classification → Severity Prediction
- **14 Crack Shapes**: C1-C14 with known K1 values from thesis
- **ResNet-50 Backbone**: Transfer learning from ImageNet
- **Confidence Scoring**: Probabilistic outputs at each stage
- **Risk Assessment**: Automatic quality control decisions
- **Data Augmentation**: Rotation, brightness, zoom, Gaussian noise
- **Batch Processing**: Inference on multiple images
- **JSON Output**: Structured results for integration

## Quick Start

### 1. Environment Setup
```bash
cd c:\Soft\MI\Grad-Project
uv sync
```

### 2. Verify Installation
```bash
python demo.py
```

### 3. Prepare Your Data
Organize images:
```
data/raw/
├── no_defect/
│   ├── specimen_001.png
│   └── ...
├── defect_cracked/
│   ├── defect_001.png
│   └── ...
└── crack_shapes/
    ├── C1_specimen_001.png
    ├── C2_specimen_001.png
    └── ...
```

Create labels file: `data/K1_DATASET_MASTER.csv`
```csv
image_name,crack_shape_class,k1_value
no_defect/specimen_001.png,0,0
C1_specimen_001.png,0,552.45
C2_specimen_001.png,1,3917.00
```

### 4. Train Models
```bash
python src/train.py \
    --data-dir data/raw \
    --labels-file data/K1_DATASET_MASTER.csv \
    --stage 0 \
    --epochs 50
```

### 5. Run Inference
```bash
python src/inference.py path/to/specimen.png --save-result
```

## Project Structure

```
Grad-Project/
├── src/
│   ├── config.py              # K1 values, paths, hyperparameters
│   ├── models.py              # CNN architectures (ResNet-50 based)
│   ├── dataset.py             # DataLoader & augmentation
│   ├── train.py               # Training scripts for 3 stages
│   ├── inference.py           # Complete inference pipeline
│   └── hello.py               # Original template file
├── data/
│   ├── raw/                   # Raw specimen images
│   ├── processed/             # Preprocessed data splits
│   └── K1_DATASET_MASTER.csv  # Labels & K1 values
├── models/                    # Trained model weights
│   ├── stage1_binary_detector.pth
│   ├── stage2_shape_classifier.pth
│   └── stage3_k1_predictor.pth
├── results/                   # Inference results & visualizations
├── references/                # Documentation & specimen images
├── demo.py                    # Quick start demo
├── QUICK_START.md             # Detailed getting started guide
├── pyproject.toml             # Dependencies
└── README.md                  # This file
```

## K1 Reference Values (From Thesis)

| Shape | K1 (N/mm^1.5) | Shape | K1 (N/mm^1.5) |
|-------|--------------|-------|--------------|
| C1    | 552.45       | C8    | 13132.00     |
| C2    | 3917.00      | C9    | 13497.00     |
| C3    | 6839.90      | C10   | 13782.00     |
| C4    | 8810.30      | C11   | 13848.00     |
| C5    | 10312.00     | C12   | 15205.00     |
| C6    | 10979.00     | C13   | 16289.00     |
| C7    | 12690.00     | C14   | 17516.00     |

## Code Examples

### Batch Inference
```python
from src.inference import DefectDetectionPipeline
from pathlib import Path

pipeline = DefectDetectionPipeline(device='cuda')

# Single image
result = pipeline.infer('specimen.png')
print(f"Defect: {result['defect_detected']}")
print(f"Shape: {result['crack_shape']}")
print(f"K1: {result['k1_predicted']}")
print(f"Risk: {result['risk_level']}")

# Batch processing
images = list(Path('data/raw/crack_shapes').glob('*.png'))
results = pipeline.infer_batch(images)
```

### Custom Training
```python
from src.models import BinaryDefectDetector
from src.dataset import create_dataloaders
import torch.optim as optim

model = BinaryDefectDetector(pretrained=True, num_classes=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader, val_loader = create_dataloaders(
    'data/raw', 
    'data/K1_DATASET_MASTER.csv',
    task='binary'
)
```

## Inference Output Example

```json
{
  "image": "data/raw/crack_shapes/C5_specimen_042.png",
  "timestamp": "2026-04-28T14:32:15.789012",
  "defect_detected": true,
  "defect_class": "Defect",
  "defect_confidence": 0.9823,
  "crack_shape": "C5",
  "crack_shape_confidence": 0.8901,
  "k1_predicted": 10156.34,
  "k1_reference": 10312.00,
  "k1_ratio": 0.9849,
  "risk_level": "HIGH"
}
```

## Model Architecture

### Stage 1: Binary Detector
- **Input**: 224×224 RGB image
- **Backbone**: ResNet-50 (pretrained ImageNet)
- **Output**: 2-class softmax (No Defect, Defect)
- **Parameters**: 24.7M

### Stage 2: Shape Classifier
- **Input**: 224×224 RGB image
- **Backbone**: ResNet-50 (pretrained ImageNet)
- **Output**: 14-class softmax (C1-C14)
- **Parameters**: 26.3M

### Stage 3: K1 Predictor
- **Input**: 224×224 RGB image
- **Backbone**: ResNet-50 (pretrained ImageNet)
- **Output**: Single continuous value (regression)
- **Parameters**: 24.7M

## Training Configuration

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224

# Data augmentation
- Rotation: ±15°
- Brightness: ±20%
- Zoom: 0.9-1.1x
- Gaussian noise: σ=0.01
```

## Performance Targets

| Stage | Metric | Target |
|-------|--------|--------|
| 1 | Accuracy | ≥95% |
| 2 | Accuracy | ≥90% |
| 3 | RMSE | <0.15 |
| 3 | R² Score | >0.85 |

## Dependencies

- PyTorch 2.0+
- TorchVision
- NumPy
- Pillow
- Albumentations
- Pandas
- Scikit-learn
- Matplotlib
- TQDM

## Files Reference

| File | Description |
|------|-------------|
| [config.py](src/config.py) | Configuration & K1 lookup table |
| [models.py](src/models.py) | CNN architectures |
| [dataset.py](src/dataset.py) | DataLoader & augmentation |
| [train.py](src/train.py) | Training scripts |
| [inference.py](src/inference.py) | Inference pipeline |
| [demo.py](demo.py) | Quick start demo |
| [QUICK_START.md](QUICK_START.md) | Detailed guide |

## Troubleshooting

**Out of Memory (OOM)**
- Reduce batch size: `BATCH_SIZE = 16`
- Use CPU: `DEVICE = "cpu"`

**Poor Accuracy**
- Increase training data
- Increase epochs
- Adjust learning rate

**Import Errors**
- Run: `uv sync` again
- Verify Python version: Python 3.13

## Documentation

- [QUICK_START.md](QUICK_START.md) - Detailed getting started guide
- [CNN_MANUFACTURING_DEFECT_DETECTION.md](references/CNN_MANUFACTURING_DEFECT_DETECTION.md) - Complete pipeline specification
- [DATASET_ORGANIZATION.md](references/DATASET_ORGANIZATION.md) - Data structure guide
- [IMAGE_REFERENCE_CATALOG.md](references/IMAGE_REFERENCE_CATALOG.md) - Specimen image references

## Thesis Integration

This project leverages thesis research:
- **14 Crack Shapes**: From thesis study
- **K1 Values**: Calculated via FEA/analytical methods
- **3-Point Bending**: Specimen loading configuration
- **Innovation**: CNN replaces expensive FEA calculations

## Next Steps (Week 15-16)

**Week 15 (Implementation)**:
1. Organize specimen images into `data/raw/`
2. Create `K1_DATASET_MASTER.csv`
3. Train 3-stage pipeline
4. Generate evaluation metrics
5. Visualize results

**Week 16 (Report)**:
1. Document methods & results
2. Compare CNN vs. FEA accuracy
3. Analyze generalization
4. Write conclusions

---

**Status**: ✅ Code complete | ⏳ Awaiting training data
**Ready for**: Specimen image organization → Model training → Results generation

### Removing a Package

Remove it from `pyproject.toml` and run:

```bash
uv sync
```

### Locking Dependencies

After modifying dependencies, commit both `pyproject.toml` and `uv.lock` to version control:

```bash
git add pyproject.toml uv.lock
git commit -m "Update dependencies"
```

## Resetting Your Environment

If you encounter issues with your Python environment, reset it completely:

**macOS:**
```bash
rm -rf .venv
uv sync
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

Then restart VS Code.

## Running Your Code

### Python Scripts

```bash
python src/my_script.py
```

### Verifying PyTorch Installation

After bootstrap setup completes, verify that PyTorch is working correctly:

```bash
python scripts/test-pytorch.py
```

This script tests PyTorch functionality, CUDA availability (if applicable), tensor operations, and autograd. All tests should pass with a green checkmark.

### Tests

```bash
python -m pytest tests/
```

## Using AI Assistance with Copilot

With GitHub Copilot installed, you can:

- Press `Ctrl+K` (or `Cmd+K` on macOS) to start an inline chat
- Use suggestions as you type code
- Ask questions about your code in the Copilot Chat panel

## Course Resources

- **Development Environment Guide:** See the course notebook on "Development Environments"
- **Python Setup:** Refer to "Setting Up Your Development Environment" section
- **Course Notebooks:** Access course materials through the main course website

## Questions or Issues?

Refer to the course's development environment troubleshooting guide, or reach out to course staff during office hours.

---

**Happy coding! 🚀**
