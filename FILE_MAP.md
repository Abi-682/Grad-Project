# Project File Map & Quick Reference

## 📂 Complete Project Structure

```
c:\Soft\MI\Grad-Project/
│
├─ 📄 README.md                          ← START HERE (Project overview)
├─ 📄 QUICK_START.md                     ← Installation & usage guide
├─ 📄 IMPLEMENTATION_SUMMARY.md           ← What's been done
│
├─ 🐍 demo.py                           ← Run this first: python demo.py
│
├─ 📁 src/
│  ├─ 🐍 config.py                      ← K1 values, paths, hyperparameters
│  ├─ 🐍 models.py                      ← 4 CNN architectures
│  ├─ 🐍 dataset.py                     ← DataLoader & augmentation
│  ├─ 🐍 train.py                       ← Training scripts for 3 stages
│  ├─ 🐍 inference.py                   ← Complete inference pipeline
│  └─ 🐍 hello.py                       ← Original template
│
├─ 📁 data/
│  ├─ 📁 raw/                          ← Your specimen images go here
│  │  ├─ no_defect/                    ← Uncracked specimens
│  │  ├─ defect_cracked/               ← Cracked + manufacturing issue
│  │  └─ crack_shapes/                 ← C1-C14 crack types
│  ├─ 📁 processed/                    ← Train/val/test splits (auto-generated)
│  └─ 📄 K1_DATASET_MASTER.csv         ← Labels file (you create this)
│
├─ 📁 models/
│  ├─ 📦 stage1_binary_detector.pth    ← Trained model (after training)
│  ├─ 📦 stage2_shape_classifier.pth   ← Trained model (after training)
│  └─ 📦 stage3_k1_predictor.pth       ← Trained model (after training)
│
├─ 📁 results/
│  ├─ 📁 visualizations/               ← Plots & charts
│  └─ 📊 inference_results.json         ← Results (after inference)
│
├─ 📁 references/
│  ├─ 📁 specimen_images/              ← Reference images
│  ├─ 📄 PROJECT_STRATEGY.md            ← Strategy document
│  ├─ 📄 CNN_MANUFACTURING_DEFECT_DETECTION.md ← Full spec
│  ├─ 📄 DATASET_ORGANIZATION.md        ← Data structure guide
│  └─ 📄 IMAGE_REFERENCE_CATALOG.md     ← Image references
│
├─ 📁 scripts/
│  ├─ bootstrap-windows.ps1
│  ├─ bootstrap-macos.sh
│  └─ bootstrap-linux.sh
│
├─ 📁 .vscode/
│  └─ settings.json
│
├─ 📄 pyproject.toml                    ← Dependencies (updated)
├─ 📄 .python-version
├─ 📄 uv.lock
└─ 📄 .gitignore
```

---

## 🚀 Quick Commands

```bash
# Install dependencies (one-time)
cd c:\Soft\MI\Grad-Project
uv sync

# Run demo (verify everything works)
python demo.py

# Train all 3 stages
python src/train.py \
    --data-dir data/raw \
    --labels-file data/K1_DATASET_MASTER.csv \
    --stage 0 \
    --epochs 50

# Train individual stage
python src/train.py \
    --data-dir data/raw \
    --labels-file data/K1_DATASET_MASTER.csv \
    --stage 1              # 1 for binary, 2 for classifier, 3 for regressor

# Run inference on single image
python src/inference.py path/to/specimen.png --save-result

# Run inference on batch
python -c "
from src.inference import DefectDetectionPipeline
from pathlib import Path

pipeline = DefectDetectionPipeline()
images = list(Path('data/raw/crack_shapes').glob('*.png'))
results = pipeline.infer_batch(images)
"
```

---

## 📊 K1 Reference Table (Hardcoded in config.py)

```
Crack Type │ K1 Value (N/mm^1.5)
───────────┼─────────────────────
C1         │ 552.45
C2         │ 3917.00
C3         │ 6839.90
C4         │ 8810.30
C5         │ 10312.00
C6         │ 10979.00
C7         │ 12690.00
C8         │ 13132.00
C9         │ 13497.00
C10        │ 13782.00
C11        │ 13848.00
C12        │ 15205.00
C13        │ 16289.00
C14        │ 17516.00
```

---

## 🔄 Workflow: Data → Train → Test → Report

```
WEEK 15: IMPLEMENTATION
┌─────────────────────────────────────────────────────┐
│  Step 1: Prepare Data (Days 1-2)                    │
│  ├─ Organize images into data/raw/ folders          │
│  ├─ Create K1_DATASET_MASTER.csv labels             │
│  └─ Verify: python -c "import pandas; ..."          │
├─────────────────────────────────────────────────────┤
│  Step 2: Train Models (Days 3-6)                    │
│  ├─ Stage 1: python src/train.py --stage 1          │
│  ├─ Stage 2: python src/train.py --stage 2          │
│  ├─ Stage 3: python src/train.py --stage 3          │
│  └─ Check: models/ folder for .pth files            │
├─────────────────────────────────────────────────────┤
│  Step 3: Evaluate Models (Day 7)                    │
│  ├─ python src/inference.py test_image.png          │
│  ├─ Generate metrics & plots                        │
│  └─ Save results to results/ folder                 │
└─────────────────────────────────────────────────────┘

WEEK 16: REPORTING
┌─────────────────────────────────────────────────────┐
│  Step 1: Analysis (Days 1-3)                        │
│  ├─ Compile confusion matrices                      │
│  ├─ Plot accuracy curves                            │
│  └─ Generate comparison tables                      │
├─────────────────────────────────────────────────────┤
│  Step 2: Write Report (Days 4-7)                    │
│  ├─ Introduction: Defect detection motivation       │
│  ├─ Methods: Architecture & training procedure      │
│  ├─ Results: Tables, plots, metrics                 │
│  └─ Conclusion: Key findings & future work          │
└─────────────────────────────────────────────────────┘
```

---

## 📝 Data CSV Format Example

**File**: `data/K1_DATASET_MASTER.csv`

```csv
image_name,crack_shape_class,k1_value
no_defect/specimen_001.png,0,0
no_defect/specimen_002.png,0,0
no_defect/specimen_003.png,0,0
defect_cracked/malformed_001.png,0,0
C1_specimen_001.png,0,552.45
C1_specimen_002.png,0,552.45
C1_specimen_003.png,0,552.45
C2_specimen_001.png,1,3917.00
C2_specimen_002.png,1,3917.00
...
C14_specimen_050.png,13,17516.00
```

**Columns:**
- `image_name`: Relative path from `data/raw/`
- `crack_shape_class`: 0 for no_defect, 0-13 for C1-C14
- `k1_value`: K1 value (0 for uncracked, actual K1 for cracked)

---

## 🔧 Configuration Quick Reference

**File**: `src/config.py`

```python
# Paths
PROJECT_ROOT = c:\Soft\MI\Grad-Project
DATA_DIR = c:\Soft\MI\Grad-Project\data
RAW_DATA_DIR = c:\Soft\MI\Grad-Project\data\raw
MODELS_DIR = c:\Soft\MI\Grad-Project\models

# K1 Values (14 shapes)
K1_VALUES = {"C1": 552.45, "C2": 3917.0, ...}

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
IMAGE_SIZE = 224

# Model Paths
MODEL_BINARY_DETECTOR = models/stage1_binary_detector.pth
MODEL_SHAPE_CLASSIFIER = models/stage2_shape_classifier.pth
MODEL_K1_PREDICTOR = models/stage3_k1_predictor.pth

# Device
DEVICE = "cuda"  # or "cpu"
```

---

## 🎯 Expected Performance

| Metric | Stage | Target |
|--------|-------|--------|
| Accuracy | Binary Detection | ≥95% |
| Accuracy | Shape Classification | ≥90% |
| RMSE | K1 Prediction | <0.15 |
| R² Score | K1 Prediction | >0.85 |

---

## ✅ What's Ready to Go

✓ All source code (5 Python modules)  
✓ Training pipeline (automated via train.py)  
✓ Inference pipeline (ready for testing)  
✓ Data structure (created & ready)  
✓ Configuration (K1 values embedded)  
✓ Documentation (4 detailed guides)  
✓ Demo script (tested & working)  
✓ Dependencies (installed via uv sync)  

---

## ⏳ What You Need to Provide

⏳ Specimen images (data/raw/ folders)  
⏳ K1_DATASET_MASTER.csv (labels file)  
⏳ Run training (automated)  
⏳ Run inference (automated)  
⏳ Write report (analysis of results)  

---

## 📚 Documentation Map

| Document | Purpose | When to Read |
|----------|---------|--------------|
| README.md | Overview & intro | First |
| QUICK_START.md | Setup & usage | Before training |
| demo.py | See it in action | Verify installation |
| config.py | Settings & K1 values | Customize training |
| IMPLEMENTATION_SUMMARY.md | What's complete | Understand status |
| CNN_MANUFACTURING_DEFECT_DETECTION.md | Full technical spec | Deep dive |

---

## 🐛 Debugging Help

```bash
# Check if all dependencies are installed
python -c "import torch, torchvision, albumentations, pandas; print('✓ All installed')"

# Verify K1 values are loaded
python -c "from src.config import K1_VALUES; print(f'K1 values: {len(K1_VALUES)} shapes')"

# Check data structure
python -c "
from pathlib import Path
for cat in ['no_defect', 'defect_cracked', 'crack_shapes']:
    p = Path(f'data/raw/{cat}')
    count = len(list(p.glob('*.png')))
    print(f'{cat}: {count} images')
"

# Test single inference (dummy)
python -c "
from src.models import BinaryDefectDetector
import torch
model = BinaryDefectDetector()
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(f'✓ Model works: output shape {y.shape}')
"
```

---

## 💾 Checkpoint: Before You Start Training

- [ ] Data organized in data/raw/
- [ ] K1_DATASET_MASTER.csv created
- [ ] CSV format verified (3 columns)
- [ ] demo.py runs successfully
- [ ] Python environment activated
- [ ] All dependencies installed (uv sync)
- [ ] Models directory exists
- [ ] Results directory exists

**Once all checked, you're ready to run:**
```bash
python src/train.py --data-dir data/raw --labels-file data/K1_DATASET_MASTER.csv --stage 0
```

---

## 🎓 Learning Resources

**In This Project:**
- Inference pipeline example → `inference.py`
- Data loading example → `dataset.py`
- Model architecture example → `models.py`
- Training loop example → `train.py`

**Quick Python Example:**
```python
# Load & use trained model
from src.inference import DefectDetectionPipeline

pipeline = DefectDetectionPipeline(device='cuda')
result = pipeline.infer('specimen.png')

print(f"Defect: {result['defect_detected']}")
print(f"Shape: {result.get('crack_shape', 'N/A')}")
print(f"K1: {result.get('k1_predicted', 'N/A')}")
print(f"Risk: {result.get('risk_level', 'N/A')}")
```

---

**Last Updated**: April 28, 2026  
**Status**: ✅ Ready for Week 15 Implementation  
**Questions?** See QUICK_START.md or README.md
