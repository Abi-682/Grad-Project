# Manufacturing Defect Detection CNN: 3-Stage Pipeline

## Project Overview

**Goal**: Automated manufacturing quality control system that detects, classifies, and quantifies defects in specimens.

**Pipeline Architecture:**
```
Specimen Image
        ↓
    ┌───────────────┐
    │  Stage 1      │
    │  DETECTION    │  → Is there a crack?
    │  (Binary)     │
    └───────────────┘
         ↓ YES
    ┌───────────────┐
    │  Stage 2      │
    │ CLASSIFICATION│  → Which shape? (14 classes)
    │  (14-way)     │
    └───────────────┘
         ↓
    ┌───────────────┐
    │  Stage 3      │
    │  REGRESSION   │  → What is K1? (Severity)
    │  (K1 Predict) │
    └───────────────┘
         ↓
    QUALITY DECISION
    "DEFECT: V-Notch | K1=1.24 | Risk: HIGH"
```

---

## Stage Breakdown

### **Stage 1: Binary Defect Detection**
- **Input**: Specimen image
- **Output**: Class = {No Defect, Defect}
- **Model**: ResNet-50 classifier (2 classes)
- **Purpose**: Quick go/no-go decision
- **Target Accuracy**: >95%

### **Stage 2: Crack Shape Classification** (only if Defect detected)
- **Input**: Specimen image + Binary classification (crack present)
- **Output**: Crack Type = {Shape_1, Shape_2, ..., Shape_14}
- **Model**: ResNet-50 classifier (14 classes)
- **Purpose**: Identify defect mechanism
- **Target Accuracy**: >90%

### **Stage 3: Severity Prediction via K1 Regression** (conditional)
- **Input**: Specimen image + Identified crack shape
- **Output**: Continuous value K1 [MPa√m]
- **Model**: ResNet-50 regression head (or shape-specific model)
- **Purpose**: Risk quantification and material state assessment
- **Target**: RMSE < 0.15 MPa√m, R² > 0.85

---

## Dataset Requirements

### **Data Composition**

```
Training Dataset Structure:
├── Class 0: No Defect (Uncracked specimens)
│   ├── specimen_1.png
│   ├── specimen_2.png
│   └── ...
│
├── Class 1: Defect Present
│   ├── Shape_1_VNotch/
│   │   ├── specimen_1.png  → K1 = 1.245 MPa√m
│   │   ├── specimen_2.png  → K1 = 1.502 MPa√m
│   │   └── ...
│   ├── Shape_2_Rounded/
│   │   ├── specimen_1.png  → K1 = 0.987 MPa√m
│   │   └── ...
│   └── Shape_14_xxx/
│       └── ...
```

### **Target Dataset Size**
- **No Defect images**: 200-400 specimens
- **Per crack shape**: 30-50 images × 14 shapes = 420-700 images
- **Total**: ~600-1100 images
- **Split**: 70% train, 15% val, 15% test

### **Data Augmentation** (increase effective dataset)
- Rotation: ±15°
- Brightness: ±20%
- Gaussian noise: σ = 0.01-0.02
- Zoom: 0.9-1.1x
- Horizontal flip: (if symmetric)

---

## Implementation Architecture

### **Option A: Sequential Pipeline (Recommended)**
```python
# Stage 1: Binary detector
binary_detector = ResNet50(num_classes=2)
is_defect = binary_detector(image)  # 0 or 1

if is_defect == 1:
    # Stage 2: Shape classifier
    shape_classifier = ResNet50(num_classes=14)
    crack_shape = shape_classifier(image)  # 0-13
    
    # Stage 3: K1 predictor (optional: shape-specific or unified)
    k1_predictor = ResNet50(output_dim=1)  # Single regression output
    predicted_k1 = k1_predictor(image)  # Continuous K1 value
    
    risk_level = "HIGH" if predicted_k1 > threshold else "MEDIUM"
else:
    crack_shape = None
    predicted_k1 = None
    risk_level = "SAFE"

return {
    "defect_present": is_defect,
    "crack_shape": crack_shape,
    "k1_value": predicted_k1,
    "risk_level": risk_level
}
```

### **Option B: Multi-Task Learning (More Elegant)**
```python
# Single model with multiple output heads
class DefectDetectionCNN(nn.Module):
    def __init__(self):
        self.backbone = ResNet50(pretrained=True)
        
        # Head 1: Binary detection
        self.detection_head = nn.Linear(2048, 2)
        
        # Head 2: Shape classification (if crack present)
        self.classification_head = nn.Linear(2048, 14)
        
        # Head 3: K1 regression
        self.regression_head = nn.Linear(2048, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        
        detection = self.detection_head(features)      # (batch, 2)
        classification = self.classification_head(features)  # (batch, 14)
        k1_value = self.regression_head(features)      # (batch, 1)
        
        return detection, classification, k1_value

# Training with combined loss
loss = (
    α * cross_entropy(detection, y_detection) +
    β * cross_entropy(classification, y_shape) +
    γ * mse(k1_value, y_k1)
)
```

### **Option C: Shape-Specific K1 Models (Most Accurate)**
```python
# Train separate K1 regressor for each crack shape
# This captures shape-specific K1 patterns better

k1_models = {}
for shape_id in range(14):
    k1_models[shape_id] = ResNet50(output_dim=1)

# During inference:
crack_shape = classifier(image)  # Get shape
predicted_k1 = k1_models[crack_shape](image)  # Use shape-specific model
```

---

## Training & Evaluation Strategy

### **Phase 1: Train Binary Detector (Stage 1)**
```
Objective: No Defect vs. Defect (2-class classification)
Epochs: 20-30
Batch size: 32
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss
Target: >95% accuracy on test set
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
```

### **Phase 2: Train Shape Classifier (Stage 2)** (on defect images only)
```
Objective: 14-way classification of crack shapes
Epochs: 30-40
Batch size: 32
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss
Target: >90% accuracy on test set
Metrics: Per-class accuracy, confusion matrix
```

### **Phase 3: Train K1 Regressor (Stage 3)** (on defect images)
```
Objective: Predict continuous K1 value
Epochs: 20-30
Batch size: 32
Optimizer: Adam (lr=0.001)
Loss: MSE (or Huber for robustness)
Target: RMSE < 0.15 MPa√m, R² > 0.85
Metrics: MAE, RMSE, R², correlation coefficient
```

### **Phase 4: Integration & Validation**
```
1. Combine all 3 stages into pipeline
2. Test on full dataset (no defect + all 14 shapes)
3. Generate confusion matrices and prediction distributions
4. Compare CNN K1 predictions vs. thesis ground truth
```

---

## Evaluation Metrics

### **Binary Detection (Stage 1)**
```
Accuracy    = (TP + TN) / Total
Precision   = TP / (TP + FP)    → How many detected defects are real?
Recall      = TP / (TP + FN)    → How many real defects did we find?
F1-Score    = 2 * (Precision * Recall) / (Precision + Recall)
ROC-AUC     → Threshold independence metric
```

### **Shape Classification (Stage 2)**
```
Overall Accuracy: Correct predictions / Total predictions
Per-class Accuracy: For each of 14 shapes
Confusion Matrix: What shapes get confused with each other?
Macro F1-Score: Average F1 across all 14 classes
```

### **K1 Regression (Stage 3)**
```
MAE (Mean Absolute Error)   = mean(|predicted - actual|)
RMSE (Root MSE)              = sqrt(mean((predicted - actual)²))
R² Score                     = 1 - (SS_res / SS_tot)
Correlation Coefficient      = Pearson correlation of predictions vs. actuals
Max Error                    = max(|predicted - actual|)
```

---

## Expected Outputs for Grad Report

### **Week 15: Implementation Results**
- [ ] Trained binary detector (accuracy ≥95%)
- [ ] Trained shape classifier (accuracy ≥90%)
- [ ] Trained K1 predictor (RMSE ≤0.15, R² ≥0.85)
- [ ] Integrated 3-stage pipeline
- [ ] Confusion matrices & prediction plots

### **Week 16: Written Report Sections**
1. **Introduction**: Manufacturing defect detection needs, CNN benefits
2. **Methods**: 
   - Dataset description (600-1100 images, 14 shapes)
   - Architecture choices (ResNet-50 backbone + task-specific heads)
   - Training procedure (losses, hyperparameters)
3. **Results**:
   - Metrics for each stage (tables & plots)
   - Confusion matrices
   - Prediction accuracy comparisons
   - CNN vs. manual inspection (if applicable)
4. **Analysis**:
   - Why each stage performs as it does
   - Failure cases (which shapes are confusing?)
   - K1 prediction accuracy vs. thesis ground truth
   - Engineering implications
5. **Conclusion & Future Work**:
   - Practical deployment considerations
   - Real-time inference capability
   - Extension to unseen crack geometries

---

## Success Checklist

✅ **Minimum (Pass)**
- Binary defect detection: ≥90% accuracy
- Pipeline runs end-to-end
- Report documents each stage

✅ **Strong (A)**
- Binary detection: ≥95% accuracy
- Shape classification: ≥85% accuracy
- K1 prediction: RMSE < 0.20, R² > 0.80
- Comparison with baseline (manual classification or traditional CV)

✅ **Excellent (A+)**
- Binary: ≥96% accuracy
- Classification: ≥90% accuracy
- K1 prediction: RMSE < 0.12, R² > 0.90
- Cross-shape generalization test (train on 10 shapes, test on 4 unseen)
- Physics insights: Discuss how CNN learns crack geometry features
- Real-time deployment demo (inference <50ms/image)

---

## Key Advantages of This Approach

1. **Clear Problem Definition**: Manufacturing quality control (real-world use case)
2. **Thesis Integration**: Uses your 14-shape dataset as ground truth
3. **Progressive Complexity**: 3 independent stages (can be tuned separately)
4. **Practical Output**: K1 value has real engineering meaning (fracture mechanics)
5. **Measurable Success**: Clear metrics at each stage
6. **Scalable**: Can extend to new materials/geometries later

---

## Timeline

**Week 15 (Implementation):**
- Days 1-2: Prepare dataset, organize by stage
- Days 3-4: Train Stage 1 (binary detector)
- Days 5-6: Train Stage 2 (14-way classifier)
- Days 7: Train Stage 3 (K1 regressor) + integrate pipeline

**Week 16 (Report & Analysis):**
- Days 1-2: Evaluation, metric calculations, plotting
- Days 3-4: Analysis, comparison studies
- Days 5-7: Report writing, visualization refinement
