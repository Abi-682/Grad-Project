# CNN for Stress Intensity Factor Prediction: Thesis-Grad Project Integration

## The Synergy: Why This is Perfect for Grad-Level Work

### Your Thesis (Existing)
- Study 14 different crack shapes
- All specimens: identical dimensions, centered cracks, same loading
- Calculate K1 (stress intensity factor) for each shape via FEA/analytical methods
- **Dataset Generated**: Hundreds of specimen images + corresponding K1 values

### Your Grad Project (Proposed)
- **Goal**: Train CNN to predict K1 directly from specimen images
- **Input**: Specimen image with visible crack shape
- **Output**: Estimated K1 value
- **Validation**: Compare CNN predictions vs. thesis-calculated K1 values
- **Innovation**: Eliminates need for expensive FEA—just take a photo, predict fracture mechanics!

---

## Why This is Exceptional for Grad Projects

### 1. **Novel Problem Formulation**
- Not just classification ("crack present?") or detection (bounding boxes)
- **Regression with physics meaning**: Predict a real engineering parameter (K1)
- Physics-informed ML—hot research topic in engineering

### 2. **Built-In Ground Truth**
- Your thesis already provides labeled data: Image ↔ K1 value
- No need to manually label (rare advantage!)
- Can use 14 shapes × multiple load levels × image variations

### 3. **Clear Evaluation Against Baseline**
- **Baseline**: Traditional FEA + analytical formulas (your thesis)
- **CNN Alternative**: Fast image-based prediction
- Easy to show CNN accuracy, speed comparison, practical value

### 4. **Multi-Level Complexity Options**
- **Level 1** (Feasible): Predict K1 as single output from full specimen image
- **Level 2** (Strong): Simultaneously classify crack shape + predict K1
- **Level 3** (Advanced): Localize crack, extract geometry, predict K1 from extracted features

### 5. **Research-Level Contributions**
- Transfer learning on FEA-generated images
- Physics-informed loss functions
- Uncertainty quantification (confidence in K1 prediction)
- Cross-shape generalization (train on 10 shapes, test on 4 unseen shapes)

---

## Practical Implementation Strategy

### **Phase 1: Data Preparation (Your Thesis Work)**
Generate dataset from thesis:
```
14 crack shapes × multiple load levels × experimental/FEA images
= Training dataset: ~500-2000 labeled pairs (Image, K1_value)

Example:
├── Shape_1_VNotch/
│   ├── image_1.png  → K1 = 1.245 MPa√m
│   ├── image_2.png  → K1 = 1.502 MPa√m
│   └── image_n.png
├── Shape_2_Rounded/
│   ├── image_1.png  → K1 = 0.987 MPa√m
│   └── ...
└── Shape_14_xxx/
    └── ...
```

### **Phase 2: CNN Architecture (Grad Project)**

#### **Option A: ResNet Regression (Recommended - Fast)**
```
ResNet-50 (pretrained on ImageNet)
  ↓
Remove classification head
  ↓
Add regression head: Conv → FC → FC → Linear(output: K1)
  ↓
Fine-tune on thesis dataset
```
- **Advantage**: Fast to implement, good accuracy
- **Loss function**: MSE(predicted_K1, actual_K1)

#### **Option B: Multi-Task CNN (Stronger)**
```
Shared backbone (ResNet-50)
  ├→ Classification head: Predict crack shape (14 classes)
  └→ Regression head: Predict K1 value
```
- **Advantage**: Learn better features, classify shape AND predict K1
- **Loss function**: α·CrossEntropy(shape) + β·MSE(K1)

#### **Option C: Physics-Informed CNN (Most Advanced)**
```
Input image
  ↓
CNN extracts features: [crack_depth, crack_angle, stress_concentration_factor, ...]
  ↓
Physics layer: K1 = physics_model(extracted_features)
  ↓
Output: Predicted K1
```
- **Advantage**: Interpretable, physically meaningful
- **Research novelty**: Physics-informed neural networks (PINNs)

### **Phase 3: Training Pipeline**
```python
1. Split thesis dataset: 70% train, 15% val, 15% test
2. Data augmentation: Rotation, brightness, noise (realistic variations)
3. Train CNN: Minimize |predicted_K1 - actual_K1|
4. Validation: Track MSE, RMSE, R² on validation set
5. Test: Final accuracy on hold-out test set
```

### **Phase 4: Evaluation & Comparison**

#### **Metrics:**
- **RMSE** (Root Mean Squared Error): How far off predictions are
- **R² Score** (Coefficient of determination): How well CNN explains K1 variance
- **MAE** (Mean Absolute Error): Average prediction error in MPa√m

#### **Comparison Studies:**
```
1. CNN vs. FEA: Speed comparison
   - FEA: Minutes per specimen
   - CNN: Milliseconds per image
   
2. CNN Accuracy vs. Theoretical K1:
   - Error bars on predictions
   - Percentage accuracy
   
3. Cross-Shape Generalization:
   - Train on shapes 1-10
   - Test on shapes 11-14 (never seen before)
   - How well does it generalize?
```

---

## Dataset Structure (Example)

### From Your Thesis:
```
Thesis Output:
├── K1_values.csv (14 shapes × multiple load levels)
│   ├── Shape, Load_Level, K1_calculated, Image_file
│   ├── V_Notch, 1000N, 1.245, v_notch_1.png
│   ├── V_Notch, 1500N, 1.502, v_notch_2.png
│   ├── Rounded, 1000N, 0.987, rounded_1.png
│   └── ...
│
└── specimen_images/
    ├── v_notch_1.png
    ├── v_notch_2.png
    ├── rounded_1.png
    └── ...
```

### For Grad Project:
```
cnn_project/
├── data/
│   ├── raw/
│   │   ├── images/ (all specimen images)
│   │   └── K1_values.csv
│   ├── processed/
│   │   ├── train/ (70%)
│   │   ├── val/ (15%)
│   │   └── test/ (15%)
│   └── splits.json
├── models/
│   ├── resnet_regression.py
│   └── multi_task_cnn.py
├── train.py
├── evaluate.py
└── results/
    ├── predictions.csv
    └── plots/
```

---

## Grad Project Scope (Pick One Level)

### **Level 1: Regression (Feasible, Solid Grade)**
- Input: Full specimen image
- Output: Single K1 value
- CNN: ResNet-50 regression head
- Dataset: 14 shapes × ~30 load levels = ~420 images
- Accuracy target: RMSE < 0.15 MPa√m, R² > 0.85

### **Level 2: Multi-Task (Strong, A-Level)**
- Input: Full specimen image
- Output 1: Crack shape class (14 classes)
- Output 2: K1 value
- CNN: Shared backbone + dual heads
- Dataset: Same as Level 1
- Accuracy target: 90%+ shape accuracy, RMSE < 0.10, R² > 0.90

### **Level 3: Physics-Informed (Advanced, A+)**
- Input: Specimen image
- Intermediate: Extract features (crack geometry, stress concentration)
- Output: K1 via physics model
- Innovation: Interpretable + physics-grounded
- Accuracy target: RMSE < 0.08, R² > 0.92, generalize to unseen shapes

---

## Week-by-Week Plan (Weeks 15-16)

### **Week 15: Implementation**

**Days 1-2:**
- Organize thesis data into training format
- Split into train/val/test (70/15/15)
- Create data loader pipeline

**Days 3-4:**
- Implement ResNet regression model
- Set up training loop, loss function
- Start training on GPU

**Days 5-7:**
- Train to convergence
- Validate on hold-out set
- Generate predictions on test set
- Calculate RMSE, R², MAE metrics

### **Week 16: Analysis & Report**

**Days 1-3:**
- Compare CNN predictions vs. thesis K1 calculations
- Generate plots: Prediction vs. Actual, residuals, error distribution
- Test generalization on unseen crack shapes (if possible)

**Days 4-5:**
- Write report:
  - Introduction (thesis context, gap in literature)
  - Method (CNN architecture, training procedure)
  - Results (metrics, comparison plots)
  - Analysis (why it works, where it fails, insights)
  - Conclusion (practical value, future work)

**Days 6-7:**
- Buffer for refinement, visualizations, final checks

---

## Why This Stands Out Academically

1. **Addresses Real Engineering Need**
   - Manufacturing: Fast K1 estimation without FEA
   - Quality control: Predict fracture risk from visual inspection
   
2. **Physics-ML Integration**
   - Not just "AI for AI's sake"
   - Grounded in classical fracture mechanics
   - Can explain CNN outputs in engineering terms

3. **Built-in Validation**
   - Compare to known correct answers (thesis K1 calculations)
   - Not a black box—can verify accuracy

4. **Scalability Potential**
   - Works for any crack shape or material
   - Transfer learning to new geometries
   - Real-world deployment ready

5. **Research Novelty**
   - Combines two domains: ML + Fracture Mechanics
   - Physics-informed neural networks
   - Uncertainty quantification optional

---

## Success Criteria (Grad-Level Expectations)

✅ **Must Have:**
- CNN trained and converged (RMSE < 0.2 MPa√m, R² > 0.80)
- Comparison with thesis baseline
- Written analysis of results

✅ **Should Have:**
- Multi-task learning (shape + K1)
- Cross-shape generalization test
- Clear visualization of predictions vs. actuals
- Discussion of engineering implications

✅ **Nice to Have (A+):**
- Physics-informed architecture
- Uncertainty quantification (confidence intervals on predictions)
- Real-time inference demo (image → K1 in <100ms)
- Synthetic data augmentation from FEA
