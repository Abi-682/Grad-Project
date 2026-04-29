# CNN-Based Defect Detection in V-Notch 3-Point Bending Specimens

## Project Overview
- **Specimen Type**: 14 different crack shapes (V-notch, rounded, blunt, etc.) on rectangular specimens
- **Crack Configuration**: Centered, identical dimensions, identical boundary conditions
- **Loading Configuration**: 3-point bending (1 load point top, 2 support points bottom)
- **AI Technique**: Convolutional Neural Networks (CNN)
- **Core Application**: Predict stress intensity factor (K1) directly from specimen images
- **Thesis Link**: Thesis calculates K1 analytically/via FEA; Grad project uses CNN to predict K1 from images
- **Innovation**: Physics-informed CNN for fracture mechanics parameter estimation

## Specimen Geometry
- Rectangular beam under 3-point bending
- Intentional V-notch creates stress concentration
- Crack propagates from V-notch under loading
- Visual variations: Crack depth, propagation angle, surface roughness

## CNN Application Areas

### 1. Classification Tasks
- Binary: Cracked / Uncracked
- Multi-class: No Crack → Minor → Moderate → Severe
- Probability output: Damage likelihood

### 2. Segmentation Tasks
- Pixel-level crack identification (U-Net, DeepLab)
- Crack geometry extraction
- Region of interest (ROI) isolation

### 3. Detection Tasks
- Bounding boxes around defects (Faster R-CNN, YOLO)
- Multi-defect specimens
- Precise localization

### 4. Regression Tasks
- Crack depth prediction (pixels → mm)
- Crack angle estimation
- Stress intensity factor (K_I) estimation from image

### 5. Time-Series Analysis
- Sequential image analysis during loading
- Damage progression tracking
- Remaining life prediction (RUL)

## Data Strategy
- **Source**: Experimental images or FEA simulations
- **Target**: 500-1000 labeled images
- **Augmentation**: Rotation, brightness, noise, crop
- **Split**: 70% train, 15% val, 15% test

## Model Architecture Recommendations
- **Transfer Learning**: ResNet-50, EfficientNet (Fast, effective)
- **From Scratch**: Custom CNN (More novel, requires more data)
- **Segmentation**: U-Net, DeepLab v3
- **Detection**: Faster R-CNN, YOLOv8

## Evaluation Metrics
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Segmentation: IoU, Dice coefficient
- Detection: mAP (mean Average Precision)
- Regression: MAE, RMSE, R² score

## Physics Integration
- Correlate CNN severity levels with estimated stress intensity factors
- Link image features to fracture mechanics parameters
- Validate CNN predictions against experimental failure data

## Grad Project Differentiation
1. Novel synthetic data generation (FEA-based training)
2. Physics-informed loss functions
3. Comparison with baseline methods (traditional image processing)
4. Real-specimen validation pipeline
5. Fracture mechanics parameter estimation from CNN features

## Week-by-Week Timeline
- **Week 15**: Data preparation, model training, baseline results
- **Week 16**: Refinement, comparison studies, written report

## Success Metrics
- **Accuracy**: Target 85-90% on test set
- **Generalization**: Works on different specimen batches
- **Speed**: Real-time inference (<100ms per image)
- **Correlation**: Strong link to fracture mechanics parameters
