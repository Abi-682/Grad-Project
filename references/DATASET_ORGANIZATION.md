# Dataset Organization & Image Reference Guide

## Project Dataset Structure

```
data/
├── raw/
│   ├── no_defect/                          # Stage 1: Specimens WITHOUT cracks
│   │   ├── specimen_001.png
│   │   ├── specimen_002.png
│   │   └── ...
│   │
│   ├── defect_cracked/                     # Cracked specimens with OTHER defects
│   │   ├── image_with_manufacturing_issue.png
│   │   └── ...
│   │
│   └── crack_shapes/                       # 14 crack shapes (proper 3-point bending)
│       ├── Shape_01_VNotch/
│       │   ├── specimen_1.png
│       │   ├── specimen_2.png
│       │   └── ...
│       ├── Shape_02_Rounded/
│       │   └── ...
│       ├── Shape_03_Blunt/
│       │   └── ...
│       └── Shape_14_xxx/
│           └── ...
│
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
└── K1_values.csv                           # Master label file
    (image_filename, category, k1_value, crack_shape)
```

## Dataset Categories Explained

### **Category 1: No Defect (Proper Specimens)**
- **Images**: Rectangular beam with proper 3-point bending geometry
- **Characteristics**: 
  - ✓ 1 load point at top (centered)
  - ✓ 2 support points at bottom
  - ✓ NO crack visible
- **Purpose**: Stage 1 training (negative class for binary detection)
- **Count Target**: 200-400 images

### **Category 2: Defect - Cracked with Manufacturing Issues**
- **Example Image 1 (provided)**: 
  - ✓ Crack present (V-notch visible)
  - ✗ Missing 2 support points at bottom (malformed geometry)
  - **Status**: DEFECT (both crack + manufacturing issue)
- **Purpose**: Real-world defects (specimens that failed quality control)
- **Count Target**: 50-100 images
- **Note**: These represent manufacturing failures beyond expected crack propagation

### **Category 3: Crack Shapes (14 Types - Proper Geometry)**
- **Example Image 2 provided**: Shows proper 3-point bending geometry
- **Specifications for each of 14 shapes**:
  - ✓ All specimens have proper 3-point bending (1 top + 2 bottom)
  - ✓ All cracks centered at same location
  - ✓ All cracks same dimensions
  - ✓ Same loading conditions
  - **Variations**: Only the CRACK SHAPE differs (V-notch, Rounded, Blunt, etc.)
- **Purpose**: Stage 2 & 3 training (classification + K1 regression)
- **Count Target**: 30-50 images × 14 shapes = 420-700 images
- **Expected Crack Shapes**:
  1. V-Notch
  2. Rounded
  3. Blunt
  4. [Shape 4]
  5. [Shape 5]
  6. ... (through Shape 14)

---

## Image Classification Guide

### **Classification Decision Tree**
```
Does specimen have proper 3-point geometry?
│
├─ NO (missing/malformed support points)
│  └─ CATEGORY: "defect_cracked"
│     (Training set for real-world edge cases)
│
└─ YES (1 top load + 2 bottom supports visible)
   │
   ├─ Is there a visible crack?
   │  │
   │  ├─ NO
   │  │  └─ CATEGORY: "no_defect"
   │  │     (Stage 1: Binary detection negative class)
   │  │
   │  └─ YES
   │     └─ CATEGORY: "crack_shapes/Shape_XX_[NAME]/"
   │        (Stage 2 & 3: Classification + K1 regression)
   │        Record K1 value from thesis calculations
```

---

## Images Provided (References)

### **Image 1: Cracked with Manufacturing Defect**
- **File**: [IMAGE_1_DEFECT_CRACKED.png](specimen_images/IMAGE_1_DEFECT_CRACKED.png)
- **Category**: defect_cracked
- **Observations**:
  - ✓ Crack visible (V-notch shape)
  - ✗ Missing 2 bottom support points
  - **Classification**: DEFECT (manufacturing + crack issue)
- **Purpose**: Example of real-world defective specimen
- **Usage**: Training reference for Stage 1 (edge case - has crack but malformed)

### **Image 2: Proper Geometry, No Crack**
- **File**: [IMAGE_2_NO_DEFECT.png](specimen_images/IMAGE_2_NO_DEFECT.png)
- **Category**: no_defect
- **Observations**:
  - ✓ Proper 3-point bending visible (1 top, 2 bottom)
  - ✓ No crack
  - **Classification**: NO DEFECT (clean specimen)
- **Purpose**: Example of proper uncracked specimen
- **Usage**: Training reference for Stage 1 (negative class)

---

## Next Steps: 14 Crack Shapes

### **Expected Upload Structure**
When you upload the 14 crack shapes, please organize as:
```
Shape_1_VNotch: [Images showing V-notch cracks]
Shape_2_Rounded: [Images showing rounded cracks]
Shape_3_Blunt: [Images showing blunt cracks]
... through Shape_14_[Description]
```

### **For Each Shape, Provide**:
- **Images**: 30-50 specimen photographs/FEA renders
- **Associated K1 values**: From your thesis calculations
- **Crack specifications**: Depth, angle, dimensions (for reference)

### **Format for K1 Labels**:
```csv
image_filename,crack_shape,k1_value,specimen_id,load_level
Shape_01_VNotch_spec001.png,V-Notch,1.245,1,1000N
Shape_01_VNotch_spec002.png,V-Notch,1.502,1,1500N
Shape_02_Rounded_spec001.png,Rounded,0.987,2,1000N
...
```

---

## Data Augmentation Strategy

For each provided image, we'll generate additional training data through:
- **Rotation**: ±5-15°
- **Brightness**: ±10-20%
- **Gaussian Noise**: σ = 0.01-0.02
- **Zoom**: 0.95-1.05x
- **Result**: ~2-3× dataset multiplication

Example:
```
1 original image
  ├─ Original (1)
  ├─ Rotated ±5° (2)
  ├─ Rotated ±10° (2)
  ├─ Bright +10% (1)
  ├─ Bright -10% (1)
  ├─ Noise applied (1)
  └─ Combinations (multiple)
  = ~10 effective training samples per original
```

---

## Master Labels File (Will Create)

After all images received, we'll create:
```
K1_DATASET_MASTER.csv
├── image_id
├── filename
├── category (no_defect / defect_cracked / crack_shapes)
├── crack_shape (if applicable: V-Notch, Rounded, ..., or NULL)
├── k1_value (if applicable: numerical, or NULL)
├── specimen_id
├── load_level
├── notes
└── split_assignment (train / val / test)
```

---

## Storage & Version Control

All images saved in two locations:
1. **Working data**: `data/raw/[category]/` (for training pipeline)
2. **References**: `references/specimen_images/` (for documentation & correlation)

This allows easy cross-referencing while keeping data organized for ML pipeline.
