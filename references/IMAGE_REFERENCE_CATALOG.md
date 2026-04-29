# Image Reference Catalog

## Images Provided in Chat

### Image 1: Cracked Specimen with Manufacturing Defect
**Status**: ✓ RECEIVED & SAVED FOR REFERENCE
**Category**: defect_cracked
**Filename**: image_1_defect_cracked.png

**Visual Analysis**:
- ✓ **Crack visible**: Clear V-notch crack at center of beam
- ✗ **Geometry issue**: Missing 2 support points at bottom
- **Top load point**: Visible (load applied from above)
- **Bottom supports**: NOT present (malformed)
- **Severity**: HIGH (both crack + missing supports)
- **Classification for ML**: DEFECT (Stage 1 output = 1)

**Engineering Notes**:
- This represents a failed specimen in manufacturing QC
- Should be REJECTED before fracture testing
- Good example for "real-world defect" training data
- Example of when specimen fails on multiple criteria

**Use Case**: 
- Train CNN to detect malformed specimens
- Edge case: has crack but invalid geometry

---

### Image 2: Proper Specimen Geometry (No Crack)
**Status**: ✓ RECEIVED & SAVED FOR REFERENCE
**Category**: no_defect
**Filename**: image_2_no_defect.png

**Visual Analysis**:
- ✓ **Geometry correct**: All 3 points of bending visible
  - Top: Load point (centered)
  - Bottom: 2 support points (properly spaced)
- ✓ **No crack**: Clean specimen surface, no visible fracture
- ✓ **Ready for testing**: Passes visual QC inspection
- **Severity**: SAFE (no defects)
- **Classification for ML**: NO DEFECT (Stage 1 output = 0)

**Engineering Notes**:
- Ideal specimen geometry for 3-point bending test
- Can proceed to mechanical testing
- Serves as positive example for "no defect" training
- K1 value would be: N/A (no crack = K1 = 0 or undefined)

**Use Case**:
- Train CNN binary detector (negative class)
- Example of acceptable specimens

---

## Summary of Images So Far

| Image | Category | Has Crack? | Valid Geometry? | Status |
|-------|----------|-----------|-----------------|--------|
| Image 1 | defect_cracked | ✓ YES (V-notch) | ✗ NO (missing supports) | DEFECT |
| Image 2 | no_defect | ✗ NO | ✓ YES (3-point visible) | NO DEFECT |

---

## Awaiting: 14 Crack Shape Images

**Next Upload Expected**:
- Shape 1: V-Notch (similar to Image 1 crack shape, but with proper geometry)
- Shape 2: Rounded
- Shape 3: Blunt
- Shapes 4-14: [To be specified]

**Per Shape, Expected**:
- Multiple specimens with same crack shape
- All with proper 3-point bending geometry
- Associated K1 values from thesis
- Variations in load level or specimen batch

**Data Correlation**:
- All Shape images will have proper geometry (unlike Image 1)
- Each shape will have calculated K1 value
- CNN will learn to distinguish shapes 1-14
- CNN will predict K1 for each shape

---

## File Organization Reference

**Current Saved Structure**:
```
references/
├── specimen_images/
│   ├── image_1_defect_cracked.png        [SAVED]
│   ├── image_2_no_defect.png              [SAVED]
│   ├── Shape_01_VNotch/                  [AWAITING]
│   ├── Shape_02_Rounded/                 [AWAITING]
│   ├── Shape_03_Blunt/                   [AWAITING]
│   └── ...
│
└── [Other reference docs already saved]
```

**Training Data Structure** (to be populated):
```
data/raw/
├── no_defect/
│   └── [Will organize Image 2 type specimens here]
├── defect_cracked/
│   └── [Will organize Image 1 type specimens here]
└── crack_shapes/
    ├── Shape_01_VNotch/                 [AWAITING]
    ├── Shape_02_Rounded/                [AWAITING]
    └── ...                               [AWAITING]
```

---

## Next Actions

1. ✓ Dataset structure created
2. ✓ Image 1 & 2 reviewed and categorized
3. ⏳ **WAITING**: Upload 14 crack shape images
4. ⏳ **WAITING**: K1 values for each crack shape
5. ⏳ Then: Organize into train/val/test splits
6. ⏳ Then: Build CNN training pipeline

