#!/usr/bin/env python3
"""
Demo: Manufacturing Defect Detection CNN
Quick demonstration of the complete 3-stage pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import K1_VALUES, CRACK_SHAPES, IDX_TO_K1
import json


def demo_config():
    """Demo: Show configuration"""
    print("\n" + "="*70)
    print("DEMO: Configuration & K1 Lookup Table")
    print("="*70)
    
    print("\nAvailable Crack Shapes (C1-C14):")
    print("-" * 70)
    print(f"{'Shape':<10} {'K1 Value (N/mm^1.5)':<25} {'Description':<30}")
    print("-" * 70)
    
    for i, (shape_id, k1_val) in enumerate(K1_VALUES.items(), 1):
        print(f"{shape_id:<10} {k1_val:<25.2f} Crack shape variant #{i}")
    
    print("\n✓ Configuration loaded successfully")


def demo_models():
    """Demo: Show model information"""
    print("\n" + "="*70)
    print("DEMO: Model Architectures")
    print("="*70)
    
    try:
        import torch
        from models import (
            BinaryDefectDetector, ShapeClassifier, K1Predictor, 
            MultiTaskDefectDetector
        )
        
        print("\nStage 1: Binary Defect Detector")
        print("-" * 70)
        model1 = BinaryDefectDetector(pretrained=False, num_classes=2)
        total_params = sum(p.numel() for p in model1.parameters())
        trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output: Binary classification (2 classes)")
        
        print("\nStage 2: Crack Shape Classifier")
        print("-" * 70)
        model2 = ShapeClassifier(pretrained=False, num_classes=14)
        total_params = sum(p.numel() for p in model2.parameters())
        trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output: 14-class classification (C1-C14)")
        
        print("\nStage 3: K1 Regressor")
        print("-" * 70)
        model3 = K1Predictor(pretrained=False)
        total_params = sum(p.numel() for p in model3.parameters())
        trainable_params = sum(p.numel() for p in model3.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output: Continuous K1 value (regression)")
        
        print("\nAdvanced: Multi-Task Detector")
        print("-" * 70)
        model_mt = MultiTaskDefectDetector(pretrained=False, num_crack_shapes=14)
        total_params = sum(p.numel() for p in model_mt.parameters())
        trainable_params = sum(p.numel() for p in model_mt.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output: 3 tasks simultaneously")
        print(f"    - Binary classification (2 classes)")
        print(f"    - Shape classification (14 classes)")
        print(f"    - K1 regression (continuous)")
        
        print("\n✓ All models initialized successfully")
        
    except Exception as e:
        print(f"⚠ Error: {e}")
        print("Make sure to run from project root with proper imports")


def demo_data_pipeline():
    """Demo: Show data pipeline"""
    print("\n" + "="*70)
    print("DEMO: Data Pipeline & Augmentation")
    print("="*70)
    
    try:
        from dataset import get_transforms
        
        print("\nTraining Augmentation Pipeline:")
        print("-" * 70)
        transform_train = get_transforms(phase="train", image_size=224)
        print("  ✓ Resize to 224x224")
        print("  ✓ Horizontal flip (10%)")
        print("  ✓ Rotation ±15°")
        print("  ✓ Gaussian noise")
        print("  ✓ Random brightness/contrast")
        print("  ✓ Zoom 0.9-1.1x")
        print("  ✓ Normalize (ImageNet mean/std)")
        
        print("\nValidation Pipeline:")
        print("-" * 70)
        transform_val = get_transforms(phase="val", image_size=224)
        print("  ✓ Resize to 224x224")
        print("  ✓ Normalize (ImageNet mean/std)")
        print("  ✓ No augmentation (deterministic)")
        
        print("\n✓ Data pipeline configured")
        
    except Exception as e:
        print(f"⚠ Error: {e}")


def demo_inference_logic():
    """Demo: Show inference logic"""
    print("\n" + "="*70)
    print("DEMO: Inference Pipeline Logic")
    print("="*70)
    
    print("\n3-Stage Inference Pipeline:")
    print("-" * 70)
    
    print("\nSTAGE 1: Binary Detection")
    print("  Input: Specimen image (224x224)")
    print("  Process: ResNet-50 → Binary classifier")
    print("  Output: Defect/No Defect + Confidence")
    print("  Decision: Is there a crack?")
    
    print("\nSTAGE 2: Shape Classification (if defect detected)")
    print("  Input: Same specimen image")
    print("  Process: ResNet-50 → 14-class classifier")
    print("  Output: Shape class (C1-C14) + Confidence")
    print("  Decision: Which crack geometry?")
    
    print("\nSTAGE 3: K1 Prediction (if defect detected)")
    print("  Input: Same specimen image")
    print("  Process: ResNet-50 → Regression head")
    print("  Output: Continuous K1 value")
    print("  Decision: What is severity (K1)?")
    
    print("\nFINAL OUTPUT: Quality Assessment")
    print("  ├─ Defect Status: Yes/No")
    print("  ├─ Crack Shape: C1-C14")
    print("  ├─ K1 Predicted: Numerical value")
    print("  ├─ K1 Reference: From lookup table")
    print("  ├─ K1 Ratio: Predicted / Reference")
    print("  └─ Risk Level: LOW / MEDIUM / HIGH / CRITICAL")
    
    print("\n✓ Inference pipeline logic defined")


def demo_expected_output():
    """Demo: Show example inference output"""
    print("\n" + "="*70)
    print("DEMO: Expected Inference Output Example")
    print("="*70)
    
    example_result = {
        "image": "data/raw/crack_shapes/C5_specimen_042.png",
        "timestamp": "2026-04-28T14:32:15.789012",
        "defect_detected": True,
        "defect_class": "Defect",
        "defect_confidence": 0.9823,
        "crack_shape": "C5",
        "crack_shape_confidence": 0.8901,
        "k1_predicted": 10156.34,
        "k1_reference": 10312.00,
        "k1_ratio": 0.9849,
        "risk_level": "HIGH"
    }
    
    print("\nExample Result (JSON):")
    print("-" * 70)
    print(json.dumps(example_result, indent=2))
    
    print("\nInterpretation:")
    print("-" * 70)
    print(f"  Image: {example_result['image']}")
    print(f"  ✓ Defect DETECTED with {example_result['defect_confidence']:.1%} confidence")
    print(f"  ➜ Crack shape identified as: {example_result['crack_shape']}")
    print(f"  ➜ Confidence in shape: {example_result['crack_shape_confidence']:.1%}")
    print(f"  ➜ Predicted K1: {example_result['k1_predicted']:.2f}")
    print(f"  ➜ Reference K1: {example_result['k1_reference']:.2f}")
    print(f"  ➜ K1 Ratio: {example_result['k1_ratio']:.2%} of reference")
    print(f"  ⚠ RISK LEVEL: {example_result['risk_level']}")
    print(f"  ⚠ ACTION: REJECT SPECIMEN (High risk)")


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Manufacturing Defect Detection CNN" + " "*19 + "║")
    print("║" + " "*21 + "Quick Start Demo" + " "*32 + "║")
    print("╚" + "="*68 + "╝")
    
    # Run demos
    demo_config()
    demo_models()
    demo_data_pipeline()
    demo_inference_logic()
    demo_expected_output()
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    print("\n✓ Configuration: K1 lookup table for C1-C14")
    print("✓ Models: 3-stage ResNet-50 pipeline")
    print("✓ Data: Augmentation & preprocessing ready")
    print("✓ Inference: Complete pipeline implemented")
    print("✓ Output: JSON results with risk assessment")
    
    print("\n" + "-"*70)
    print("NEXT STEPS:")
    print("-"*70)
    print("\n1. Prepare your data:")
    print("   - Organize images into data/raw/ directories")
    print("   - Create K1_DATASET_MASTER.csv labels file")
    
    print("\n2. Install dependencies:")
    print("   cd c:\\Soft\\MI\\Grad-Project")
    print("   uv sync")
    
    print("\n3. Train the models:")
    print("   python src/train.py --data-dir data/raw --labels-file data/K1_DATASET_MASTER.csv")
    
    print("\n4. Run inference:")
    print("   python src/inference.py path/to/specimen_image.png --save-result")
    
    print("\n5. See detailed guide in QUICK_START.md")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
