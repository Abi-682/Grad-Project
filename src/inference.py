"""
Inference Pipeline for Manufacturing Defect Detection
Provides functions to:
1. Detect if specimen has defect (binary classification)
2. Identify crack shape type (C1-C14)
3. Predict K1 value (stress intensity factor)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
import json
from datetime import datetime

from config import (
    CRACK_SHAPES, K1_VALUES, IDX_TO_K1, CLASS_NAMES,
    IMAGE_SIZE, MEAN, STD, DEVICE,
    MODEL_BINARY_DETECTOR, MODEL_SHAPE_CLASSIFIER, MODEL_K1_PREDICTOR
)
from models import BinaryDefectDetector, ShapeClassifier, K1Predictor
from dataset import get_transforms


class DefectDetectionPipeline:
    """
    Complete 3-stage inference pipeline for defect detection and K1 prediction
    """
    
    def __init__(self, device=DEVICE):
        """Initialize models"""
        self.device = device
        
        # Load models
        self.binary_detector = BinaryDefectDetector(pretrained=False, num_classes=2)
        self.shape_classifier = ShapeClassifier(pretrained=False, num_classes=14)
        self.k1_predictor = K1Predictor(pretrained=False)
        
        # Load trained weights
        self.load_models()
        
        # Set to eval mode
        self.binary_detector.eval()
        self.shape_classifier.eval()
        self.k1_predictor.eval()
        
        # Get transforms
        self.transform = get_transforms(phase="val", image_size=IMAGE_SIZE)
        
    def load_models(self):
        """Load trained model weights"""
        print("Loading trained models...")
        
        try:
            state_dict = torch.load(MODEL_BINARY_DETECTOR, map_location=self.device)
            self.binary_detector.load_state_dict(state_dict)
            print(f"✓ Loaded binary detector from {MODEL_BINARY_DETECTOR}")
        except FileNotFoundError:
            print(f"⚠ Warning: Binary detector model not found at {MODEL_BINARY_DETECTOR}")
        
        try:
            state_dict = torch.load(MODEL_SHAPE_CLASSIFIER, map_location=self.device)
            self.shape_classifier.load_state_dict(state_dict)
            print(f"✓ Loaded shape classifier from {MODEL_SHAPE_CLASSIFIER}")
        except FileNotFoundError:
            print(f"⚠ Warning: Shape classifier model not found at {MODEL_SHAPE_CLASSIFIER}")
        
        try:
            state_dict = torch.load(MODEL_K1_PREDICTOR, map_location=self.device)
            self.k1_predictor.load_state_dict(state_dict)
            print(f"✓ Loaded K1 predictor from {MODEL_K1_PREDICTOR}")
        except FileNotFoundError:
            print(f"⚠ Warning: K1 predictor model not found at {MODEL_K1_PREDICTOR}")
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def detect_defect(self, image_tensor):
        """
        Stage 1: Binary Detection
        Returns: is_defect (bool), confidence (float)
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.binary_detector(image_tensor)
            probs = F.softmax(logits, dim=1)
            
            # Get prediction
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            
            is_defect = pred_class == 1  # 1 = Defect, 0 = No Defect
            
            return is_defect, confidence, pred_class
    
    def classify_crack_shape(self, image_tensor):
        """
        Stage 2: Shape Classification
        Returns: crack_shape (str), confidence (float), class_idx (int)
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.shape_classifier(image_tensor)
            probs = F.softmax(logits, dim=1)
            
            # Get prediction
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            
            crack_shape = CRACK_SHAPES[pred_class]
            
            return crack_shape, confidence, pred_class
    
    def predict_k1(self, image_tensor):
        """
        Stage 3: K1 Prediction
        Returns: k1_value (float)
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            k1_output = self.k1_predictor(image_tensor)
            k1_value = k1_output.item()
            
            return max(0, k1_value)  # Ensure non-negative
    
    def infer(self, image_path):
        """
        Complete inference pipeline
        
        Args:
            image_path: Path to specimen image
        
        Returns:
            Dictionary with results:
            {
                "image": image_path,
                "defect_detected": bool,
                "defect_confidence": float,
                "crack_shape": str (if defect),
                "crack_shape_confidence": float (if defect),
                "k1_value": float (if defect),
                "risk_level": str (if defect)
            }
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Stage 1: Binary Detection
        is_defect, det_confidence, det_class = self.detect_defect(image_tensor)
        
        result = {
            "image": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "defect_detected": is_defect,
            "defect_class": CLASS_NAMES[det_class],
            "defect_confidence": round(det_confidence, 4)
        }
        
        # If defect detected, proceed to classification and regression
        if is_defect:
            # Stage 2: Shape Classification
            crack_shape, clf_confidence, shape_idx = self.classify_crack_shape(image_tensor)
            
            # Stage 3: K1 Prediction
            k1_value = self.predict_k1(image_tensor)
            
            # Get K1 reference from lookup table
            k1_reference = K1_VALUES[crack_shape]
            
            # Determine risk level
            k1_ratio = k1_value / k1_reference if k1_reference > 0 else 0
            if k1_ratio > 0.8:
                risk_level = "CRITICAL"
            elif k1_ratio > 0.5:
                risk_level = "HIGH"
            elif k1_ratio > 0.2:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            result.update({
                "crack_shape": crack_shape,
                "crack_shape_confidence": round(clf_confidence, 4),
                "k1_predicted": round(k1_value, 2),
                "k1_reference": round(k1_reference, 2),
                "k1_ratio": round(k1_ratio, 4),
                "risk_level": risk_level
            })
        
        return result
    
    def infer_batch(self, image_paths):
        """Inference on multiple images"""
        results = []
        for img_path in image_paths:
            result = self.infer(img_path)
            results.append(result)
        
        return results


def format_result(result):
    """Pretty-print inference result"""
    print("\n" + "="*70)
    print(f"DEFECT DETECTION RESULT")
    print("="*70)
    print(f"Image: {result['image']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"\nStage 1 - Defect Detection:")
    print(f"  Status: {result['defect_class']}")
    print(f"  Confidence: {result['defect_confidence']:.2%}")
    
    if result['defect_detected']:
        print(f"\nStage 2 - Crack Shape Classification:")
        print(f"  Shape: {result['crack_shape']}")
        print(f"  Confidence: {result['crack_shape_confidence']:.2%}")
        
        print(f"\nStage 3 - K1 Prediction (Stress Intensity Factor):")
        print(f"  Predicted K1: {result['k1_predicted']} (N/mm^1.5)")
        print(f"  Reference K1: {result['k1_reference']} (N/mm^1.5)")
        print(f"  K1 Ratio: {result['k1_ratio']:.2%}")
        
        print(f"\nQUALITY ASSESSMENT:")
        print(f"  ⚠ RISK LEVEL: {result['risk_level']}")
        print(f"  ACTION: {'REJECT SPECIMEN' if result['risk_level'] in ['CRITICAL', 'HIGH'] else 'ACCEPT/REVIEW'}")
    else:
        print(f"\n✓ QUALITY DECISION: SPECIMEN IS SAFE - ACCEPT")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defect Detection Inference Pipeline")
    parser.add_argument("image_path", type=str, help="Path to specimen image")
    parser.add_argument("--output", type=str, default="result.json", 
                        help="Save results to JSON file")
    parser.add_argument("--save-result", action="store_true", help="Save result to JSON")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("\nInitializing Defect Detection Pipeline...")
    pipeline = DefectDetectionPipeline(device=DEVICE)
    
    # Run inference
    print(f"\nProcessing image: {args.image_path}")
    result = pipeline.infer(args.image_path)
    
    # Display result
    format_result(result)
    
    # Save if requested
    if args.save_result:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
