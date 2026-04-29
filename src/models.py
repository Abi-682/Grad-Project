"""
CNN Model Definitions for Manufacturing Defect Detection
Includes Stage 1 (Binary Detection), Stage 2 (Shape Classification), and Stage 3 (K1 Prediction)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BinaryDefectDetector(nn.Module):
    """
    Stage 1: Binary Classification
    Input: Specimen image
    Output: 2-class probability (No Defect vs Defect)
    """
    def __init__(self, pretrained=True, num_classes=2):
        super(BinaryDefectDetector, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.backbone(x)


class ShapeClassifier(nn.Module):
    """
    Stage 2: Multi-class Classification
    Input: Specimen image (with crack)
    Output: 14-class probability (C1 through C14)
    """
    def __init__(self, pretrained=True, num_classes=14):
        super(ShapeClassifier, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.backbone(x)


class K1Predictor(nn.Module):
    """
    Stage 3: Regression
    Input: Specimen image (with identified crack shape)
    Output: Continuous K1 value (Stress Intensity Factor)
    """
    def __init__(self, pretrained=True):
        super(K1Predictor, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final layer for regression (single continuous output)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Single output for K1 value
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.backbone(x)


class MultiTaskDefectDetector(nn.Module):
    """
    Multi-Task Learning Model (Optional Advanced Version)
    Simultaneously performs:
    - Binary detection (defect present?)
    - Shape classification (which crack type?)
    - K1 prediction (severity value)
    """
    def __init__(self, pretrained=True, num_crack_shapes=14):
        super(MultiTaskDefectDetector, self).__init__()
        
        # Shared backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        
        # Remove original FC layer
        self.backbone.fc = nn.Identity()
        
        # Task 1: Binary detection head
        self.detection_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification
        )
        
        # Task 2: Shape classification head
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_crack_shapes)  # 14-class classification
        )
        
        # Task 3: K1 regression head
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single continuous output
        )
    
    def forward(self, x):
        """Forward pass returns outputs from all three heads"""
        features = self.backbone(x)
        
        detection_output = self.detection_head(features)      # (batch, 2)
        classification_output = self.classification_head(features)  # (batch, 14)
        regression_output = self.regression_head(features)    # (batch, 1)
        
        return detection_output, classification_output, regression_output


def get_model(model_type="binary", pretrained=True, num_classes=2):
    """
    Factory function to create models
    
    Args:
        model_type: "binary", "classifier", "regressor", or "multitask"
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of classes (for applicable models)
    
    Returns:
        Model instance
    """
    if model_type == "binary":
        return BinaryDefectDetector(pretrained=pretrained, num_classes=num_classes)
    elif model_type == "classifier":
        return ShapeClassifier(pretrained=pretrained, num_classes=num_classes)
    elif model_type == "regressor":
        return K1Predictor(pretrained=pretrained)
    elif model_type == "multitask":
        return MultiTaskDefectDetector(pretrained=pretrained, num_crack_shapes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model instantiation
    print("Testing model architectures...\n")
    
    # Test Binary Detector
    print("Stage 1: Binary Defect Detector")
    detector = BinaryDefectDetector()
    x = torch.randn(2, 3, 224, 224)
    y = detector(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape} (2 classes: No Defect, Defect)")
    print(f"  Trainable params: {sum(p.numel() for p in detector.parameters() if p.requires_grad):,}\n")
    
    # Test Shape Classifier
    print("Stage 2: Shape Classifier")
    classifier = ShapeClassifier()
    y = classifier(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape} (14 classes: C1-C14)")
    print(f"  Trainable params: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}\n")
    
    # Test K1 Predictor
    print("Stage 3: K1 Predictor")
    predictor = K1Predictor()
    y = predictor(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape} (continuous K1 value)")
    print(f"  Trainable params: {sum(p.numel() for p in predictor.parameters() if p.requires_grad):,}\n")
    
    # Test Multi-Task Model
    print("Advanced: Multi-Task Detector")
    multitask = MultiTaskDefectDetector(num_crack_shapes=14)
    det, clf, reg = multitask(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Detection output: {det.shape}")
    print(f"  Classification output: {clf.shape}")
    print(f"  Regression output: {reg.shape}")
    print(f"  Trainable params: {sum(p.numel() for p in multitask.parameters() if p.requires_grad):,}")
