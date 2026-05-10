"""
Dataset and DataLoader classes for Manufacturing Defect Detection
Handles image loading, preprocessing, and augmentation
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DefectDataset(Dataset):
    """
    Custom PyTorch Dataset for manufacturing defect detection
    Handles loading images and corresponding labels
    """
    
    def __init__(self, image_dir, labels_file, transform=None, task="binary"):
        """
        Args:
            image_dir: Directory containing all images
            labels_file: CSV file with columns: image_name, label, k1_value
            transform: Albumentations transform pipeline
            task: "binary" (defect/no-defect), "classifier" (C1-C14), or "regression" (K1 value)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.task = task
        self.images = []
        self.labels = []
        self.k1_values = []
        
        # Load labels from CSV
        with open(labels_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    img_name, label, k1_value = parts[0], parts[1], float(parts[2])
                    img_path = self.image_dir / img_name
                    
                    if img_path.exists():
                        self.images.append(str(img_path))
                        self.labels.append(int(label))
                        self.k1_values.append(k1_value)
        
        print(f"Loaded {len(self.images)} images from {image_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Load and return image and label"""
        # Load image
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default to tensor conversion
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        label = self.labels[idx]
        k1_value = self.k1_values[idx]
        
        if self.task == "binary":
            return image, label
        elif self.task == "classifier":
            return image, label
        elif self.task == "regression":
            return image, torch.tensor([k1_value], dtype=torch.float32)
        else:
            return image, label, k1_value


def get_transforms(phase="train", image_size=224):
    """
    Get albumentations transform pipeline
    
    Args:
        phase: "train" (with augmentation) or "val"/"test" (without)
        image_size: Target image size
    
    Returns:
        Composed transform pipeline
    """
    
    if phase == "train":
        transforms_list = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.1),
            A.Rotate(limit=15, p=0.7),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Affine(scale=(0.9, 1.1), p=0.5),  # Changed from A.Zoom to A.Affine with scale
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    else:  # val or test
        transforms_list = [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]
    
    return A.Compose(transforms_list)


def create_dataloaders(data_dir, labels_file, batch_size=32, num_workers=4, 
                       task="binary", image_size=224):
    """
    Create train and validation dataloaders
    
    Args:
        data_dir: Root data directory
        labels_file: Path to labels CSV file
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        task: Type of task ("binary", "classifier", "regression")
        image_size: Target image size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    train_transform = get_transforms(phase="train", image_size=image_size)
    val_transform = get_transforms(phase="val", image_size=image_size)
    
    # Create datasets
    train_dataset = DefectDataset(
        image_dir=data_dir,
        labels_file=labels_file,
        transform=train_transform,
        task=task
    )
    
    val_dataset = DefectDataset(
        image_dir=data_dir,
        labels_file=labels_file,
        transform=val_transform,
        task=task
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Dataset module loaded successfully")
    print("Available functions:")
    print("  - DefectDataset: Custom dataset class")
    print("  - get_transforms: Get augmentation pipeline")
    print("  - create_dataloaders: Create train/val dataloaders")
