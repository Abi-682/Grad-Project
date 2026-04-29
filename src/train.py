"""
Training script for Manufacturing Defect Detection CNN
Trains 3-stage pipeline:
- Stage 1: Binary defect detector
- Stage 2: Shape classifier
- Stage 3: K1 regressor
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY,
    DEVICE, MODEL_BINARY_DETECTOR, MODEL_SHAPE_CLASSIFIER, MODEL_K1_PREDICTOR
)
from models import BinaryDefectDetector, ShapeClassifier, K1Predictor
from dataset import create_dataloaders


class Trainer:
    """Base trainer class"""
    
    def __init__(self, model, criterion, optimizer, device=DEVICE):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for images, labels in pbar:
            images = images.to(self.device)
            
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.device)
            else:
                labels = torch.tensor(labels).to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            
            if outputs.dim() > 1 and outputs.size(1) > 1:
                preds = torch.argmax(outputs, dim=1)
                if labels.dim() > 1:
                    labels_class = torch.argmax(labels, dim=1)
                else:
                    labels_class = labels
                total_correct += (preds == labels_class).sum().item()
            
            total_samples += images.size(0)
            pbar.set_postfix({"loss": loss.item():.4f})
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                else:
                    labels = torch.tensor(labels).to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    preds = torch.argmax(outputs, dim=1)
                    if labels.dim() > 1:
                        labels_class = torch.argmax(labels, dim=1)
                    else:
                        labels_class = labels
                    total_correct += (preds == labels_class).sum().item()
                
                total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_acc
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")


def train_stage1_detector(data_dir, labels_file, num_epochs=50):
    """
    Train Stage 1: Binary Defect Detector
    Detects: No Defect vs Defect Present
    """
    print("\n" + "="*70)
    print("TRAINING STAGE 1: BINARY DEFECT DETECTOR")
    print("="*70)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir, labels_file, batch_size=BATCH_SIZE, task="binary"
    )
    
    # Model, loss, optimizer
    model = BinaryDefectDetector(pretrained=True, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    trainer = Trainer(model, criterion, optimizer, device=DEVICE)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch + 1)
        val_loss, val_acc = trainer.validate(val_loader)
        
        trainer.history["train_loss"].append(train_loss)
        trainer.history["train_acc"].append(train_acc)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(MODEL_BINARY_DETECTOR)
    
    return trainer


def train_stage2_classifier(data_dir, labels_file, num_epochs=50):
    """
    Train Stage 2: Shape Classifier
    Classifies: C1-C14 (14 different crack shapes)
    """
    print("\n" + "="*70)
    print("TRAINING STAGE 2: CRACK SHAPE CLASSIFIER")
    print("="*70)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir, labels_file, batch_size=BATCH_SIZE, task="classifier"
    )
    
    # Model, loss, optimizer
    model = ShapeClassifier(pretrained=True, num_classes=14).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    trainer = Trainer(model, criterion, optimizer, device=DEVICE)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch + 1)
        val_loss, val_acc = trainer.validate(val_loader)
        
        trainer.history["train_loss"].append(train_loss)
        trainer.history["train_acc"].append(train_acc)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(MODEL_SHAPE_CLASSIFIER)
    
    return trainer


def train_stage3_regressor(data_dir, labels_file, num_epochs=50):
    """
    Train Stage 3: K1 Regressor
    Predicts: Continuous K1 value (Stress Intensity Factor)
    """
    print("\n" + "="*70)
    print("TRAINING STAGE 3: K1 REGRESSOR")
    print("="*70)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir, labels_file, batch_size=BATCH_SIZE, task="regression"
    )
    
    # Model, loss, optimizer
    model = K1Predictor(pretrained=True).to(DEVICE)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    trainer = Trainer(model, criterion, optimizer, device=DEVICE)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, _ = trainer.train_epoch(train_loader, epoch + 1)
        val_loss, _ = trainer.validate(val_loader)
        
        trainer.history["train_loss"].append(train_loss)
        trainer.history["val_loss"].append(val_loss)
        
        # For regression, compute RMSE
        train_rmse = np.sqrt(train_loss)
        val_rmse = np.sqrt(val_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train MSE: {train_loss:.4f}, RMSE: {train_rmse:.4f} | "
              f"Val MSE: {val_loss:.4f}, RMSE: {val_rmse:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(MODEL_K1_PREDICTOR)
    
    return trainer


def plot_training_history(history, save_path="training_history.png"):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(history["val_loss"], label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training History - Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot (if available)
    if history["train_acc"]:
        axes[1].plot(history["train_acc"], label="Train Acc", marker='o')
        axes[1].plot(history["val_acc"], label="Val Acc", marker='s')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training History - Accuracy")
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Defect Detection Models")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--labels-file", type=str, required=True, help="Labels CSV file")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 0], default=0,
                        help="Stage to train (1/2/3), 0 for all")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    
    args = parser.parse_args()
    
    print(f"\nDefect Detection CNN Training Pipeline")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Train requested stages
    if args.stage == 0 or args.stage == 1:
        train_stage1_detector(args.data_dir, args.labels_file, num_epochs=args.epochs)
    
    if args.stage == 0 or args.stage == 2:
        train_stage2_classifier(args.data_dir, args.labels_file, num_epochs=args.epochs)
    
    if args.stage == 0 or args.stage == 3:
        train_stage3_regressor(args.data_dir, args.labels_file, num_epochs=args.epochs)
    
    print("\n✓ Training complete!")
