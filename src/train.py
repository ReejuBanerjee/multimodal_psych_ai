# Author: Reeju Banerjee & Shourya
# Registration Number: RA2511003010548
# Project: Explainable Multimodal AI Framework - Advanced Training Script

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import PsychMultimodalDataset
from models import MultimodalPsychNet
from utils import FocalLoss  # Custom F1-boosting loss function

def main():
    # Setup absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    train_csv = os.path.join(project_root, 'data', 'tabular', 'train_data.csv')
    audio_dir = os.path.join(project_root, 'data', 'audio')
    vision_dir = os.path.join(project_root, 'data', 'vision')
    
    print("Initializing Training DataLoader...")
    train_dataset = PsychMultimodalDataset(train_csv, audio_dir, vision_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize multi-task model
    model = MultimodalPsychNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ==========================================
    # ADVANCED OBJECTIVE METRICS
    # ==========================================
    # Classification: Focal Loss (Boosts F1 by penalizing hard edge cases)
    criterion_class = FocalLoss(gamma=2.0)
    
    # Regression: Mean Squared Error
    criterion_reg = nn.MSELoss()
    
    # Scheduler: Automatically drops learning rate if loss stops improving
    # (Fixed for PyTorch 2.2+ by removing verbose=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    # ==========================================
    
    epochs = 15
    print(f"\nStarting Advanced Training Loop for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_class = 0
        total_samples = 0
        
        for batch_idx, (images, audios, tabulars, class_labels, reg_labels) in enumerate(train_loader):
            images = images.to(device)
            audios = audios.to(device)
            tabulars = tabulars.to(device)
            class_labels = class_labels.to(device)
            reg_labels = reg_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            class_preds, reg_preds = model(images, audios, tabulars)
            
            # Compute dual losses
            loss_class = criterion_class(class_preds, class_labels)
            loss_reg = criterion_reg(reg_preds, reg_labels)
            
            # Combined multi-task loss (0.1 scales regression to balance with classification loss)
            total_loss = loss_class + (0.1 * loss_reg)
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Track training accuracy
            _, predicted = torch.max(class_preds.data, 1)
            total_samples += class_labels.size(0)
            correct_class += (predicted == class_labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_class / total_samples
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
        
        # Step the scheduler dynamically based on this epoch's loss
        scheduler.step(epoch_loss)
        
    # Save the highly-tuned trained model weights
    model_save_path = os.path.join(project_root, 'psych_model_weights.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining Complete! Advanced weights successfully saved to: {model_save_path}")

if __name__ == "__main__":
    main()