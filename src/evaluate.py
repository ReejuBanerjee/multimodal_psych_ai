# Author: Reeju Banerjee
# Registration Number: RA2511003010548
# Project: Explainable Multimodal AI Framework

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             mean_absolute_error, mean_squared_error, r2_score, explained_variance_score)
from dataloader import PsychMultimodalDataset
from models import MultimodalPsychNet

def evaluate_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    test_csv = os.path.join(project_root, 'data', 'tabular', 'test_data.csv')
    audio_dir = os.path.join(project_root, 'data', 'audio')
    vision_dir = os.path.join(project_root, 'data', 'vision')
    
    print("Loading unseen Test Data...")
    test_dataset = PsychMultimodalDataset(test_csv, audio_dir, vision_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalPsychNet().to(device)
    
    weights_path = os.path.join(project_root, 'psych_model_weights.pth')
    if not os.path.exists(weights_path):
        print("[ERROR] Train the model first to generate weights!")
        return
        
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    all_true_classes, all_pred_classes = [], []
    all_true_regs, all_pred_regs = [], []
    
    print("Running Inference on unseen Test set...\n")
    with torch.no_grad():
        for images, audios, tabulars, class_labels, reg_labels in test_loader:
            images = images.to(device)
            audios = audios.to(device)
            tabulars = tabulars.to(device)
            
            class_preds, reg_preds = model(images, audios, tabulars)
            _, predicted = torch.max(class_preds.data, 1)
            
            all_true_classes.extend(class_labels.cpu().numpy())
            all_pred_classes.extend(predicted.cpu().numpy())
            all_true_regs.extend(reg_labels.cpu().numpy())
            all_pred_regs.extend(reg_preds.cpu().numpy())
            
    print("="*45)
    print("      OBJECTIVE 1: CLASSIFICATION METRICS")
    print("="*45)
    print(f"Accuracy:        {accuracy_score(all_true_classes, all_pred_classes):.4f}")
    print(f"Precision (Mac): {precision_score(all_true_classes, all_pred_classes, average='macro', zero_division=0):.4f}")
    print(f"Recall (Mac):    {recall_score(all_true_classes, all_pred_classes, average='macro', zero_division=0):.4f}")
    print(f"F1-Score (Mac):  {f1_score(all_true_classes, all_pred_classes, average='macro', zero_division=0):.4f}")
    print(f"F1-Score (Wgt):  {f1_score(all_true_classes, all_pred_classes, average='weighted', zero_division=0):.4f}")
    
    print("\n" + "="*45)
    print("      OBJECTIVE 2: REGRESSION METRICS")
    print("="*45)
    true_regs_np = np.array(all_true_regs)
    pred_regs_np = np.array(all_pred_regs)
    
    print(f"MAE:             {mean_absolute_error(true_regs_np, pred_regs_np):.4f}")
    print(f"MSE:             {mean_squared_error(true_regs_np, pred_regs_np):.4f}")
    print(f"RMSE:            {np.sqrt(mean_squared_error(true_regs_np, pred_regs_np)):.4f}")
    print(f"R2 Score:        {r2_score(true_regs_np, pred_regs_np):.4f}")
    print(f"Explained Var:   {explained_variance_score(true_regs_np, pred_regs_np):.4f}")
    print("="*45)

if __name__ == "__main__":
    evaluate_model()