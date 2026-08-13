# Author: Reeju Banerjee & Shourya
# Project: Explainable Multimodal AI Framework - Isolated Modality Testing (Ablation)

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import PsychMultimodalDataset
from models import MultimodalPsychNet
from sklearn.metrics import accuracy_score, f1_score

def evaluate_modality(model, test_loader, device, mode='all'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, audios, tabulars, class_labels, _ in test_loader:
            images = images.to(device)
            audios = audios.to(device)
            tabulars = tabulars.to(device)
            class_labels = class_labels.to(device)

            # --- ABLATION MASKS ---
            # Zero out the inputs we don't want to test
            if mode == 'vision':
                audios = torch.zeros_like(audios)
                tabulars = torch.zeros_like(tabulars)
            elif mode == 'audio':
                images = torch.zeros_like(images)
                tabulars = torch.zeros_like(tabulars)
            elif mode == 'tabular':
                images = torch.zeros_like(images)
                audios = torch.zeros_like(audios)

            # Forward pass
            class_preds, _ = model(images, audios, tabulars)
            _, predicted = torch.max(class_preds.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(class_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    return acc, f1

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    test_csv = os.path.join(project_root, 'data', 'tabular', 'test_data.csv')
    audio_dir = os.path.join(project_root, 'data', 'audio')
    vision_dir = os.path.join(project_root, 'data', 'vision')
    weights_path = os.path.join(project_root, 'psych_model_weights.pth')
    
    print("Loading Test Dataset...")
    test_dataset = PsychMultimodalDataset(test_csv, audio_dir, vision_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalPsychNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    print("\n" + "="*50)
    print("ISOLATED MODALITY PERFORMANCE (ABLATION STUDY)")
    print("="*50)
    
    modes = ['vision', 'audio', 'tabular', 'all']
    labels = ['Face-Only (CNN)', 'Voice-Only (Audio)', 'Telemetry-Only (Tabular)', 'Combined Fused (Overall)']
    
    for mode, label in zip(modes, labels):
        acc, f1 = evaluate_modality(model, test_loader, device, mode=mode)
        print(f"{label:<30} | Accuracy: {acc:.2f}% | F1-Score: {f1:.2f}%")
        
    print("="*50)

if __name__ == "__main__":
    main()