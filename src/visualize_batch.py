# Author: Reeju Banerjee & Shourya
# Project: Explainable Multimodal AI Framework - Batch Visualizer

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import PsychMultimodalDataset
from models import MultimodalPsychNet

def generate_inference_grid(num_samples=8):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    test_csv = os.path.join(project_root, 'data', 'tabular', 'test_data.csv')
    audio_dir = os.path.join(project_root, 'data', 'audio')
    vision_dir = os.path.join(project_root, 'data', 'vision')
    weights_path = os.path.join(project_root, 'psych_model_weights.pth')
    
    # Load Test Dataset
    test_dataset = PsychMultimodalDataset(test_csv, audio_dir, vision_dir)
    
    # Use DataLoader to grab a random batch
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=True)
    images, audios, tabulars, true_classes, _ = next(iter(test_loader))
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalPsychNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    images_dev = images.to(device)
    audios_dev = audios.to(device)
    tabulars_dev = tabulars.to(device)
    
    # Run Inference
    with torch.no_grad():
        class_preds, _ = model(images_dev, audios_dev, tabulars_dev)
        probabilities = torch.softmax(class_preds, dim=1)
        confs, pred_classes = torch.max(probabilities, 1)
        
    idx_to_label = {0: 'Healthy', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    
    # Setup Matplotlib Plot (2 rows, 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Multimodal AI: Live Test Set Inference', fontsize=18, fontweight='bold', y=0.95)
    
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Un-normalize image for displaying ([ -1, 1 ] -> [ 0, 1 ])
        img = images[i].squeeze().numpy()
        img = (img * 0.5) + 0.5
        
        # Display the 48x48 face crop
        ax.imshow(img, cmap='gray')
        
        # Get labels
        true_label = idx_to_label[true_classes[i].item()]
        pred_label = idx_to_label[pred_classes[i].item()]
        conf_score = confs[i].item() * 100
        
        # Color code text: Green if correct, Red if wrong
        color = 'green' if true_label == pred_label else 'red'
        
        title = f"True: {true_label}\nPred: {pred_label} ({conf_score:.1f}%)"
        ax.set_title(title, color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot so you can put it in your slide deck
    save_path = os.path.join(project_root, 'inference_grid.png')
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Inference grid saved to {save_path}")
    
    # Show the plot live
    plt.show()

if __name__ == "__main__":
    generate_inference_grid(num_samples=8)