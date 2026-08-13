# Author: Reeju Banerjee
# Registration Number: RA2511003010548
# Project: Explainable Multimodal AI Framework

import os
import torch
from dataloader import PsychMultimodalDataset
from models import MultimodalPsychNet

def predict_single_sample(sample_index=0):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    test_csv = os.path.join(project_root, 'data', 'tabular', 'test_data.csv')
    audio_dir = os.path.join(project_root, 'data', 'audio')
    vision_dir = os.path.join(project_root, 'data', 'vision')
    weights_path = os.path.join(project_root, 'psych_model_weights.pth')
    
    if not os.path.exists(weights_path):
        print("[ERROR] Train the model first to generate weights!")
        return
        
    # Load dataset and fetch a specific sample
    dataset = PsychMultimodalDataset(test_csv, audio_dir, vision_dir)
    image, audio, tabular, true_class, true_regs = dataset[sample_index]
    
    # Add a batch dimension since models expect batches: [Channels, Height, Width] -> [1, Channels, Height, Width]
    image = image.unsqueeze(0)
    audio = audio.unsqueeze(0)
    tabular = tabular.unsqueeze(0)
    
    # Load model architecture and weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalPsychNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Move tensors to device and run inference
    image, audio, tabular = image.to(device), audio.to(device), tabular.to(device)
    
    with torch.no_grad():
        class_preds, reg_preds = model(image, audio, tabular)
        
        # Classification (Softmax probabilities)
        probabilities = torch.softmax(class_preds, dim=1)
        pred_class_idx = torch.argmax(probabilities, dim=1).item()
        
    # Mapping indices back to text labels
    idx_to_label = {0: 'Healthy', 1: 'Mild_Stress', 2: 'Moderate_Stress', 3: 'Severe_Stress'}
    pred_label = idx_to_label[pred_class_idx]
    true_label = idx_to_label[true_class.item()]
    
    reg_preds = reg_preds.squeeze().cpu().numpy()
    true_regs = true_regs.numpy()
    
    # Print Live Prediction Results
    print("="*50)
    print("          LIVE MULTIMODAL AI PREDICTION")
    print("="*50)
    print(f"Sample Index in Test Set: {sample_index}")
    print(f"Assigned Image File:      {dataset.data.iloc[sample_index]['Assigned_Image']}")
    print(f"Assigned Audio File:      {dataset.data.iloc[sample_index]['Assigned_Audio']}")
    print("-" * 50)
    print(f"Actual Mental Status:     {true_label}")
    print(f"Predicted Mental Status:  {pred_label} (Confidence: {probabilities[0][pred_class_idx].item()*100:.2f}%)")
    print("-" * 50)
    print("Continuous Severity Scores (Depression, Anxiety, Stress):")
    print(f"  -> Actual Scores:    {[round(score, 2) for score in true_regs]}পন্থী")
    print(f"  -> Predicted Scores: {[round(score, 2) for score in reg_preds]}")
    print("="*50)

if __name__ == "__main__":
    # You can change the index to test different profiles from your test set!
    predict_single_sample(sample_index=0)