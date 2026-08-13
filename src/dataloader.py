# Author: Reeju Banerjee
# Registration Number: RA2511003010548
# Project: Explainable Multimodal AI Framework

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from torchvision import transforms
from PIL import Image

class PsychMultimodalDataset(Dataset):
    def __init__(self, csv_path, audio_dir, vision_dir):
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.vision_dir = vision_dir
        
        # Transform 1: Process 48x48 grayscale images
        self.vision_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Transform 2: Extract 40 Mel-Frequency Cepstral Coefficients (MFCCs)
        self.audio_transform = torchaudio.transforms.MFCC(
            sample_rate=16000, 
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- 1. Tabular Features ---
        numeric_features = torch.tensor(row.iloc[0:18].values.astype('float32'))
        
        # Labels for Classification & Regression
        class_label = torch.tensor(row['Class_Label'], dtype=torch.long)
        reg_labels = torch.tensor([
            row['Depression_Score'], 
            row['Anxiety_Score'], 
            row['Stress_Score']
        ], dtype=torch.float32)
        
        # --- 2. Vision Features ---
        img_path = os.path.join(self.vision_dir, row['Assigned_Image'])
        image = Image.open(img_path).convert('L')
        vision_tensor = self.vision_transform(image)
        
        # --- 3. Audio Features ---
        audio_path = os.path.join(self.audio_dir, row['Assigned_Audio'])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Fix: Force stereo to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if the audio isn't 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            
        audio_tensor = self.audio_transform(waveform)
        
        # Pool the audio over time to get a flat 40-dimension tensor
        audio_tensor = torch.mean(audio_tensor, dim=2).squeeze() 
        
        return vision_tensor, audio_tensor, numeric_features, class_label, reg_labels

# Quick test block
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    train_csv_path = os.path.join(project_root, 'data', 'tabular', 'train_data.csv')
    audio_dir_path = os.path.join(project_root, 'data', 'audio')
    vision_dir_path = os.path.join(project_root, 'data', 'vision')
    
    if not os.path.exists(train_csv_path):
        print(f"[ERROR] Could not find the CSV at: {train_csv_path}")
    else:
        print(f"Loading data from: {train_csv_path}")
        dataset = PsychMultimodalDataset(train_csv_path, audio_dir_path, vision_dir_path)
        v, a, n, c, r = dataset[0]
        print(f"Success! Shapes -> Vision: {v.shape}, Audio: {a.shape}, Numeric: {n.shape}")