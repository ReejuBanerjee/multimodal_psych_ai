# Author: Reeju Banerjee
# Registration Number: RA2511003010548
# Project: Explainable Multimodal AI Framework

import torch
import torch.nn as nn

class MultimodalPsychNet(nn.Module):
    def __init__(self):
        super(MultimodalPsychNet, self).__init__()
        
        # Vision Branch
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU()
        )
        
        # Audio Branch
        self.audio_network = nn.Sequential(
            nn.Linear(40, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Tabular Branch
        self.tabular_fnn = nn.Sequential(
            nn.Linear(18, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Late Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Multi-Task Output Heads
        self.classification_head = nn.Linear(256, 4) 
        self.regression_head = nn.Linear(256, 3)     

    def forward(self, image, audio, tabular):
        v_feat = self.vision_cnn(image)
        a_feat = self.audio_network(audio)
        t_feat = self.tabular_fnn(tabular)
        
        fused = torch.cat((v_feat, a_feat, t_feat), dim=1)
        fused_out = self.fusion(fused)
        
        class_out = self.classification_head(fused_out)
        reg_out = self.regression_head(fused_out)
        
        return class_out, reg_out