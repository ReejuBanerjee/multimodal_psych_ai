# Author: Reeju Banerjee & Shourya
# Project: Explainable Multimodal AI Framework - Training Utilities

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss dynamically scales down the loss for easy, highly confident predictions,
    forcing the model to focus entirely on hard, misclassified edge cases.
    Perfect for improving F1-Scores in imbalanced or highly overlapping datasets.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # gamma controls how much the loss drops for easy examples (default 2.0 is standard)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross entropy loss (don't reduce yet)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # Calculate the probability of the true class (pt)
        pt = torch.exp(-ce_loss)
        
        # Apply the Focal Loss formula: (1 - pt)^gamma * CE_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss