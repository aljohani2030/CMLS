import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitationAttention(nn.Module):
    """
    Channel attention mechanism for feature fusion
    Implements Equations 11-13 from the paper
    """
    def __init__(self, feature_dim, reduction_ratio=16):
        super().__init__()
        self.feature_dim = feature_dim
        reduced_dim = max(feature_dim // reduction_ratio, 16)
        
        # Squeeze: global average pooling
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: two-layer FC network
        self.excitation = nn.Sequential(
            nn.Linear(feature_dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Args:
            features: (batch, n_branches, feature_dim)
        Returns:
            attended: (batch, feature_dim) - weighted sum of features
            weights: (batch, n_branches) - attention weights per branch
        """
        # Squeeze operation (Eq 11)
        squeezed = features.mean(dim=1)  # (batch, feature_dim)
        
        # Excitation operation (Eq 12)
        attention_scores = self.excitation(squeezed)  # (batch, feature_dim)
        
        # Apply attention to each branch (Eq 13)
        attended = (features * attention_scores.unsqueeze(1)).sum(dim=1)
        
        # Also compute branch-level importance weights (for interpretability)
        branch_importance = F.softmax(features.mean(dim=-1), dim=-1)  # (batch, n_branches)
        
        return attended, attention_scores, branch_importance


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive fusion module with learnable weights per branch
    """
    def __init__(self, feature_dim=512, num_branches=3, dropout=0.2):
        super().__init__()
        
        self.num_branches = num_branches
        
        # Learnable branch weights (initialized equally)
        self.branch_weights = nn.Parameter(torch.ones(num_branches) / num_branches)
        
        # Channel attention module
        self.channel_attention = SqueezeExcitationAttention(feature_dim)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, 2)  # binary: real vs fake
        
    def forward(self, branch_features):
        """
        Args:
            branch_features: list of [F_cnn, F_trans, F_mamba] each (batch, feature_dim)
        Returns:
            logits: (batch, 2)
            fusion_features: (batch, feature_dim)
            attention_weights: tuple for visualization
        """
        # Stack features
        stacked = torch.stack(branch_features, dim=1)  # (batch, 3, feature_dim)
        
        # Apply channel attention (Eq 11-13)
        attended, channel_weights, branch_importance = self.channel_attention(stacked)
        
        # Apply fusion layer
        fusion_features = self.fusion_layer(attended)
        
        # Classification
        logits = self.classifier(fusion_features)
        
        return logits, fusion_features, (channel_weights, branch_importance)