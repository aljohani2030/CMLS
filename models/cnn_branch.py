import torch
import torch.nn as nn
import torchvision.models as models

class CNNBranch(nn.Module):
    """
    CNN branch for capturing high-frequency forgery traces
    Uses progressive convolutional layers with attention
    """
    def __init__(self, backbone='resnet50', pretrained=True, feature_dim=512):
        super().__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            # Remove classification head
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 2048
            
        elif backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_extractor = self.backbone.features
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 1536
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Progressive feature refinement
        self.progressive_conv = nn.Sequential(
            nn.Conv2d(in_features, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Channel attention (Squeeze-and-Excitation)
        self.se_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )
        
        # Final projection to common feature space
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: input images (batch, 3, H, W)
        Returns:
            features: (batch, feature_dim)
        """
        # Extract features from backbone
        features = self.feature_extractor(x)  # (batch, C, H', W')
        
        # Progressive convolution refinement
        refined = self.progressive_conv(features)
        
        # Channel attention
        attention_weights = self.se_attention(refined).unsqueeze(-1).unsqueeze(-1)
        attended = refined * attention_weights
        
        # Global pooling
        pooled = self.avg_pool(attended).flatten(1)  # (batch, 512)
        
        # Project to common space
        output = self.projection(pooled)
        
        return output
    
    def get_intermediate_features(self, x, layer='middle'):
        """For visualization and interpretability"""
        features = self.feature_extractor(x)
        
        if layer == 'early':
            return features[:, :64, :, :]  # early conv features
        elif layer == 'middle':
            refined = self.progressive_conv(features)
            return refined
        else:  # late
            attention_weights = self.se_attention(features)
            return attention_weights