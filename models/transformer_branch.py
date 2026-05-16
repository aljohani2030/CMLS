import torch
import torch.nn as nn
import timm

class TransformerBranch(nn.Module):
    """
    Transformer branch for capturing global inconsistent patterns
    Uses Vision Transformer architecture
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, feature_dim=512):
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        if 'vit' in model_name:
            self.feature_dim_vit = self.model.num_features  # 768 for base
            self.is_vit = True
        else:
            self.feature_dim_vit = self.model.num_features
            self.is_vit = False
        
        # Attention-based feature refinement
        self.attention_refinement = nn.MultiheadAttention(
            embed_dim=self.feature_dim_vit,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.feature_dim_vit)
        
        # Projection to common feature space
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim_vit, 512),
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
        # Extract patch embeddings and pass through transformer
        features = self.model.forward_features(x)  # (batch, num_patches, dim)
        
        # Self-attention refinement
        attended, attention_weights = self.attention_refinement(features, features, features)
        attended = self.layer_norm(attended + features)  # residual connection
        
        # Global pooling (use CLS token for ViT, mean pooling for others)
        if self.is_vit:
            cls_token = attended[:, 0, :]  # (batch, dim)
            pooled = cls_token
        else:
            pooled = attended.mean(dim=1)  # (batch, dim)
        
        # Project to common space
        output = self.projection(pooled)
        
        return output, attention_weights  # return attention weights for visualization
    
    def get_attention_maps(self, x):
        """Extract attention maps for Grad-CAM visualization"""
        with torch.no_grad():
            features = self.model.forward_features(x)
            _, attention_weights = self.attention_refinement(features, features, features)
        return attention_weights