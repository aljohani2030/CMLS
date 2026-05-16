import torch
import torch.nn as nn
from .cnn_branch import CNNBranch
from .transformer_branch import TransformerBranch
from .mamba_branch import MambaBranch
from .attention_fusion import AdaptiveFeatureFusion
from .mutual_learning_loss import CollaborativeLoss

class CMLSModel(nn.Module):
    """
    Complete CMLS (Collaborative Mutual Learning Supervision) Model
    Integrates CNN, Transformer, and Mamba branches with mutual learning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Three heterogeneous branches
        self.cnn_branch = CNNBranch(
            backbone=config.CNN_BACKBONE,
            pretrained=config.CNN_PRETRAINED,
            feature_dim=512
        )
        
        self.transformer_branch = TransformerBranch(
            model_name=config.TRANSFORMER_MODEL,
            pretrained=config.TRANSFORMER_PRETRAINED,
            feature_dim=512
        )
        
        self.mamba_branch = MambaBranch(
            d_model=config.MAMBA_D_MODEL,
            n_layers=config.MAMBA_N_LAYERS,
            d_state=config.MAMBA_D_STATE,
            feature_dim=512
        )
        
        # Adaptive feature fusion
        self.fusion = AdaptiveFeatureFusion(
            feature_dim=512,
            num_branches=3,
            dropout=config.FUSION_DROPOUT
        )
        
        # Mutual learning loss (enables branch collaboration)
        self.collaborative_loss = CollaborativeLoss(
            mutual_loss_weight=config.MUTUAL_LOSS_WEIGHT
        )
        
        # Store intermediate features for visualization
        self.intermediate_features = {'cnn': None, 'trans': None, 'mamba': None}
        
    def forward(self, x, return_features=False, return_attention=False):
        """
        Forward pass with optional returns for visualization
        
        Args:
            x: input images (batch, 3, H, W)
            return_features: if True, return branch features
            return_attention: if True, return attention weights
        Returns:
            logits: (batch, 2)
            (optional) features: list of branch features
            (optional) attention_weights: tuple for visualization
        """
        # Extract features from each branch
        cnn_features = self.cnn_branch(x)
        trans_features, trans_attention = self.transformer_branch(x)
        mamba_features, mamba_attention = self.mamba_branch(x)
        
        # Store for loss computation
        branch_features = [cnn_features, trans_features, mamba_features]
        self.intermediate_features = {
            'cnn': cnn_features,
            'trans': trans_features,
            'mamba': mamba_features
        }
        
        # Adaptive fusion
        logits, fusion_features, fusion_attention = self.fusion(branch_features)
        
        # Prepare returns
        outputs = [logits]
        
        if return_features:
            outputs.append(branch_features)
        
        if return_attention:
            attention_data = {
                'trans_attention': trans_attention,
                'mamba_attention': mamba_attention,
                'fusion_branch_weights': fusion_attention[1],
                'fusion_channel_weights': fusion_attention[0]
            }
            outputs.append(attention_data)
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def compute_loss(self, predictions, features, targets):
        """
        Compute collaborative loss (classification + mutual learning)
        """
        total_loss, class_loss, mutual_loss = self.collaborative_loss(
            predictions, features, targets
        )
        return total_loss, class_loss, mutual_loss
    
    def get_features_for_visualization(self, x, layer='middle'):
        """
        Extract intermediate features for visualization (Grad-CAM, feature maps)
        """
        cnn_features = self.cnn_branch.get_intermediate_features(x, layer)
        _, trans_attention = self.transformer_branch(x)
        _, mamba_attention = self.mamba_branch(x)
        
        return {
            'cnn_maps': cnn_features,
            'trans_attention': trans_attention,
            'mamba_attention': mamba_attention
        }