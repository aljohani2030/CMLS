import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualLearningLoss(nn.Module):
    """
    Collaborative Mutual Learning Supervision Loss
    Implements equation: L_mutual = sum_i ||F_i - (1/2)*sum_{j!=i} alpha_ij * F_j||^2
    """
    def __init__(self, feature_dim, num_branches=3, learnable_alpha=True):
        super().__init__()
        self.num_branches = num_branches
        self.learnable_alpha = learnable_alpha
        
        if learnable_alpha:
            # Learnable weights alpha_ij for each pair
            self.alphas = nn.Parameter(torch.ones(num_branches, num_branches) * 0.5)
            # Mask to ignore self (alpha_ii)
            self.register_buffer('eye_mask', torch.eye(num_branches).bool())
    
    def forward(self, features):
        """
        Args:
            features: list of tensors [F1, F2, F3] each of shape (batch, feature_dim)
        Returns:
            mutual_loss: scalar tensor
        """
        assert len(features) == self.num_branches
        batch_size = features[0].shape[0]
        
        # Normalize features for stable training
        features_norm = [F.normalize(f, p=2, dim=-1) for f in features]
        
        total_loss = 0.0
        
        for i in range(self.num_branches):
            # Compute weighted average of other branches' features
            other_features = []
            other_weights = []
            
            for j in range(self.num_branches):
                if i != j:
                    if self.learnable_alpha:
                        weight = torch.sigmoid(self.alphas[i, j])  # constrain to [0,1]
                    else:
                        weight = 1.0 / (self.num_branches - 1)
                    
                    other_features.append(features_norm[j])
                    other_weights.append(weight)
            
            # Weighted sum of other features
            weights_tensor = torch.tensor(other_weights, device=features[0].device)
            weights_tensor = weights_tensor / weights_tensor.sum()  # normalize
            
            weighted_other = sum(w * f for w, f in zip(other_weights, other_features))
            
            # L2 distance between branch i and weighted other branches
            loss_i = F.mse_loss(features_norm[i], weighted_other)
            total_loss += loss_i
        
        return total_loss / self.num_branches


class CollaborativeLoss(nn.Module):
    """
    Combined loss: classification loss + mutual learning loss
    """
    def __init__(self, mutual_loss_weight=0.5, class_weight=None):
        super().__init__()
        self.mutual_loss_weight = mutual_loss_weight
        self.class_criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.mutual_criterion = MutualLearningLoss(feature_dim=512)
    
    def forward(self, predictions, features, targets):
        """
        Args:
            predictions: list of [pred1, pred2, pred3, fusion_pred] or single fusion_pred
            features: list of [F1, F2, F3] from each branch
            targets: ground truth labels (batch,)
        Returns:
            total_loss, class_loss, mutual_loss
        """
        # Classification loss (using fusion output)
        if isinstance(predictions, list):
            class_loss = self.class_criterion(predictions[-1], targets)  # fusion output
        else:
            class_loss = self.class_criterion(predictions, targets)
        
        # Mutual learning loss
        mutual_loss = self.mutual_criterion(features)
        
        # Total loss
        total_loss = class_loss + self.mutual_loss_weight * mutual_loss
        
        return total_loss, class_loss, mutual_loss


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for domain adaptation (optional)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None