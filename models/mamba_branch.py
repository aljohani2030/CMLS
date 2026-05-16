import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model core module (Mamba)
    Implements the core selective scan mechanism
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(d_state, self.d_inner)
        
        # Out projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x, res = x_and_res.chunk(2, dim=-1)
        
        # 1D convolution
        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]
        x = F.silu(x_conv)
        
        # SSM computation
        delta, B, C = self.x_proj(x).split([self.d_state, self.d_state, 1], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        # Simplified selective scan (full implementation would be more complex)
        # This is a pytorch approximation of the selective scan
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        y = []
        
        for t in range(seq_len):
            delta_t = delta[:, t, :]
            B_t = B[:, t, :].unsqueeze(-1)
            C_t = C[:, t, :].unsqueeze(1)
            
            h = h + delta_t.unsqueeze(-1) * (x[:, t, :].unsqueeze(-1) * B_t - h)
            y_t = (C_t @ h).squeeze(-1)
            y.append(y_t)
        
        y = torch.stack(y, dim=1)
        
        # Output
        output = F.silu(y + res)
        output = self.out_proj(output)
        
        return output


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection"""
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        return x + residual


class MambaBranch(nn.Module):
    """
    Mamba branch for capturing sequential temporal artifacts
    Uses selective state space modeling
    """
    def __init__(self, d_model=512, n_layers=4, d_state=16, feature_dim=512, img_size=224, patch_size=16):
        super().__init__()
        
        # Patch embedding (convert image to sequence)
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Temporal attention for capturing frame-to-frame artifacts
        self.temporal_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Projection to common feature space
        self.projection = nn.Sequential(
            nn.Linear(d_model, 512),
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
        batch = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, d_model, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, d_model)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Temporal attention across patches (captures spatial-temporal relationships)
        x, temporal_weights = self.temporal_attention(x, x, x)
        
        # Global pooling
        pooled = x.mean(dim=1)  # (batch, d_model)
        
        # Project to common space
        output = self.projection(pooled)
        
        return output, temporal_weights