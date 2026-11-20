"""
PICAN: Physics-Informed Cross-Attention Network with Meta-Calibrated Few-Shot Adaptation.
Novel AI model architecture designed for hardware-workload interaction modeling under covariate shift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkloadEncoder(nn.Module):
    def __init__(self, in_features, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, w_feats):
        return self.mlp(w_feats)

class HardwareEncoder(nn.Module):
    def __init__(self, in_features, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, h_feats):
        return self.mlp(h_feats)

class FiLMAdapter(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # Initialized to identity (gamma=1, beta=0)
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        return self.gamma * x + self.beta

class PICAN(nn.Module):
    """
    Physics-Informed Cross-Attention Network.
    Decomposes inference prediction into:
    1. Deterministic Microarchitectural Roofline Base (or Physical TDP Base)
    2. Deep Cross-Attention Interaction Latent Space
    3. Multiplicative Exponential Epistemic Residual Head
    4. Heteroscedastic Aleatoric Uncertainty Output
    """
    def __init__(self, w_dim, h_dim, embed_dim=64, num_heads=4, target="itl"):
        super().__init__()
        self.target = target
        self.embed_dim = embed_dim
        
        self.workload_encoder = WorkloadEncoder(w_dim, embed_dim=embed_dim)
        self.hardware_encoder = HardwareEncoder(h_dim, embed_dim=embed_dim)
        
        # Microarchitectural Cross-Attention (Workload queries attend to Hardware keys/values)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # FiLM Layer for few-shot adaptation
        self.film = FiLMAdapter(embed_dim)
        
        # Residual Head
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Uncertainty Head (log-variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.constant_(self.uncertainty_head[-1].bias, -2.0)

    def forward(self, w_feats, h_feats, physics_base=None):
        qw = self.workload_encoder(w_feats).unsqueeze(1) # [B, 1, d]
        kh = self.hardware_encoder(h_feats).unsqueeze(1) # [B, 1, d]
        
        attn_out, _ = self.cross_attention(query=qw, key=kh, value=kh)
        z = self.norm1(qw + attn_out)
        z = self.norm2(z + self.feed_forward(z)).squeeze(1) # [B, d]
        
        # FiLM adaptation
        z = self.film(z)
        
        raw_res = self.residual_head(z).squeeze(-1) # [B]
        log_var = self.uncertainty_head(z).squeeze(-1) # [B]
        
        if self.target == "itl":
            res = torch.clamp(raw_res, min=0.0) # Physical floor constraint
            pred = physics_base * torch.exp(res) if physics_base is not None else torch.exp(raw_res)
        else:
            res = torch.sigmoid(raw_res)
            pred = physics_base * res if physics_base is not None else res
            
        return pred, log_var, raw_res

    def freeze_backbone_for_few_shot(self):
        """Freezes all representation weights, exposing only the FiLM adapter for few-shot calibration."""
        for name, param in self.named_parameters():
            if "film" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
