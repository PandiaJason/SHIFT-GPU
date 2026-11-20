"""
PICAN: Physics-Informed Cross-Attention Network with Meta-Calibrated Few-Shot Adaptation.
Novel AI model architecture designed for hardware-workload interaction modeling under covariate shift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkloadEncoder(nn.Module):
    def __init__(self, num_cont_features, scenario_cardinality=3, model_type_cardinality=25, embed_dim=64):
        super().__init__()
        self.scenario_embed = nn.Embedding(scenario_cardinality, 8)
        self.model_type_embed = nn.Embedding(model_type_cardinality, 16)
        total_in = num_cont_features + 8 + 16
        self.mlp = nn.Sequential(
            nn.Linear(total_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, cont_x, scenario_idx, model_type_idx):
        e_scen = self.scenario_embed(scenario_idx)
        e_type = self.model_type_embed(model_type_idx)
        h = torch.cat([cont_x, e_scen, e_type], dim=-1)
        return self.mlp(h)

class HardwareEncoder(nn.Module):
    def __init__(self, num_cont_features, mem_type_cardinality=5, arch_cardinality=8, embed_dim=64):
        super().__init__()
        self.mem_type_embed = nn.Embedding(mem_type_cardinality, 8)
        self.arch_embed = nn.Embedding(arch_cardinality, 8)
        total_in = num_cont_features + 8 + 8
        self.mlp = nn.Sequential(
            nn.Linear(total_in, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, cont_g, mem_type_idx, arch_idx):
        e_mem = self.mem_type_embed(mem_type_idx)
        e_arch = self.arch_embed(arch_idx)
        h = torch.cat([cont_g, e_mem, e_arch], dim=-1)
        return self.mlp(h)

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
    1. Deterministic Microarchitectural Roofline Base
    2. Deep Cross-Attention Interaction Latent Space
    3. Multiplicative Exponential Epistemic Residual Head
    4. Heteroscedastic Aleatoric Uncertainty Output
    """
    def __init__(self, num_workload_cont, num_hardware_cont, embed_dim=64, num_heads=4, target="itl"):
        super().__init__()
        self.target = target
        self.embed_dim = embed_dim
        
        self.workload_encoder = WorkloadEncoder(num_workload_cont, embed_dim=embed_dim)
        self.hardware_encoder = HardwareEncoder(num_hardware_cont, embed_dim=embed_dim)
        
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
        
        # Residual and Uncertainty Heads
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        # Initialize residual to start near empirical median ratio (ln 1.6 ≈ 0.47 for ITL, -0.4 for power)
        nn.init.zeros_(self.residual_head[-1].weight)
        if target == "itl":
            nn.init.constant_(self.residual_head[-1].bias, 0.47)
        else:
            nn.init.constant_(self.residual_head[-1].bias, 0.0)

        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.constant_(self.uncertainty_head[-1].bias, -2.0) # start with small log-variance

    def forward(self, cont_w, scen_idx, mtype_idx, cont_h, mem_idx, arch_idx, physics_base):
        """
        Forward pass with Physics residual binding.
        physics_base: Roofline latency for ITL (seconds) OR TDP for Power (Watts)
        """
        w = self.workload_encoder(cont_w, scen_idx, mtype_idx).unsqueeze(1) # [B, 1, d]
        h = self.hardware_encoder(cont_h, mem_idx, arch_idx).unsqueeze(1) # [B, 1, d]
        
        # Cross Attention: Q=w, K=h, V=h
        attn_out, _ = self.cross_attention(query=w, key=h, value=h)
        z = self.norm1(w + attn_out)
        z = self.norm2(z + self.feed_forward(z)).squeeze(1) # [B, d]
        
        # FiLM adaptation
        z = self.film(z)
        
        raw_res = self.residual_head(z).squeeze(-1) # [B]
        log_var = self.uncertainty_head(z).squeeze(-1) # [B]
        
        if self.target == "itl":
            # Multiplicative residual bound: T_hat = T_roof * exp(res)
            # res >= 0 via softplus enforces physical lower bound T_hat >= T_roof
            res = F.softplus(raw_res)
            pred = physics_base * torch.exp(res)
        else:
            res = raw_res
            pred = physics_base * torch.sigmoid(res)
            
        return pred, log_var, res

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
