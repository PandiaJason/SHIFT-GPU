"""
Generates publication-quality Figure 1 (Modular Software Architecture & Data-Flow Diagram)
for SHIFT-GPU, matching the exact visual style, pastel palette, rounded boxes,
and typography of SIMPAT Figure 1.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIG_DIR = "paper/figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("experiments/figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Palette definitions matching SIMPAT Figure 1
fc_orchestrator = "#fff2cc"  # Warm Pastel Gold
ec_orchestrator = "#d6b656"

fc_physics = "#dae8fc"       # Soft Steel Blue
ec_physics = "#6c8ebf"

fc_neural = "#d5e8d4"        # Soft Mint / Sage Green
ec_neural = "#82b366"

fc_conformal = "#e1d5e7"     # Soft Lavender
ec_conformal = "#9673a6"

fc_active = "#ffe6cc"        # Soft Peach / Orange
ec_active = "#d79b00"

fc_decision = "#f8cecc"      # Soft Rose / Coral
ec_decision = "#b85450"

def draw_box(x, y, w, h, fc, ec, title, subtitle_lines):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.8,rounding_size=2.5",
        facecolor=fc, edgecolor=ec, linewidth=1.6
    )
    ax.add_patch(box)
    
    # Title
    ax.text(x + w/2, y + h - 2.8, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#111111", family="sans-serif")
    
    # Subtitle items
    y_offset = y + h - 6.2
    for line in subtitle_lines:
        ax.text(x + w/2, y_offset, line, ha="center", va="top",
                fontsize=8.5, color="#222222", family="sans-serif")
        y_offset -= 3.2

# 1. Top Orchestrator Box
draw_box(
    x=26, y=82, w=48, h=15,
    fc=fc_orchestrator, ec=ec_orchestrator,
    title="SHIFT-GPU Orchestrator",
    subtitle_lines=[
        "Workload Specification x = (m, s)  |  Candidate Hardware Pool G",
        "Target SLA Latency T_max  |  Admission Control & Profiling Coordinator"
    ]
)

# 2. Left Box: Microarchitectural Roofline Base
draw_box(
    x=5, y=50, w=40, h=18,
    fc=fc_physics, ec=ec_physics,
    title="Deterministic Microarchitectural Base",
    subtitle_lines=[
        "Analytical Roofline Latency Lower Bound:",
        "T_roof = max( 2*N_params / (B_mem * B(s)),  2*N_params / F_peak )",
        "Operational Serving Scenario Scaling B(s)",
        "Physical Upper Power Bound: P <= TDP(g)"
    ]
)

# 3. Right Box: PICAN Neural Core
draw_box(
    x=55, y=50, w=40, h=18,
    fc=fc_neural, ec=ec_neural,
    title="PICAN: Physics-Informed Cross-Attention",
    subtitle_lines=[
        "Workload Encoder Ew(x) & Hardware Encoder Eh(g)",
        "Bipartite Cross-Attention Core: Q=Wq*w_x, K=Wk*h_g, V=Wv*h_g",
        "Multiplicative Exponential Residual: T_hat = T_roof * exp(r_theta)",
        "Heteroscedastic Aleatoric Log-Variance Head: s_theta = log(sigma^2)"
    ]
)

# 4. Middle Box: Shift-Aware Conformal Risk Calibrator
draw_box(
    x=20, y=24, w=60, h=16,
    fc=fc_conformal, ec=ec_conformal,
    title="Covariate Shift Conformal Risk Calibrator",
    subtitle_lines=[
        "Microarchitectural Mahalanobis Discrepancy: d_shift(g, D_train)",
        "Covariate-Shift Scaled Conformal Quantile: q_shift = q_(1-alpha) * (1 + gamma * d_shift)",
        "Certified SLA Upper Latency Bound: T_upper(x, g) = T_hat(x, g) + q_shift * sigma(x, g)"
    ]
)

# 5. Bottom Left Box: ROAA Active Probing
draw_box(
    x=5, y=2, w=42, h=14,
    fc=fc_active, ec=ec_active,
    title="Regret-Optimal Active Acquisition (ROAA)",
    subtitle_lines=[
        "Boundary Probing Utility alpha_ROAA(x) (k in {1, 3, 5} budget)",
        "Focus on Decision Boundary Proximity: exp(-(T_hat - T_max)^2 / 2*sigma^2)",
        "Few-Shot FiLM Latent Modulation: z_adapted = gamma_k * z + beta_k"
    ]
)

# 6. Bottom Right Box: SLA-Constrained Hardware Recommendation
draw_box(
    x=53, y=2, w=42, h=14,
    fc=fc_decision, ec=ec_decision,
    title="SLA-Constrained Hardware Selector",
    subtitle_lines=[
        "Certified SLA Feasibility Filter: T_upper(x, g) <= T_max",
        "Min Operational Power Draw: g* = argmin P_hat(x, g)",
        "Near-Oracle Regret Minimization (<2.1% SLA Violations)"
    ]
)

# ARROWS & LABELS
def draw_arrow(x1, y1, x2, y2, label="", label_pos=(0.5, 0.5), label_ha="center", label_va="center"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6, mutation_scale=14)
    )
    if label:
        lx = x1 + (x2 - x1) * label_pos[0]
        ly = y1 + (y2 - y1) * label_pos[1]
        ax.text(lx, ly, label, fontsize=8, color="#1c3b5e", weight="bold",
                ha=label_ha, va=label_va,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))

# Orchestrator to Roofline & PICAN
draw_arrow(40, 82, 25, 68, "Hardware g & Workload x", (0.5, 0.4), "right")
draw_arrow(60, 82, 75, 68, "Hardware g & Workload x", (0.5, 0.4), "left")

# Roofline & PICAN to Conformal Calibrator
draw_arrow(25, 50, 40, 40, "Physics Floor T_roof", (0.5, 0.5), "right")
draw_arrow(75, 50, 60, 40, "Residual r_theta, Uncertainty sigma", (0.5, 0.5), "left")

# Conformal to Decision Selector
draw_arrow(50, 24, 74, 16, "Certified SLA Bound T_upper", (0.5, 0.5), "left")

# Conformal to ROAA (if unfamiliar GPU)
draw_arrow(35, 24, 26, 16, "Shift d_shift > tau (Few-Shot Trigger)", (0.5, 0.5), "right")

# ROAA to PICAN (FiLM Feedback)
ax.annotate(
    "", xy=(95, 59), xytext=(47, 9),
    arrowprops=dict(arrowstyle="-|>", color="#d79b00", lw=1.6, linestyle="--",
                    connectionstyle="arc3,rad=-0.35", mutation_scale=14)
)
ax.text(92, 33, "FiLM Modulation\n[gamma_k, beta_k]", fontsize=8, color="#b45f06", weight="bold",
        ha="center", bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))

# ROAA to Hardware Selector
draw_arrow(47, 9, 53, 9, "Calibrated P, T", (0.5, 0.5), "center", "bottom")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_architecture.png"), dpi=300, bbox_inches="tight")
plt.savefig("experiments/figures/fig1_architecture.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 1 (Modular Software Architecture) successfully generated!")
