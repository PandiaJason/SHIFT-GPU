#!/usr/bin/env python3
"""
Generate a Publication-Grade System Architecture & Data-Flow Diagram for SHIFT-GPU.
Includes embedded micro-diagrams:
- Mini Roofline Plot (memory vs compute bound)
- Bipartite Cross-Attention Heatmap (workload tokens vs hardware features)
- Mahalanobis Covariate Shift Manifold (training GPUs vs unseen target)
- Conformal Prediction Interval & Empirical Upper Latency Bound
- ROAA Boundary Utility Curve peaking at T_max

Author: Jason Pandian (Nehru Institute of Technology)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

FIG_DIR = "paper/figures"
EXP_FIG_DIR = "experiments/figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EXP_FIG_DIR, exist_ok=True)

# Set up canvas
fig = plt.figure(figsize=(15.5, 8.8), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Typography & Palette
font_family = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

c_dark_navy = "#0f172a"
c_blue_border = "#2563eb"
c_blue_bg = "#eff6ff"
c_teal_border = "#0d9488"
c_teal_bg = "#f0fdfa"
c_purple_border = "#7c3aed"
c_purple_bg = "#f5f3ff"
c_amber_border = "#d97706"
c_amber_bg = "#fffbeb"
c_green_border = "#16a34a"
c_green_bg = "#f0fdf4"
c_gray_border = "#94a3b8"
c_gray_bg = "#f8fafc"

def draw_container(x, y, w, h, bg_color, border_color, title="", subtitle="", badge=""):
    # Container background with subtle border
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.5,rounding_size=1.8",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=1
    )
    ax.add_patch(rect)
    
    # Title header bar
    if title:
        header_h = 4.2
        header_rect = FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle="round,pad=0.5,rounding_size=1.8",
            facecolor=border_color, edgecolor="none", zorder=2
        )
        ax.add_patch(header_rect)
        # Flatten bottom corners of header
        header_flat = patches.Rectangle(
            (x, y + h - header_h), w, 2.0,
            facecolor=border_color, edgecolor="none", zorder=2
        )
        ax.add_patch(header_flat)
        
        ax.text(x + 1.8, y + h - 2.1, title, fontsize=10.5, fontweight="bold",
                color="white", family=font_family, va="center", zorder=3)
        
        if badge:
            badge_box = FancyBboxPatch(
                (x + w - 18.5, y + h - 3.4), 17.0, 2.6,
                boxstyle="round,pad=0.2,rounding_size=0.8",
                facecolor="white", edgecolor="none", zorder=3
            )
            ax.add_patch(badge_box)
            ax.text(x + w - 10.0, y + h - 2.1, badge, fontsize=7.5, fontweight="bold",
                    color=border_color, family=font_family, ha="center", va="center", zorder=4)

    if subtitle:
        ax.text(x + 2.0, y + h - 6.2, subtitle, fontsize=8.0, color="#475569",
                family=font_family, va="center", zorder=3)

# ==============================================================================
# TOP BANNER: CLUSTER ORCHESTRATION & ADMISSION CONTROLLER
# ==============================================================================
banner_rect = FancyBboxPatch(
    (2, 92.5), 96, 6.2,
    boxstyle="round,pad=0.4,rounding_size=1.2",
    facecolor="#1e293b", edgecolor="#0f172a", linewidth=1.2, zorder=2
)
ax.add_patch(banner_rect)
ax.text(4, 95.6, "SHIFT-GPU: Physics-Guided & Conformal Resource Management Architecture",
        fontsize=12, fontweight="bold", color="#f8fafc", family=font_family, va="center", zorder=3)
ax.text(96, 95.6, "Production Admission Controller & Hardware Provisioning Pipeline",
        fontsize=8.5, color="#94a3b8", family=font_family, ha="right", va="center", zorder=3)

# ==============================================================================
# STAGE 1: INPUT WORKLOAD & CANDIDATE HARDWARE SPACE (LEFT: x in [2, 23])
# ==============================================================================
draw_container(2, 2, 22, 88.5, c_blue_bg, c_blue_border,
               title="1. Specification Space", badge="Offline & Server")

# Input Card 1: LLM Workload Specification
card1 = FancyBboxPatch((3.5, 61), 19, 21.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                       facecolor="white", edgecolor="#bfdbfe", linewidth=1.2, zorder=2)
ax.add_patch(card1)
ax.text(4.5, 80.5, "Workload Spec  x = (m, s)", fontsize=9.2, fontweight="bold", color="#1e3a8a", zorder=3)
ax.text(4.5, 78.0, "Model Architecture Parameters m:", fontsize=7.8, fontweight="bold", color="#334155", zorder=3)
specs_m = [
    r"$\bullet$ Parameter Count: $N_{\mathrm{params}}$ (124M – 27B)",
    r"$\bullet$ Hidden Dimension: $H$ (768 – 4608)",
    r"$\bullet$ Transformer Layers: $L$ (12 – 46)",
    r"$\bullet$ Attention Heads: $N_h$ / KV Heads: $N_{kv}$",
    r"$\bullet$ Context & KV-Cache: $C_{\mathrm{kv}}$ (seq, tokens)"
]
for i, line in enumerate(specs_m):
    ax.text(5.0, 75.2 - i*2.2, line, fontsize=7.2, color="#475569", zorder=3)

ax.text(4.5, 63.8, "Operational Serving Scenario s:", fontsize=7.8, fontweight="bold", color="#334155", zorder=3)
ax.text(5.0, 61.8, r"$\bullet$ Scenario $s \in \{\mathrm{offline}, \mathrm{server\_low}, \mathrm{server\_high}\}$",
        fontsize=7.2, color="#475569", zorder=3)

# Input Card 2: Candidate Accelerator Space
card2 = FancyBboxPatch((3.5, 33), 19, 26, boxstyle="round,pad=0.3,rounding_size=1.0",
                       facecolor="white", edgecolor="#bfdbfe", linewidth=1.2, zorder=2)
ax.add_patch(card2)
ax.text(4.5, 56.8, "Hardware Pool  g in G", fontsize=9.2, fontweight="bold", color="#1e3a8a", zorder=3)
ax.text(4.5, 54.4, "Microarchitectural Specifications:", fontsize=7.8, fontweight="bold", color="#334155", zorder=3)
specs_g = [
    r"$\bullet$ Memory Bandwidth: $B_{\mathrm{mem}}$ (GB/s)",
    r"$\bullet$ Peak Compute: $F_{\mathrm{peak}}$ (FP16 TFLOPS)",
    r"$\bullet$ Device VRAM: $M_{\mathrm{size}}$ (16 – 141 GB)",
    r"$\bullet$ Thermal Design Power: $TDP$ (70 – 700 W)",
    r"$\bullet$ Clock Speeds: $f_{\mathrm{base}}, f_{\mathrm{boost}}$ (MHz)",
    r"$\bullet$ Architecture Family: $A_{\mathrm{gen}} \in$ {Volta,",
    r"   Turing, Ampere, Ada Lovelace, Hopper}"
]
for i, line in enumerate(specs_g):
    ax.text(5.0, 52.0 - i*2.2, line, fontsize=7.2, color="#475569", zorder=3)

ax.text(4.5, 36.2, "Evaluated Target Hardware:", fontsize=7.8, fontweight="bold", color="#1e3a8a", zorder=3)
ax.text(5.0, 34.2, r"$\bullet$ Unseen Target GPU $g_{\mathrm{target}}$ (Zero-Shot LOGO)",
        fontsize=7.2, fontweight="bold", color="#b91c1c", zorder=3)

# Input Card 3: Operational Constraints
card3 = FancyBboxPatch((3.5, 4.5), 19, 26.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                       facecolor="white", edgecolor="#bfdbfe", linewidth=1.2, zorder=2)
ax.add_patch(card3)
ax.text(4.5, 28.5, "Operational Policy Inputs", fontsize=9.2, fontweight="bold", color="#1e3a8a", zorder=3)
policy_lines = [
    r"$\bullet$ Target Latency SLA: $T_{\mathrm{max}} \in [20, 40]$ ms",
    r"$\bullet$ Nominal Coverage Level: $1 - \alpha = 90\%$",
    r"$\bullet$ Power Optimization Goal: $\min P(x, g)$",
    r"$\bullet$ Few-Shot Profiling Budget: $k \in \{1, 3, 5\}$",
    r"$\bullet$ Target-Label Blinding Quarantine"
]
for i, line in enumerate(policy_lines):
    ax.text(5.0, 26.0 - i*2.4, line, fontsize=7.2, color="#475569", zorder=3)

# Status pill
pill_blinding = FancyBboxPatch((4.5, 6.0), 17.0, 5.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                               facecolor="#fee2e2", edgecolor="#ef4444", linewidth=1.0, zorder=3)
ax.add_patch(pill_blinding)
ax.text(13.0, 9.2, "Zero-Leakage Protocol", fontsize=7.5, fontweight="bold", color="#991b1b",
        ha="center", zorder=4)
ax.text(13.0, 7.2, "Target Labels Wiped to NaN Prior to Lock", fontsize=6.5, color="#b91c1c",
        ha="center", zorder=4)

# ==============================================================================
# STAGE 2: PICAN DUAL-PATH PHYSICS-GUIDED CORE (CENTER-LEFT: x in [26, 54])
# ==============================================================================
draw_container(26, 2, 28, 88.5, c_teal_bg, c_teal_border,
               title="2. PICAN Physics-Informed Cross-Attention Engine", badge="Core Latency & Power")

# UPPER BRANCH: Deterministic Microarchitectural Roofline Base
roof_box = FancyBboxPatch((27.5, 57), 25, 25.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                          facecolor="white", edgecolor="#99f6e4", linewidth=1.2, zorder=2)
ax.add_patch(roof_box)
ax.text(28.5, 80.5, "Path A: Microarchitectural Roofline Floor", fontsize=9.0, fontweight="bold", color="#0f766e", zorder=3)
ax.text(28.5, 78.5, r"$T_{\mathrm{roof}}(x, g) = \max\left(\frac{2 N_{\mathrm{params}}}{B_{\mathrm{mem}} \cdot B(s)}, \frac{2 N_{\mathrm{params}}}{F_{\mathrm{peak}}}\right)$",
        fontsize=8.0, color="#115e59", zorder=3)

# Mini Roofline Plot embedded directly into the box!
sub_ax1 = fig.add_axes([0.282, 0.585, 0.145, 0.155])
sub_ax1.set_facecolor("#f8fafc")
i_vals = np.logspace(-2, 2, 100)
# Synthetic roofline curve
knee = 1.2
perf = np.minimum(i_vals * 150, 180)
sub_ax1.plot(i_vals, perf, color="#0f766e", lw=2.0)
sub_ax1.axvline(0.08, color="#dc2626", ls="--", lw=1.2)
sub_ax1.text(0.09, 25, "Decode\n(I << 1)", color="#dc2626", fontsize=6.5, fontweight="bold")
sub_ax1.text(0.02, 110, "Memory-Bound\nSlope = B_mem*B(s)", color="#0f766e", fontsize=6.2)
sub_ax1.text(1.5, 185, "Compute Ceiling: F_peak", color="#047857", fontsize=6.2)
sub_ax1.set_xscale("log")
sub_ax1.set_yscale("log")
sub_ax1.set_xlabel("Operational Intensity (FLOPs/Byte)", fontsize=6.2, labelpad=1)
sub_ax1.set_ylabel("Attainable TFLOPS", fontsize=6.2, labelpad=1)
sub_ax1.tick_params(axis="both", which="both", labelsize=5.5, pad=1)
sub_ax1.grid(True, alpha=0.3, ls=":")

# Non-bypassable constraint badge
ax.text(45.5, 71.0, "Physical Floor:\nExecution Time\nCan NEVER\nFall Below\nMemory Bus Limit",
        fontsize=6.8, color="#0f766e", fontweight="bold", ha="center", va="center", zorder=3,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ccfbf1", ec="#5eead4", lw=0.8))

# LOWER BRANCH: Bipartite Cross-Attention Core
attn_box = FancyBboxPatch((27.5, 23.5), 25, 31.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                          facecolor="white", edgecolor="#99f6e4", linewidth=1.2, zorder=2)
ax.add_patch(attn_box)
ax.text(28.5, 52.8, "Path B: Bipartite Cross-Attention (MCA)", fontsize=9.0, fontweight="bold", color="#0f766e", zorder=3)
ax.text(28.5, 50.6, r"Query $Q = W_q E_W(x)$, Key $K = W_k E_H(g)$, Value $V = W_v E_H(g)$",
        fontsize=7.4, color="#134e4a", zorder=3)

# Mini Attention Heatmap embedded!
sub_ax2 = fig.add_axes([0.282, 0.252, 0.12, 0.165])
attn_matrix = np.array([
    [0.72, 0.15, 0.08, 0.05],
    [0.18, 0.65, 0.12, 0.05],
    [0.05, 0.10, 0.75, 0.10],
    [0.12, 0.20, 0.18, 0.50]
])
im = sub_ax2.imshow(attn_matrix, cmap="YlGnBu", aspect="auto")
sub_ax2.set_xticks([0, 1, 2, 3])
sub_ax2.set_yticks([0, 1, 2, 3])
sub_ax2.set_xticklabels([r"$B_{\mathrm{mem}}$", r"$F_{\mathrm{pk}}$", r"$TDP$", r"$Gen$"], fontsize=6.2)
sub_ax2.set_yticklabels([r"$N_{\mathrm{p}}$", r"$H$", r"$L$", r"$C_{\mathrm{kv}}$"], fontsize=6.2)
sub_ax2.tick_params(axis="both", which="both", pad=1)
sub_ax2.set_title(r"Cross-Attention Weights $A = \mathrm{softmax}(QK^T / \sqrt{d})$", fontsize=6.5, pad=3)

ax.text(44.8, 38.0, "Dynamic\nWorkload-Hardware\nAffinity Mapping:\nDirectly links\nmodel geometry\nto memory bus\ncontrollers",
        fontsize=6.8, color="#134e4a", ha="center", va="center", zorder=3,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ccfbf1", ec="#5eead4", lw=0.8))

# Output Recombination Box
recomb_box = FancyBboxPatch((27.5, 4.5), 25, 17.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                            facecolor="white", edgecolor="#99f6e4", linewidth=1.2, zorder=2)
ax.add_patch(recomb_box)
ax.text(28.5, 20.0, "Multiplicative Physics Recombination", fontsize=8.8, fontweight="bold", color="#0f766e", zorder=3)
ax.text(28.5, 17.8, r"$\hat{T}(x, g) = T_{\mathrm{roof}}(x, g) \cdot \exp(\mathrm{softplus}(r_\theta(z))) \geq T_{\mathrm{roof}}$",
        fontsize=7.8, fontweight="bold", color="#0f766e", zorder=3)
ax.text(28.5, 15.0, r"$\bullet$ Operational Power Head: $\hat{P}(x, g) = TDP(g) \cdot \mathrm{sigmoid}(p_\theta(z))$",
        fontsize=7.2, color="#334155", zorder=3)
ax.text(28.5, 12.6, r"$\bullet$ Aleatoric Log-Variance Head: $s_\theta = \log \hat{\sigma}^2(x, g)$ (Heteroscedastic NLL)",
        fontsize=7.2, color="#334155", zorder=3)
ax.text(28.5, 10.2, r"$\bullet$ Prevents non-physical extrapolations on out-of-distribution GPUs",
        fontsize=7.0, color="#047857", fontstyle="italic", zorder=3)

# Recombination connector badge
ax.text(40.0, 21.0, r"$\otimes$ Non-Bypassable Inductive Anchor", fontsize=7.2, fontweight="bold",
        color="white", ha="center", zorder=4,
        bbox=dict(boxstyle="round,pad=0.25", fc="#0d9488", ec="none"))

# ==============================================================================
# STAGE 3: COVARIATE-SHIFT CONFORMAL RISK CALIBRATOR (CENTER-RIGHT: x in [56, 76])
# ==============================================================================
draw_container(56, 2, 20, 88.5, c_purple_bg, c_purple_border,
               title="3. Conformal Risk Calibrator", badge="Covariate Shift")

# Top Sub-box: Mahalanobis Hardware Discrepancy
maha_box = FancyBboxPatch((57.5, 55), 17, 27.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                          facecolor="white", edgecolor="#ddd6fe", linewidth=1.2, zorder=2)
ax.add_patch(maha_box)
ax.text(58.5, 80.5, "Mahalanobis Discrepancy", fontsize=9.0, fontweight="bold", color="#6d28d9", zorder=3)
ax.text(58.5, 78.4, r"$d_{\mathrm{shift}} = \sqrt{(g - \mu_{\mathrm{tr}})^T \Sigma_{\mathrm{tr}}^{-1} (g - \mu_{\mathrm{tr}})}$",
        fontsize=7.6, color="#5b21b6", zorder=3)

# Mini Manifold Scatter Plot embedded!
sub_ax3 = fig.add_axes([0.585, 0.565, 0.09, 0.165])
np.random.seed(10)
# Known GPUs cluster
g_known_x = np.random.normal(0, 0.8, 25)
g_known_y = np.random.normal(0, 0.8, 25)
sub_ax3.scatter(g_known_x, g_known_y, c="#7c3aed", s=18, alpha=0.6, label="Train GPUs")
# Unseen GPU target
sub_ax3.scatter([2.8], [2.4], c="#dc2626", s=60, marker="*", label="Unseen GPU")
sub_ax3.annotate("", xy=(2.8, 2.4), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2, ls="--"))
sub_ax3.text(1.2, 1.4, r"$d_{\mathrm{shift}}$", color="#dc2626", fontsize=7.0, fontweight="bold")
sub_ax3.set_xlim(-2.5, 3.5)
sub_ax3.set_ylim(-2.5, 3.5)
sub_ax3.tick_params(axis="both", which="both", labelsize=5.0, pad=1)
sub_ax3.set_title("Hardware Feature Manifold", fontsize=6.5, pad=2)

ax.text(66.0, 56.5, "Quantifies microarchitectural distance\nfrom training accelerator distribution",
        fontsize=6.8, color="#4c1d95", ha="center", zorder=3)

# Middle Sub-box: Conformal Quantile Inflation
quant_box = FancyBboxPatch((57.5, 33), 17, 20.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                           facecolor="white", edgecolor="#ddd6fe", linewidth=1.2, zorder=2)
ax.add_patch(quant_box)
ax.text(58.5, 51.5, "Shift-Adjusted Calibration", fontsize=9.0, fontweight="bold", color="#6d28d9", zorder=3)
ax.text(58.5, 49.0, "Base In-Distribution Quantile:", fontsize=7.2, fontweight="bold", color="#334155", zorder=3)
ax.text(58.5, 46.8, r"$\hat{q}_{1-\alpha} = \mathrm{Quantile}_{\frac{\lceil (n+1)(1-\alpha) \rceil}{n}}(\{s_i\})$",
        fontsize=7.4, color="#5b21b6", zorder=3)
ax.text(58.5, 44.0, "Shift Discrepancy Scaling:", fontsize=7.2, fontweight="bold", color="#334155", zorder=3)
ax.text(58.5, 41.8, r"$\hat{q}_{\mathrm{shift}}(g) = \hat{q}_{1-\alpha} \cdot (1 + \gamma \min(d_{\mathrm{shift}}, d_{\mathrm{max}}))$",
        fontsize=7.2, fontweight="bold", color="#6d28d9", zorder=3)
ax.text(58.5, 39.0, r"Empirical Heuristic under Covariate Shift", fontsize=6.8, color="#6b21a8", fontstyle="italic", zorder=3)
ax.text(58.5, 37.0, r"Measured Coverage: 86.8% (Nominal 90%)", fontsize=6.8, color="#9333ea", fontweight="bold", zorder=3)

# Bottom Sub-box: Empirical Upper SLA Latency Bound
bound_box = FancyBboxPatch((57.5, 4.5), 17, 27, boxstyle="round,pad=0.3,rounding_size=1.0",
                           facecolor="white", edgecolor="#ddd6fe", linewidth=1.2, zorder=2)
ax.add_patch(bound_box)
ax.text(58.5, 29.5, "Empirical SLA Latency Bound", fontsize=9.0, fontweight="bold", color="#6d28d9", zorder=3)
ax.text(58.5, 27.2, r"$\hat{T}_{\mathrm{upper}}(x, g) = \hat{T}(x, g) + \hat{q}_{\mathrm{shift}}(g) \cdot \hat{\sigma}(x, g)$",
        fontsize=7.6, fontweight="bold", color="#6d28d9", zorder=3)

# Mini Distribution Density Plot embedded!
sub_ax4 = fig.add_axes([0.585, 0.065, 0.09, 0.155])
x_lat = np.linspace(10, 45, 100)
mu_lat, sigma_lat = 25, 4.5
pdf_lat = (1 / (sigma_lat * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_lat - mu_lat)/sigma_lat)**2)
sub_ax4.plot(x_lat, pdf_lat, color="#7c3aed", lw=1.5)
t_upper = mu_lat + 1.645 * sigma_lat
sub_ax4.fill_between(x_lat, 0, pdf_lat, where=(x_lat <= t_upper), color="#ddd6fe", alpha=0.5)
sub_ax4.axvline(t_upper, color="#dc2626", lw=1.4, ls="-")
sub_ax4.text(t_upper - 1, 0.05, r"$\hat{T}_{\mathrm{upper}}$", color="#dc2626", fontsize=7.2, fontweight="bold", ha="right")
sub_ax4.text(mu_lat, 0.02, r"$\hat{T}$", color="#7c3aed", fontsize=7.2, fontweight="bold", ha="center")
sub_ax4.set_title("Empirical Latency Interval", fontsize=6.5, pad=2)
sub_ax4.tick_params(axis="both", which="both", labelsize=5.0, pad=1)

# ==============================================================================
# STAGE 4: DECISION ENGINE & ACTIVE ADAPTATION (RIGHT: x in [78, 98])
# ==============================================================================
draw_container(78, 2, 20, 88.5, c_green_bg, c_green_border,
               title="4. Admission & Adaptation", badge="ROAA & Selection")

# Sub-box 1: SLA Feasibility Filter & Hardware Selector
sel_box = FancyBboxPatch((79.5, 48), 17, 34.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                         facecolor="white", edgecolor="#bbf7d0", linewidth=1.2, zorder=2)
ax.add_patch(sel_box)
ax.text(80.5, 80.5, "SLA-Constrained Selector", fontsize=9.0, fontweight="bold", color="#15803d", zorder=3)
ax.text(80.5, 78.0, "1. Latency SLA Filter:", fontsize=7.4, fontweight="bold", color="#334155", zorder=3)
ax.text(80.5, 76.0, r"$\mathcal{G}_{\mathrm{feas}} = \{g \in \mathcal{G} \mid \hat{T}_{\mathrm{upper}}(x, g) \leq T_{\mathrm{max}}\}$",
        fontsize=7.4, color="#166534", zorder=3)

ax.text(80.5, 72.8, "2. Operational Power Objective:", fontsize=7.4, fontweight="bold", color="#334155", zorder=3)
ax.text(80.5, 70.8, r"$g^* = \arg\min_{g \in \mathcal{G}_{\mathrm{feas}}} \hat{P}(x, g)$",
        fontsize=7.8, fontweight="bold", color="#166534", zorder=3)

# Candidate GPU outcome table mini visual
sub_ax5 = fig.add_axes([0.805, 0.505, 0.09, 0.165])
bars = sub_ax5.barh(["L4 (72W)", "A30 (165W)", "L40S (350W)", "H100 (400W)"],
                    [72, 165, 350, 400], color=["#16a34a", "#94a3b8", "#94a3b8", "#cbd5e1"])
sub_ax5.set_xlabel("Mean Power (Watts)", fontsize=6.2, labelpad=1)
sub_ax5.tick_params(axis="both", which="both", labelsize=6.0, pad=1)
sub_ax5.set_title("Candidate Energy Ranking", fontsize=6.5, pad=2)
# Highlight selected
sub_ax5.text(85, 0, "Selected: g*", color="#16a34a", fontsize=6.5, fontweight="bold", va="center")

ax.text(88.0, 49.2, "Zero SLA Violations (0.0%)\nAcross 125 Accepted Workloads",
        fontsize=6.8, color="#15803d", fontweight="bold", ha="center", zorder=3)

# Sub-box 2: ROAA Regret-Optimal Active Probing
roaa_box = FancyBboxPatch((79.5, 4.5), 17, 42, boxstyle="round,pad=0.3,rounding_size=1.0",
                          facecolor="white", edgecolor="#fde68a", linewidth=1.2, zorder=2)
ax.add_patch(roaa_box)
ax.text(80.5, 44.5, "ROAA Active Adaptation Loop", fontsize=9.0, fontweight="bold", color="#b45309", zorder=3)
ax.text(80.5, 42.2, r"Trigger: If $d_{\mathrm{shift}} > \tau$ & budget $k \in \{1, 3, 5\}$",
        fontsize=7.0, color="#78350f", zorder=3)
ax.text(80.5, 39.8, "Decision-Boundary Utility:", fontsize=7.2, fontweight="bold", color="#334155", zorder=3)
ax.text(80.5, 37.5, r"$\alpha_{\mathrm{ROAA}}(x) \propto \hat{\sigma} \exp\left(-\frac{(\hat{T}-T_{\mathrm{max}})^2}{2\hat{\sigma}^2}\right) \frac{\hat{P}}{TDP}$",
        fontsize=7.0, fontweight="bold", color="#b45309", zorder=3)

# Mini ROAA Utility Plot embedded!
sub_ax6 = fig.add_axes([0.805, 0.155, 0.09, 0.165])
t_grid = np.linspace(15, 55, 100)
t_max_sla = 35.0
sigma_roaa = 4.0
u_roaa = np.exp(-0.5 * ((t_grid - t_max_sla) / sigma_roaa)**2)
sub_ax6.plot(t_grid, u_roaa, color="#d97706", lw=1.8, label="ROAA (Boundary)")
sub_ax6.axvline(t_max_sla, color="#dc2626", ls="--", lw=1.2, label=r"$T_{\mathrm{max}}$ SLA")
sub_ax6.plot(t_grid, np.ones_like(t_grid)*0.25, color="#94a3b8", ls=":", lw=1.2, label="Passive Uncertainty")
sub_ax6.set_xlabel("Latency T (ms)", fontsize=6.2, labelpad=1)
sub_ax6.set_ylabel("Acquisition Utility", fontsize=6.2, labelpad=1)
sub_ax6.set_title("ROAA Concentrates on SLA Border", fontsize=6.2, pad=2)
sub_ax6.tick_params(axis="both", which="both", labelsize=5.5, pad=1)
sub_ax6.legend(fontsize=4.8, loc="upper right")

# FiLM Latent Modulation
ax.text(80.5, 11.5, "Latent FiLM Modulation:", fontsize=7.2, fontweight="bold", color="#334155", zorder=3)
ax.text(80.5, 9.2, r"$z_{\mathrm{adapted}} = \gamma_k \odot z + \beta_k$",
        fontsize=7.4, fontweight="bold", color="#b45309", zorder=3)
ax.text(80.5, 7.0, r"Calibrates $k$ target GPU measurements without", fontsize=6.6, color="#475569", zorder=3)
ax.text(80.5, 5.2, r"modifying backbone physics weights (zero drift)", fontsize=6.6, color="#475569", zorder=3)

# ==============================================================================
# INTER-STAGE CONNECTING ARROWS & PIPELINE FLOW
# ==============================================================================
def connect_stages(x1, y1, x2, y2, text="", color="#475569", rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.8, mutation_scale=14,
            connectionstyle=f"arc3,rad={rad}"
        ),
        zorder=5
    )
    if text:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(mx, my, text, fontsize=7.2, fontweight="bold", color=color,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8, alpha=0.95))

# Stage 1 -> Stage 2 (Workload & Hardware features into Roofline and PICAN)
connect_stages(24.0, 70.0, 26.0, 70.0, color="#2563eb")
connect_stages(24.0, 40.0, 26.0, 40.0, color="#2563eb")

# Stage 2 -> Stage 3 (Predictions & Uncertainty into Conformal Calibrator)
connect_stages(54.0, 70.0, 56.0, 70.0, color="#0d9488")
connect_stages(54.0, 15.0, 56.0, 15.0, text=r"$\hat{T}, \hat{P}, \hat{\sigma}$", color="#0d9488")

# Stage 3 -> Stage 4 (Empirical Bounds into SLA Selector)
connect_stages(76.0, 65.0, 78.0, 65.0, text=r"$\hat{T}_{\mathrm{upper}} \leq T_{\mathrm{max}}$", color="#7c3aed")

# Stage 4 Active Adaptation Loop (ROAA feedback into PICAN via FiLM)
ax.annotate(
    "", xy=(40.0, 23.5), xytext=(88.0, 4.5),
    arrowprops=dict(
        arrowstyle="-|>", color="#d97706", lw=1.8, ls="--", mutation_scale=14,
        connectionstyle="arc3,rad=-0.28"
    ),
    zorder=5
)
ax.text(64.0, 0.8, r"ROAA Feedback Loop: FiLM Latent Adaptation $[\gamma_k, \beta_k]$ (Few-Shot Trigger)",
        fontsize=7.2, fontweight="bold", color="#b45309", ha="center", zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="#fffbeb", ec="#d97706", lw=0.9))

plt.savefig(os.path.join(FIG_DIR, "fig1_architecture.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(EXP_FIG_DIR, "fig1_architecture.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Figure 1 (Modular Software Architecture) with embedded micro-plots successfully generated!")
