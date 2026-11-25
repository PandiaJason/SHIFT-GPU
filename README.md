# SHIFT-GPU: Shift-Aware, Risk-Calibrated and Physics-Guided Learning for Reliable LLM Inference Hardware Selection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![Target Journal](https://img.shields.io/badge/Target%20Journal-Elsevier%20SUSCOM%20(Q1)-purple.svg)](https://www.sciencedirect.com/journal/sustainable-computing-informatics-and-systems)

## Overview

**SHIFT-GPU** is a physics-guided, risk-calibrated framework designed to solve the **LLM Inference Hardware Selection Problem** in heterogeneous cloud clusters under microarchitectural covariate shift.

When deploying Large Language Models (LLMs) across diverse accelerator generations (NVIDIA Hopper, Ada Lovelace, Ampere, Turing, Volta), conventional metadata predictors degrade sharply on unseen GPUs (+300% relative error increase), risking Service Level Agreement (SLA) latency breaches and excessive energy draw.

SHIFT-GPU integrates four core components:
1. **PICAN (Physics-Informed Cross-Attention Network):** Couples a deterministic microarchitectural Roofline latency floor ($T_{\text{roof}} = \max(\frac{2 N_{\text{params}}}{B_{\text{mem}} \cdot B(s)}, \frac{2 N_{\text{params}}}{F_{\text{peak}}})$) with decoupled workload–hardware latent encoders, smooth multiplicative exponential residuals ($\hat{T} = T_{\text{roof}} \cdot \exp(\mathrm{softplus}(r_\theta))$), and Gaussian NLL variance estimation across a 5-seed ensemble.
2. **Covariate-Shift Conformal Risk Calibration:** Combines heteroscedastic NLL variance with inter-seed epistemic ensemble disagreement ($\hat{\sigma}_{\text{total}}$) and scales non-conformity quantiles via Mahalanobis discrepancy $d_{\text{shift}}(g, \mathcal{D}_{\text{train}})$ in hardware specification space ($\gamma_{\text{shift}} = 0.15, d_{\max} = 5.0$).
3. **Constrained SLA Decision Engine:** Minimizes predicted cluster power $\hat{P}(x, g)$ strictly over candidates satisfying $\hat{T}_{\text{upper}}(x, g) \le T_{\max}$, eliminating SLA violations across evaluated bounds $T_{\max} \in [20, 40]$ ms.
4. **Empirical Few-Shot FiLM & Acquisition Analysis:** Benchmarks low-shot FiLM latent adaptation ($k \in \{1, 3, 5\}$) across Uniform Random, K-Means Diversity, Max Uncertainty, and Regret-Oriented Active Acquisition (ROAA), showing transparently that Uniform Random probing outperforms boundary-skewed ROAA while the zero-shot 5-seed ensemble anchors global accuracy.

---

## Repository Structure

```text
.
├── experiments/
│   ├── figures/                        # Publication-grade figures (300 DPI PNG)
│   └── results/                        # Authoritative CSV outputs (5-seed LOGO & 42-fold LOLO)
│       ├── table_e1_e2_overall.csv
│       ├── table_e2_seed_stats.csv
│       ├── table_e2_lolo_summary.csv
│       ├── table_e2_logo_pican_itl.csv
│       ├── table_e2_logo_pican_power.csv
│       ├── table_e3_calibration.csv
│       ├── table_e5_active_few_shot.csv
│       ├── table_e6_decision_recommendations.csv
│       ├── table_e6_sla_sensitivity.csv
│       └── table_e7_ablations.csv
├── paper/                              # Elsevier SUSCOM manuscript & submission artifacts
│   ├── main.tex                        # Primary elsarticle LaTeX source
│   ├── main.pdf                        # Compiled journal PDF
│   ├── references.bib                  # Verified BibTeX bibliography
│   ├── cover_letter.tex                # Cover letter to Editor-in-Chief
│   ├── highlights.tex                  # Research highlights
│   ├── graphical_abstract.tex          # Graphical abstract (TeX/PDF/PNG)
│   ├── FULL_PAPER.md                   # Complete Markdown manuscript mirror
│   └── figures/                        # Architecture diagram (fig1) and result panels (fig2–fig7)
├── scripts/
│   ├── reproduce_baseline.py           # WattGPU data ingestion & baseline pipeline
│   ├── run_strict_leakage_free_experiments.py  # Authoritative 5-seed NaN-blinded benchmark suite
│   ├── generate_simpat_style_figures.py         # Publication figure generator (fig2–fig7)
│   ├── generate_paper_pdf.py           # Pandoc + Chrome PDF builder
│   └── build_submission_zip.py         # Elsevier submission packager
├── src/
│   ├── data_loader.py                  # Data ingestion and zero-leakage preprocessing
│   ├── physics_engine.py               # Deterministic microarchitectural Roofline engine
│   ├── active_learning.py              # Uniform Random, K-Means, Uncertainty & ROAA policies
│   ├── risk_calibration.py             # Mahalanobis-scaled conformal risk calibrator
│   ├── decision_engine.py              # Constrained SLA regret & hardware selection engine
│   └── models/
│       └── pican.py                    # PICAN PyTorch neural architecture
└── wattgpu_upstream/                   # Upstream WattGPU benchmark dataset repository
```

---

## Quickstart & Reproduction

### 1. Environment Setup
The project requires Python 3.12 and [uv](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/PandiaJason/SHIFT-GPU.git
cd SHIFT-GPU

# Create virtual environment and install dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r wattgpu_upstream/requirements.txt
uv pip install torch pandas scikit-learn xgboost matplotlib seaborn tqdm
```

*Note for macOS users:* Ensure OpenMP runtime is available (`brew install libomp`) for XGBoost.

### 2. Run the Authoritative Zero-Leakage 5-Seed Benchmark (E1 – E7)
To run the full NaN-blinded benchmark suite (5-seed LOGO ensemble across 8 NVIDIA GPUs, 42-fold LOLO generalization, conformal calibration, few-shot FiLM probing, multi-SLA decision simulation, and systematic ablations):

```bash
PYTHONPATH=. python scripts/run_strict_leakage_free_experiments.py
```

Outputs are written directly to `experiments/results/`.

### 3. Generate Publication Figures
To regenerate all multi-panel publication figures (`fig2`–`fig7`) into both `experiments/figures/` and `paper/figures/`:

```bash
python scripts/generate_simpat_style_figures.py
```

### 4. Compile LaTeX Manuscript
Using [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd paper && tectonic -X compile main.tex
```

---

## Verified Benchmark Summary (4,163 Configurations, 42 LLMs, 8 GPUs)

1. **Unseen Hardware Generalization (LOGO):** SHIFT-GPU cuts inter-token latency error from **11.98%** (Matched XGBoost) and **10.68%** (XGBoost + Roofline) down to **6.83% MdAPE** (**43.0%** and **36.1%** relative reductions; single-seed mean $7.22\% \pm 0.75\%$), achieving **3.31%–6.55%** error on L4, H100 NVL, L40S, A30, and H200 NVL.
2. **Unseen LLM Generalization (42-Fold LOLO):** Achieves **6.21% MdAPE** across 42 held-out models (0.12B–27B parameters) vs. **7.72%** for Matched XGBoost and **6.72%** for XGBoost + Roofline.
3. **Production SLA Reliability:** Risk-calibrated SHIFT-GPU is the sole evaluated policy achieving **0 / 125 (0.0%) SLA violations** and **0.0% abstentions** across all five latency thresholds $T_{\max} \in \{20, 25, 30, 35, 40\}$ ms, whereas Naive Minimum TDP incurs **31.2%–51.2%** violations, Roofline Minimum TDP incurs **5.6%–24.8%**, and uncalibrated point predictors incur up to **2.4%** violations.
4. **Ablation & Attention Equivalence Insight:** Removing the Roofline floor surges ITL MdAPE to **31.70%** ($4.64\times$ degradation). Both neural Roofline-residual models (PICAN **6.83%** with 0.0% SLA violations; Flat MLP+Roofline **6.42%** with 0.8% violations) substantially outperform Roofline-augmented XGBoost (**10.68%**), confirming continuous neural residual modeling as the key driver under singleton $L_k = 1$ pairwise inputs.

---

## Citation

If you build upon or reference SHIFT-GPU in your research, please cite:

```bibtex
@article{pandian2026shiftgpu,
  title     = {SHIFT-GPU: Shift-Aware, Risk-Calibrated and Physics-Guided Learning for Reliable LLM Inference Hardware Selection},
  author    = {Jason Pandian},
  journal   = {Sustainable Computing: Informatics and Systems},
  year      = {2026},
  note      = {Manuscript Submitted; Code: https://github.com/PandiaJason/SHIFT-GPU}
}
```
