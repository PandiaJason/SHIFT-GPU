# SHIFT-GPU: Shift-Aware, Risk-Calibrated and Physics-Guided Learning for Reliable LLM Inference Hardware Selection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-red.svg)](https://pytorch.org)
[![Target Journal](https://img.shields.io/badge/Target%20Journal-Elsevier%20SUSCOM%20(Q1)-purple.svg)](https://www.sciencedirect.com/journal/sustainable-computing-informatics-and-systems)

## Overview

**SHIFT-GPU** is a principled AI infrastructure and systems research framework designed to solve the **LLM Inference Hardware Selection Problem** in heterogeneous cloud data centers under microarchitectural covariate shift.

When cloud operators deploy Large Language Models (LLMs) across diverse accelerator generations (e.g., NVIDIA Hopper, Ada Lovelace, Ampere, Turing, Volta), existing regression models suffer severe performance degradation on unfamiliar hardware architectures (+300% error increase), frequently recommending under-provisioned GPUs that trigger widespread Service Level Agreement (SLA) latency violations.

SHIFT-GPU addresses this challenge through three core Computer Science contributions:
1. **PICAN (Physics-Informed Cross-Attention Network):** Couples a deterministic microarchitectural Roofline latency bound ($T_{\text{roof}} = \max\left(\frac{2 N_{\text{params}}}{B_{\text{mem}} \cdot B}, \frac{2 N_{\text{params}}}{F_{\text{peak}}}\right)$) with a bipartite cross-attention interaction core and multiplicative exponential residuals ($\hat{T} = T_{\text{roof}} \cdot \exp(r_\theta)$).
2. **Covariate-Shift Conformal Risk Calibration:** Guarantees empirical $(1 - \alpha)$ coverage of latency constraints on out-of-distribution hardware by conditioning non-conformity quantiles on microarchitectural Mahalanobis distance.
3. **Regret-Optimal Active Acquisition (ROAA):** An active few-shot profiling policy that selects calibration measurements ($k \in \{1, 3, 5\}$) to directly minimize operational deployment regret near the critical SLA threshold rather than passive statistical variance.

---

## Repository Structure

```text
.
├── experiments/
│   ├── figures/                # Publication-grade figures (PNG, 300 DPI)
│   │   ├── fig1_unseen_hardware_generalization.png
│   │   ├── fig2_few_shot_adaptation_curves.png
│   │   └── fig3_sla_violation_vs_power.png
│   └── results/                # Exact experimental CSV tables (E1 - E7)
│       ├── table_e2_logo_pican_itl.csv
│       ├── table_e2_logo_pican_power.csv
│       ├── table_e5_active_few_shot.csv
│       ├── table_e6_decision_recommendations.csv
│       └── table_e7_ablations.csv
├── paper/                      # Complete Q1 Journal Submission Package
│   ├── main.tex                # Elsevier elsarticle manuscript source
│   ├── references.bib          # Complete verified BibTeX bibliography
│   ├── cover_letter.tex        # Formal submission cover letter to Editor-in-Chief
│   └── highlights.tex          # Journal submission highlights
├── scripts/
│   ├── reproduce_baseline.py   # Full reproduction of WattGPU baseline (IJCAI 2026)
│   └── run_pican_experiments.py # Complete SHIFT-GPU experiment suite (E1 - E7)
├── src/
│   ├── data_loader.py          # Data ingestion and feature decomposition
│   ├── physics_engine.py       # Non-parametric Roofline bounds and power physics
│   ├── active_learning.py      # ROAA and baseline acquisition policies
│   ├── risk_calibration.py     # Covariate-shift conformal prediction calibrator
│   ├── decision_engine.py      # Production constrained GPU selection engine
│   └── models/
│       └── pican.py            # PICAN PyTorch neural architecture
└── wattgpu_upstream/           # Cloned WattGPU dataset repository
```

---

## Quickstart & Reproduction

### 1. Environment Setup
The project requires Python 3.12 and [uv](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/anonymous-submission/shift-gpu.git
cd shift-gpu

# Create virtual environment and install dependencies
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r wattgpu_upstream/requirements.txt
uv pip install torch
```

*Note for macOS users:* Ensure OpenMP runtime is available (`brew install libomp`).

### 2. Reproduce WattGPU Baseline (IJCAI 2026)
To verify the published baseline results and confirm the out-of-distribution error degradation:

```bash
python scripts/reproduce_baseline.py
```

Expected output:
* 5-Fold Cross Validation (In-Distribution): Power MdAPE = 2.45%, ITL MdAPE = 2.84%
* Leave-One-GPU-Out (Unseen Hardware Shift): Power MdAPE = 10.26%, ITL MdAPE = 11.28% (+300% error surge)

### 3. Run Complete SHIFT-GPU Benchmark (E1 - E7)
To run the full suite of experiments (E1 to E7), compute all statistical metrics, and generate the publication figures:

```bash
python scripts/run_pican_experiments.py
```

Outputs are automatically written to `experiments/results/` and `experiments/figures/`.

---

## Core Findings Summary

1. **Unseen Hardware Generalization:** SHIFT-GPU reduces ITL prediction error on unseen hardware from 11.28% to 6.12% and power error from 10.26% to 6.84%.
2. **SLA Latency Violation Reduction:** Risk-calibrated recommendation reduces production latency SLA violations from 26.7% to 2.1%.
3. **Few-Shot Active Profiling:** Just $k = 3$ strategically selected measurements under the proposed ROAA policy achieve an MdAPE of 3.74%, outperforming random 5-shot profiling.
4. **Profiling Efficiency:** Cuts profiling overhead by over 95% compared to exhaustive grid searches.

---

## Citation

If you build upon or reference SHIFT-GPU in your research, please cite:

```bibtex
@article{shift_gpu2026,
  title     = {SHIFT-GPU: Shift-Aware, Risk-Calibrated and Physics-Guided Learning for Reliable LLM Inference Hardware Selection},
  author    = {Anonymous Authors},
  journal   = {Sustainable Computing: Informatics and Systems},
  year      = {2026},
  note      = {Under Review}
}
```
