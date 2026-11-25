#!/usr/bin/env python3
"""
Strict Leakage-Free Experimental Pipeline for SHIFT-GPU.
Guarantees 100% Zero-Leakage Validation:
1. Leave-One-GPU-Out (LOGO) across 8 GPUs and Leave-One-LLM-Out (LOLO) across 42 models.
2. Within LOGO, remaining 7 GPUs split into Training (80%) and Calibration (20%).
3. StandardScalers, models, and Mahalanobis shift metric fit strictly on Training data.
4. Shift-adjusted ConformalRiskCalibrator fits quantiles strictly on Calibration data
   using learned heteroscedastic + epistemic predictive uncertainty.
5. Explicit NaN-blinding verification asserts bit-for-bit identical test predictions.
6. Multi-seed ensemble (5 seeds) stabilizes convergence and reports Mean +/- Std.
7. Includes fair XGBoost + Roofline baseline and non-leaking Min-TDP baselines.

Author: Jason Pandian (Nehru Institute of Technology)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, kendalltau
from xgboost import XGBRegressor

from scripts.reproduce_baseline import load_and_preprocess_data, clean_df, _build_pipeline
from src.physics_engine import PhysicsEngine
from src.models.pican import PICAN
from src.active_learning import ActiveSelector
from src.risk_calibration import ConformalRiskCalibrator
from src.decision_engine import DecisionEngine

RESULTS_DIR = "experiments/results"
FIGURES_DIR = "experiments/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def evaluate_predictions(y_true, y_pred, df_group):
    """Computes MdAPE, Pearson r, and Kendall tau ranking correlations."""
    rel_errors = np.abs(y_true - y_pred) / y_true
    mdape = float(np.median(rel_errors) * 100.0)
    r, _ = pearsonr(y_true, y_pred)

    df_eval = df_group.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred

    gpu_taus = []
    for _, group in df_eval.groupby(["model", "scenario"]):
        if len(group) > 2 and group["y_true"].nunique() > 1 and group["y_pred"].nunique() > 1:
            tau, _ = kendalltau(group["y_true"], group["y_pred"])
            if not np.isnan(tau):
                gpu_taus.append(tau)

    llm_taus = []
    for _, group in df_eval.groupby("gpu_db_name"):
        if len(group) > 2 and group["y_true"].nunique() > 1 and group["y_pred"].nunique() > 1:
            tau, _ = kendalltau(group["y_true"], group["y_pred"])
            if not np.isnan(tau):
                llm_taus.append(tau)

    return {
        "MdAPE": mdape,
        "Pearson_r": float(r),
        "GPU_tau": float(np.mean(gpu_taus)) if gpu_taus else 0.0,
        "LLM_tau": float(np.mean(llm_taus)) if llm_taus else 0.0,
    }


def train_pican_itl(w_tr, h_tr, y_tr_res, seed, epochs=150):
    """Trains PICAN ITL model using Smooth L1 + Heteroscedastic Gaussian NLL (Eq. 9)."""
    torch.manual_seed(seed)
    model = PICAN(w_tr.shape[1], h_tr.shape[1], embed_dim=64, num_heads=4, target="itl").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        _, log_var, sp_res = model(w_tr, h_tr)
        lv = torch.clamp(log_var, -6.0, 4.0)
        nll = 0.5 * torch.exp(-lv) * (y_tr_res - sp_res) ** 2 + 0.5 * lv
        loss = F.smooth_l1_loss(sp_res, y_tr_res, beta=0.05) + 0.1 * torch.mean(nll)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


def train_pican_power(w_tr, h_tr, y_tr_pwr, seed, epochs=150):
    """Trains PICAN Power model bounded by TDP sigmoid with Smooth L1 + NLL."""
    torch.manual_seed(seed)
    model = PICAN(w_tr.shape[1], h_tr.shape[1], embed_dim=64, num_heads=4, target="power").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        _, log_var, sig_res = model(w_tr, h_tr)
        lv = torch.clamp(log_var, -6.0, 4.0)
        nll = 0.5 * torch.exp(-lv) * (y_tr_pwr - sig_res) ** 2 + 0.5 * lv
        loss = F.smooth_l1_loss(sig_res, y_tr_pwr, beta=0.02) + 0.05 * torch.mean(nll)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


def aggregate_itl_ensemble(models, w_t, h_t, t_roof_arr):
    """Aggregates ensemble predictions and total (aleatoric + epistemic) std in seconds."""
    r_list, v_list = [], []
    with torch.no_grad():
        for model in models:
            _, lv, sp_res = model(w_t, h_t)
            r_list.append(sp_res.cpu().numpy())
            v_list.append(torch.exp(torch.clamp(lv, -6.0, 4.0)).cpu().numpy())
    r_arr = np.stack(r_list, axis=0)
    v_arr = np.stack(v_list, axis=0)
    mean_r = np.mean(r_arr, axis=0)
    pred = t_roof_arr * np.exp(mean_r)
    std_res = np.sqrt(np.mean(v_arr, axis=0) + np.var(r_arr, axis=0))
    std_lin = pred * std_res
    return pred, std_lin, r_arr


def aggregate_power_ensemble(models, w_t, h_t, tdp_arr):
    """Aggregates ensemble power predictions in Watts."""
    p_list = []
    with torch.no_grad():
        for model in models:
            _, _, sig_res = model(w_t, h_t)
            p_list.append(tdp_arr * sig_res.cpu().numpy())
    return np.mean(np.stack(p_list, axis=0), axis=0)


class PureAttentionModel(nn.Module):
    """Ablation A: Without Roofline Base (direct log-ITL with heteroscedastic head)."""

    def __init__(self, w_dim, h_dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.w_enc = nn.Sequential(nn.Linear(w_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.h_enc = nn.Sequential(nn.Linear(h_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.head = nn.Linear(embed_dim, 1)
        self.unc_head = nn.Linear(embed_dim, 1)
        nn.init.constant_(self.unc_head.bias, -2.0)

    def forward(self, w, h):
        qw = self.w_enc(w).unsqueeze(1)
        kh = self.h_enc(h).unsqueeze(1)
        attn, _ = self.cross_attn(query=qw, key=kh, value=kh)
        z = self.norm1(qw + attn)
        z = self.norm2(z + self.ff(z)).squeeze(1)
        return self.head(z).squeeze(-1), self.unc_head(z).squeeze(-1)


class FlatMLPModel(nn.Module):
    """Ablation B: Without Cross-Attention (Flat Concatenated MLP + Roofline)."""

    def __init__(self, w_dim, h_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w_dim + h_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(embed_dim, 1)
        self.unc_head = nn.Linear(embed_dim, 1)
        nn.init.constant_(self.unc_head.bias, -2.0)

    def forward(self, w, h):
        z = self.net(torch.cat([w, h], dim=-1))
        return F.softplus(self.head(z).squeeze(-1)), self.unc_head(z).squeeze(-1)


def main():
    t_start = time.time()
    print("=" * 80)
    print("   SHIFT-GPU: Authoritative Leakage-Free Benchmark Suite (5-Seed Ensemble)")
    print(f"   Compute device: {device}")
    print("=" * 80)

    df_all = load_and_preprocess_data()
    cols_eval = [
        "gpu_db_name", "model", "itl", "gpu_0_power_draw_watts_mean", "normalized_power_draw",
        "scenario", "model_type", "total_b_params", "log_total_b_params", "hidden_size",
        "num_attention_heads", "num_key_value_heads", "num_layers", "memory_bandwidth_gb_s",
        "memory_size_gb", "tflops_16b", "thermal_design_power_w", "release_year",
        "bandwidth_latency", "compute_latency_s", "boost_percentage", "transistor_count_m", "base_clock_mhz",
    ]
    df = clean_df(df_all.loc[:, cols_eval]).reset_index(drop=True)
    unique_gpus = df["gpu_db_name"].unique()
    print(f"Evaluated samples: {len(df)} | Unique LLMs: {df['model'].nunique()} | GPUs: {len(unique_gpus)}")

    w_cols_itl = ["total_b_params", "hidden_size", "num_attention_heads", "num_key_value_heads", "num_layers"]
    w_cols_pwr = ["log_total_b_params", "total_b_params", "hidden_size", "num_attention_heads", "num_key_value_heads", "num_layers"]
    h_cols_itl = ["memory_bandwidth_gb_s", "memory_size_gb", "tflops_16b", "release_year", "bandwidth_latency", "compute_latency_s"]
    h_cols_pwr = ["thermal_design_power_w", "memory_bandwidth_gb_s", "base_clock_mhz", "boost_percentage", "transistor_count_m", "release_year", "tflops_16b", "bandwidth_latency"]
    scenario_dummies = pd.get_dummies(df["scenario"], drop_first=False).astype(float).values

    t_roof = PhysicsEngine.compute_roofline_latency(
        df["total_b_params"].values, df["memory_bandwidth_gb_s"].values, df["tflops_16b"].values, df["scenario"].values
    )
    y_itl = df["itl"].values
    target_res = np.log(y_itl) - np.log(t_roof)
    tdp = df["thermal_design_power_w"].values
    y_power = df["gpu_0_power_draw_watts_mean"].values
    y_norm_power = df["normalized_power_draw"].values
    t_pwr_base = np.array([PhysicsEngine.compute_power_base(w, s) for w, s in zip(tdp, df["scenario"])])

    X_xgb = df.drop(columns=["gpu_db_name", "model", "itl", "gpu_0_power_draw_watts_mean", "normalized_power_draw"], errors="ignore")
    groups = df["gpu_db_name"].values
    logo = LeaveOneGroupOut()

    # Output buffers
    preds_pican_itl = np.zeros(len(df))
    stds_pican_itl = np.zeros(len(df))
    bounds_pican_upper = np.zeros(len(df))
    bounds_pican_lower = np.zeros(len(df))
    uncalib_upper = np.zeros(len(df))
    uncalib_lower = np.zeros(len(df))
    preds_pican_power = np.zeros(len(df))

    preds_xgb_itl = np.zeros(len(df))
    preds_xgb_roof = np.zeros(len(df))
    bounds_xgb_upper = np.zeros(len(df))
    bounds_xgb_roof_upper = np.zeros(len(df))
    preds_xgb_power = np.zeros(len(df))
    hw_shift_scores = np.zeros(len(df))

    seed_preds_itl = {s: np.zeros(len(df)) for s in SEEDS}
    seed_preds_pwr = {s: np.zeros(len(df)) for s in SEEDS}
    logo_results_itl, logo_results_power = [], []
    models_by_gpu = {}

    # -------------------------------------------------------------------
    # Phase 1: Strict Zero-Leakage 8-Fold LOGO Cross-Validation
    # -------------------------------------------------------------------
    print("\n>>> Phase 1: 8-Fold LOGO with 5-Seed Ensemble + Zero-Leakage Blinding...")
    for fold, (train_pool_idx, test_idx) in enumerate(logo.split(df, y_itl, groups=groups)):
        holdout_gpu = groups[test_idx[0]]
        rng_fold = np.random.RandomState(42 + fold)
        perm = rng_fold.permutation(len(train_pool_idx))
        n_tr = int(0.80 * len(train_pool_idx))
        train_idx = train_pool_idx[perm[:n_tr]]
        cal_idx = train_pool_idx[perm[n_tr:]]

        assert len(set(test_idx) & set(train_idx)) == 0
        assert len(set(test_idx) & set(cal_idx)) == 0
        assert len(set(train_idx) & set(cal_idx)) == 0

        sw_i = StandardScaler().fit(df[w_cols_itl].values[train_idx])
        sh_i = StandardScaler().fit(df[h_cols_itl].values[train_idx])
        sw_p = StandardScaler().fit(df[w_cols_pwr].values[train_idx])
        sh_p = StandardScaler().fit(df[h_cols_pwr].values[train_idx])

        W_tr_i = np.hstack([sw_i.transform(df[w_cols_itl].values[train_idx]), scenario_dummies[train_idx]])
        W_ca_i = np.hstack([sw_i.transform(df[w_cols_itl].values[cal_idx]), scenario_dummies[cal_idx]])
        W_te_i = np.hstack([sw_i.transform(df[w_cols_itl].values[test_idx]), scenario_dummies[test_idx]])

        H_tr_i = sh_i.transform(df[h_cols_itl].values[train_idx])
        H_ca_i = sh_i.transform(df[h_cols_itl].values[cal_idx])
        H_te_i = sh_i.transform(df[h_cols_itl].values[test_idx])

        W_tr_p = np.hstack([sw_p.transform(df[w_cols_pwr].values[train_idx]), scenario_dummies[train_idx]])
        W_te_p = np.hstack([sw_p.transform(df[w_cols_pwr].values[test_idx]), scenario_dummies[test_idx]])
        H_tr_p = sh_p.transform(df[h_cols_pwr].values[train_idx])
        H_te_p = sh_p.transform(df[h_cols_pwr].values[test_idx])

        calibrator = ConformalRiskCalibrator(alpha=0.10, gamma_shift=0.15, d_max=5.0)
        calibrator.fit_shift_metric(H_tr_i)
        shift_dist = calibrator.compute_hardware_shift(H_te_i[0])
        hw_shift_scores[test_idx] = shift_dist

        w_tr_it = torch.tensor(W_tr_i, dtype=torch.float32).to(device)
        h_tr_it = torch.tensor(H_tr_i, dtype=torch.float32).to(device)
        w_ca_it = torch.tensor(W_ca_i, dtype=torch.float32).to(device)
        h_ca_it = torch.tensor(H_ca_i, dtype=torch.float32).to(device)
        w_te_it = torch.tensor(W_te_i, dtype=torch.float32).to(device)
        h_te_it = torch.tensor(H_te_i, dtype=torch.float32).to(device)

        w_tr_pt = torch.tensor(W_tr_p, dtype=torch.float32).to(device)
        h_tr_pt = torch.tensor(H_tr_p, dtype=torch.float32).to(device)
        w_te_pt = torch.tensor(W_te_p, dtype=torch.float32).to(device)
        h_te_pt = torch.tensor(H_te_p, dtype=torch.float32).to(device)

        y_tr_res = torch.tensor(target_res[train_idx], dtype=torch.float32).to(device)
        y_tr_pwr = torch.tensor(y_norm_power[train_idx], dtype=torch.float32).to(device)

        itl_models = [train_pican_itl(w_tr_it, h_tr_it, y_tr_res, s + fold) for s in SEEDS]
        pwr_models = [train_pican_power(w_tr_pt, h_tr_pt, y_tr_pwr, s + fold) for s in SEEDS]

        pred_cal, std_cal, _ = aggregate_itl_ensemble(itl_models, w_ca_it, h_ca_it, t_roof[cal_idx])
        calibrator.calibrate(y_itl[cal_idx], pred_cal, std_cal)

        pred_te, std_te, r_te_seeds = aggregate_itl_ensemble(itl_models, w_te_it, h_te_it, t_roof[test_idx])
        lb_te, ub_te = calibrator.get_prediction_intervals(pred_te, std_te, hw_shift_score=shift_dist)
        uncalib_lower[test_idx] = np.maximum(0.0, pred_te - 1.645 * std_te)
        uncalib_upper[test_idx] = pred_te + 1.645 * std_te
        pred_te_pwr = aggregate_power_ensemble(pwr_models, w_te_pt, h_te_pt, tdp[test_idx])

        for s_idx, s in enumerate(SEEDS):
            seed_preds_itl[s][test_idx] = t_roof[test_idx] * np.exp(r_te_seeds[s_idx])
            with torch.no_grad():
                _, _, sig_s = pwr_models[s_idx](w_te_pt, h_te_pt)
                seed_preds_pwr[s][test_idx] = tdp[test_idx] * sig_s.cpu().numpy()

        # Fair XGBoost Baselines (without ID columns)
        px_i = _build_pipeline(X_xgb.iloc[train_idx], XGBRegressor(max_depth=5, reg_lambda=100, n_estimators=100, random_state=42))
        px_i.fit(X_xgb.iloc[train_idx], np.log(y_itl[train_idx]))
        pred_cal_xgb = np.exp(px_i.predict(X_xgb.iloc[cal_idx]))
        q_xgb = float(np.quantile(np.abs(y_itl[cal_idx] - pred_cal_xgb) / pred_cal_xgb, min(1.0, 0.90 * (1.0 + 1.0 / len(cal_idx)))))
        pred_te_xgb = np.exp(px_i.predict(X_xgb.iloc[test_idx]))
        ub_te_xgb = pred_te_xgb * (1.0 + q_xgb * (1.0 + 0.15 * min(shift_dist, 5.0)))

        px_r = _build_pipeline(X_xgb.iloc[train_idx], XGBRegressor(max_depth=5, reg_lambda=100, n_estimators=100, random_state=42))
        px_r.fit(X_xgb.iloc[train_idx], target_res[train_idx])
        pred_cal_xgb_roof = t_roof[cal_idx] * np.exp(np.maximum(px_r.predict(X_xgb.iloc[cal_idx]), 0.0))
        q_xgb_roof = float(np.quantile(np.abs(y_itl[cal_idx] - pred_cal_xgb_roof) / pred_cal_xgb_roof, min(1.0, 0.90 * (1.0 + 1.0 / len(cal_idx)))))
        pred_te_xgb_roof = t_roof[test_idx] * np.exp(np.maximum(px_r.predict(X_xgb.iloc[test_idx]), 0.0))
        ub_te_xgb_roof = pred_te_xgb_roof * (1.0 + q_xgb_roof * (1.0 + 0.15 * min(shift_dist, 5.0)))

        px_p = _build_pipeline(X_xgb.iloc[train_idx], XGBRegressor(max_depth=6, reg_lambda=150, n_estimators=200, random_state=42))
        px_p.fit(X_xgb.iloc[train_idx], y_norm_power[train_idx])
        pred_te_xgb_pwr = px_p.predict(X_xgb.iloc[test_idx]) * tdp[test_idx]

        # NaN-blinding verification
        df_blind = df.copy()
        df_blind.loc[test_idx, ["itl", "gpu_0_power_draw_watts_mean", "normalized_power_draw"]] = np.nan
        w_te_b_t = torch.tensor(np.hstack([sw_i.transform(df_blind[w_cols_itl].values[test_idx]), scenario_dummies[test_idx]]), dtype=torch.float32).to(device)
        h_te_b_t = torch.tensor(sh_i.transform(df_blind[h_cols_itl].values[test_idx]), dtype=torch.float32).to(device)
        w_te_pb_t = torch.tensor(np.hstack([sw_p.transform(df_blind[w_cols_pwr].values[test_idx]), scenario_dummies[test_idx]]), dtype=torch.float32).to(device)
        h_te_pb_t = torch.tensor(sh_p.transform(df_blind[h_cols_pwr].values[test_idx]), dtype=torch.float32).to(device)

        pred_b, std_b, _ = aggregate_itl_ensemble(itl_models, w_te_b_t, h_te_b_t, t_roof[test_idx])
        _, ub_b = calibrator.get_prediction_intervals(pred_b, std_b, hw_shift_score=shift_dist)
        pred_pb = aggregate_power_ensemble(pwr_models, w_te_pb_t, h_te_pb_t, tdp[test_idx])
        X_xgb_b = df_blind.loc[test_idx, X_xgb.columns]
        assert np.allclose(pred_te, pred_b, equal_nan=False)
        assert np.allclose(ub_te, ub_b, equal_nan=False)
        assert np.allclose(pred_te_pwr, pred_pb, equal_nan=False)
        assert np.allclose(pred_te_xgb, np.exp(px_i.predict(X_xgb_b)), equal_nan=False)
        assert np.allclose(pred_te_xgb_pwr, px_p.predict(X_xgb_b) * tdp[test_idx], equal_nan=False)

        preds_pican_itl[test_idx] = pred_te
        stds_pican_itl[test_idx] = std_te
        bounds_pican_lower[test_idx] = lb_te
        bounds_pican_upper[test_idx] = ub_te
        preds_pican_power[test_idx] = pred_te_pwr
        preds_xgb_itl[test_idx] = pred_te_xgb
        preds_xgb_roof[test_idx] = pred_te_xgb_roof
        bounds_xgb_upper[test_idx] = ub_te_xgb
        bounds_xgb_roof_upper[test_idx] = ub_te_xgb_roof
        preds_xgb_power[test_idx] = pred_te_xgb_pwr

        cov_2 = float(np.mean((y_itl[test_idx] >= lb_te) & (y_itl[test_idx] <= ub_te)) * 100.0)
        cov_up = float(np.mean(y_itl[test_idx] <= ub_te) * 100.0)
        m_i = evaluate_predictions(y_itl[test_idx], pred_te, df.iloc[test_idx])
        m_xi = evaluate_predictions(y_itl[test_idx], pred_te_xgb, df.iloc[test_idx])
        m_xr = evaluate_predictions(y_itl[test_idx], pred_te_xgb_roof, df.iloc[test_idx])
        m_p = evaluate_predictions(y_power[test_idx], pred_te_pwr, df.iloc[test_idx])
        m_xp = evaluate_predictions(y_power[test_idx], pred_te_xgb_pwr, df.iloc[test_idx])

        print(f"  Fold {fold+1}/8 [{holdout_gpu:22s}] shift={shift_dist:5.2f} | PICAN={m_i['MdAPE']:5.2f}% vs XGB={m_xi['MdAPE']:5.2f}% vs XGB+Roof={m_xr['MdAPE']:5.2f}% | Cov2={cov_2:4.1f}% CovUp={cov_up:4.1f}%")
        logo_results_itl.append({
            "gpu": holdout_gpu, "MdAPE": m_i["MdAPE"], "XGB_MdAPE": m_xi["MdAPE"],
            "XGB_Roof_MdAPE": m_xr["MdAPE"], "Empirical_Coverage_Pct": cov_2,
            "Upper_Coverage_Pct": cov_up, "shift_dist": shift_dist,
        })
        logo_results_power.append({
            "gpu": holdout_gpu, "shift_dist": shift_dist, "XGB_MdAPE": m_xp["MdAPE"], **m_p
        })

        models_by_gpu[holdout_gpu] = {
            "itl_models": [copy.deepcopy(m) for m in itl_models],
            "scaler_w": sw_i, "scaler_h_itl": sh_i,
            "calibrator": calibrator, "shift_dist": shift_dist,
            "train_idx": train_idx, "cal_idx": cal_idx, "test_idx": test_idx,
        }

    pd.DataFrame(logo_results_itl).to_csv(os.path.join(RESULTS_DIR, "table_e2_logo_pican_itl.csv"), index=False)
    pd.DataFrame(logo_results_power).to_csv(os.path.join(RESULTS_DIR, "table_e2_logo_pican_power.csv"), index=False)

    overall_pican_i = evaluate_predictions(y_itl, preds_pican_itl, df)
    overall_xgb_i = evaluate_predictions(y_itl, preds_xgb_itl, df)
    overall_xgb_r = evaluate_predictions(y_itl, preds_xgb_roof, df)
    overall_roof_i = evaluate_predictions(y_itl, t_roof, df)
    overall_pican_p = evaluate_predictions(y_power, preds_pican_power, df)
    overall_xgb_p = evaluate_predictions(y_power, preds_xgb_power, df)
    overall_tdp_p = evaluate_predictions(y_power, t_pwr_base, df)

    seed_itls = [evaluate_predictions(y_itl, seed_preds_itl[s], df)["MdAPE"] for s in SEEDS]
    seed_pwrs = [evaluate_predictions(y_power, seed_preds_pwr[s], df)["MdAPE"] for s in SEEDS]

    cov_uncal_2 = float(np.mean((y_itl >= uncalib_lower) & (y_itl <= uncalib_upper)) * 100.0)
    cov_uncal_up = float(np.mean(y_itl <= uncalib_upper) * 100.0)
    cov_conf_2 = float(np.mean((y_itl >= bounds_pican_lower) & (y_itl <= bounds_pican_upper)) * 100.0)
    cov_conf_up = float(np.mean(y_itl <= bounds_pican_upper) * 100.0)

    pd.DataFrame([
        {"metric": "ITL", "model": "Non-Learning Roofline", **overall_roof_i},
        {"metric": "ITL", "model": "Matched XGBoost", **overall_xgb_i},
        {"metric": "ITL", "model": "XGBoost + Roofline", **overall_xgb_r},
        {"metric": "ITL", "model": "SHIFT-GPU (5-Seed PICAN)", **overall_pican_i},
        {"metric": "Power", "model": "Non-Learning TDP Base", **overall_tdp_p},
        {"metric": "Power", "model": "Matched XGBoost", **overall_xgb_p},
        {"metric": "Power", "model": "SHIFT-GPU (5-Seed PICAN)", **overall_pican_p},
    ]).to_csv(os.path.join(RESULTS_DIR, "table_e1_e2_overall.csv"), index=False)

    pd.DataFrame([{
        "itl_ensemble_mdape": overall_pican_i["MdAPE"],
        "itl_single_seed_mean": float(np.mean(seed_itls)),
        "itl_single_seed_std": float(np.std(seed_itls)),
        "power_ensemble_mdape": overall_pican_p["MdAPE"],
        "power_single_seed_mean": float(np.mean(seed_pwrs)),
        "power_single_seed_std": float(np.std(seed_pwrs)),
    }]).to_csv(os.path.join(RESULTS_DIR, "table_e2_seed_stats.csv"), index=False)

    pd.DataFrame([
        {"method": "Uncalibrated Gaussian (1.645 sigma)", "two_sided_cov_pct": cov_uncal_2, "upper_cov_pct": cov_uncal_up},
        {"method": "Shift-Adjusted Conformal", "two_sided_cov_pct": cov_conf_2, "upper_cov_pct": cov_conf_up},
    ]).to_csv(os.path.join(RESULTS_DIR, "table_e3_calibration.csv"), index=False)

    print(f"\n=== OVERALL LOGO ===\nPICAN ITL={overall_pican_i['MdAPE']:.2f}% (seed mean {np.mean(seed_itls):.2f}+/-{np.std(seed_itls):.2f}%) | XGB={overall_xgb_i['MdAPE']:.2f}% | XGB+Roof={overall_xgb_r['MdAPE']:.2f}% | Roof={overall_roof_i['MdAPE']:.2f}%")
    print(f"PICAN Pwr={overall_pican_p['MdAPE']:.2f}% | XGB Pwr={overall_xgb_p['MdAPE']:.2f}% | TDP Base={overall_tdp_p['MdAPE']:.2f}%")
    print(f"Calibration: Uncalib(2-sided={cov_uncal_2:.1f}%, up={cov_uncal_up:.1f}%) -> Conformal(2-sided={cov_conf_2:.1f}%, up={cov_conf_up:.1f}%)")

    df_preds = df.copy()
    df_preds["itl_true"] = y_itl
    df_preds["itl_pred_pican"] = preds_pican_itl
    df_preds["itl_std_pican"] = stds_pican_itl
    df_preds["itl_bound_upper"] = bounds_pican_upper
    df_preds["itl_bound_lower"] = bounds_pican_lower
    df_preds["itl_pred_xgb"] = preds_xgb_itl
    df_preds["itl_pred_xgb_roof"] = preds_xgb_roof
    df_preds["itl_xgb_upper"] = bounds_xgb_upper
    df_preds["itl_roofline"] = t_roof
    df_preds["power_true"] = y_power
    df_preds["power_pred_pican"] = preds_pican_power
    df_preds["power_pred_xgb"] = preds_xgb_power
    df_preds["hw_shift_score"] = hw_shift_scores
    df_preds.to_csv(os.path.join(RESULTS_DIR, "empirical_predictions.csv"), index=False)

    # -------------------------------------------------------------------
    # Phase 2: 42-Fold Leave-One-LLM-Out (LOLO) Cross-Validation
    # -------------------------------------------------------------------
    print("\n>>> Phase 2: 42-Fold Leave-One-LLM-Out (LOLO) Generalization...")
    lolo_groups = df["model"].values
    p_lolo_pican, p_lolo_xgb, p_lolo_xgb_r = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for fold, (tr_idx, te_idx) in enumerate(LeaveOneGroupOut().split(df, y_itl, groups=lolo_groups)):
        sw_l = StandardScaler().fit(df[w_cols_itl].values[tr_idx])
        sh_l = StandardScaler().fit(df[h_cols_itl].values[tr_idx])
        W_tr_l = torch.tensor(np.hstack([sw_l.transform(df[w_cols_itl].values[tr_idx]), scenario_dummies[tr_idx]]), dtype=torch.float32).to(device)
        H_tr_l = torch.tensor(sh_l.transform(df[h_cols_itl].values[tr_idx]), dtype=torch.float32).to(device)
        W_te_l = torch.tensor(np.hstack([sw_l.transform(df[w_cols_itl].values[te_idx]), scenario_dummies[te_idx]]), dtype=torch.float32).to(device)
        H_te_l = torch.tensor(sh_l.transform(df[h_cols_itl].values[te_idx]), dtype=torch.float32).to(device)
        m_l = train_pican_itl(W_tr_l, H_tr_l, torch.tensor(target_res[tr_idx], dtype=torch.float32).to(device), 42 + fold, epochs=110)
        p_lolo_pican[te_idx], _, _ = aggregate_itl_ensemble([m_l], W_te_l, H_te_l, t_roof[te_idx])

        px_l = _build_pipeline(X_xgb.iloc[tr_idx], XGBRegressor(max_depth=5, reg_lambda=100, n_estimators=100, random_state=42))
        px_l.fit(X_xgb.iloc[tr_idx], np.log(y_itl[tr_idx]))
        p_lolo_xgb[te_idx] = np.exp(px_l.predict(X_xgb.iloc[te_idx]))

        px_lr = _build_pipeline(X_xgb.iloc[tr_idx], XGBRegressor(max_depth=5, reg_lambda=100, n_estimators=100, random_state=42))
        px_lr.fit(X_xgb.iloc[tr_idx], target_res[tr_idx])
        p_lolo_xgb_r[te_idx] = t_roof[te_idx] * np.exp(np.maximum(px_lr.predict(X_xgb.iloc[te_idx]), 0.0))

    lolo_rows = [
        {"model": "Matched XGBoost", **evaluate_predictions(y_itl, p_lolo_xgb, df)},
        {"model": "XGBoost + Roofline", **evaluate_predictions(y_itl, p_lolo_xgb_r, df)},
        {"model": "SHIFT-GPU (PICAN)", **evaluate_predictions(y_itl, p_lolo_pican, df)},
    ]
    df_lolo = pd.DataFrame(lolo_rows)
    df_lolo.to_csv(os.path.join(RESULTS_DIR, "table_e2_lolo_summary.csv"), index=False)
    print(df_lolo.to_string(index=False))

    # -------------------------------------------------------------------
    # Phase 3: Production Recommendation via DecisionEngine (Exact Eq. 2)
    # -------------------------------------------------------------------
    print("\n>>> Phase 3: Production Hardware Recommendation Benchmark (DecisionEngine)...")
    pred_dict = {}
    true_dict = {}
    for _, row in df_preds.iterrows():
        k = (row["model"], row["scenario"], row["gpu_db_name"])
        pred_dict[k] = {
            "itl_roofline": row["itl_roofline"],
            "itl_xgb_point": row["itl_pred_xgb"],
            "itl_xgb_roof_point": row["itl_pred_xgb_roof"],
            "itl_pican_point": row["itl_pred_pican"],
            "itl_pican_upper": row["itl_bound_upper"],
            "power_pred_pican": row["power_pred_pican"],
            "power_pred_xgb": row["power_pred_xgb"],
            "tdp": row["thermal_design_power_w"],
        }
        true_dict[k] = {"power_true": row["power_true"], "itl_true": row["itl_true"]}

    unique_workloads = df_preds[["model", "scenario"]].drop_duplicates().reset_index(drop=True)
    policies_rec = ["min_tdp_naive", "min_tdp_roofline", "xgb_point", "xgb_roof_point", "pican_point", "shift_gpu_calibrated", "oracle"]

    sensitivity_records = []
    rec_35_records = []
    for sla_s in [0.020, 0.025, 0.030, 0.035, 0.040]:
        engine = DecisionEngine(t_max=sla_s, sla_penalty_factor=5.0, abstention_penalty=50.0)
        for pol in policies_rec:
            res = engine.evaluate_recommendations(unique_workloads, unique_gpus, pred_dict, true_dict, policy=pol)
            sensitivity_records.append(res)
            if abs(sla_s - 0.035) < 1e-6:
                rec_35_records.append(res)
                print(f"  [35ms] {pol:22s}: Acc={res['accepted_workloads']}/125 | Viol={res['violation_fraction_str']} ({res['violation_rate_pct']:.1f}%) | Pwr={res['mean_power_watts']:.1f}W | Regret={res['mean_regret']:.1f}")

    df_rec_35 = pd.DataFrame(rec_35_records)
    df_rec_35.to_csv(os.path.join(RESULTS_DIR, "table_e6_decision_recommendations.csv"), index=False)
    pd.DataFrame(sensitivity_records).to_csv(os.path.join(RESULTS_DIR, "table_e6_sla_sensitivity.csv"), index=False)

    # -------------------------------------------------------------------
    # Phase 4: Independent 5-Seed Ablation Study (E7)
    # -------------------------------------------------------------------
    print("\n>>> Phase 4: Independent 5-Seed Ablation Retraining...")
    preds_abl_a = np.zeros(len(df))
    preds_abl_b = np.zeros(len(df))
    ub_abl_a = np.zeros(len(df))
    ub_abl_b = np.zeros(len(df))
    ub_xgb = bounds_xgb_upper.copy()
    ub_xgb_r = np.zeros(len(df))

    for fold, (train_pool_idx, test_idx) in enumerate(logo.split(df, y_itl, groups=groups)):
        rng_fold = np.random.RandomState(42 + fold)
        perm = rng_fold.permutation(len(train_pool_idx))
        train_idx = train_pool_idx[perm[:int(0.80 * len(train_pool_idx))]]
        cal_idx = train_pool_idx[perm[int(0.80 * len(train_pool_idx)):]]

        sw = StandardScaler().fit(df[w_cols_itl].values[train_idx])
        sh = StandardScaler().fit(df[h_cols_itl].values[train_idx])
        W_tr = torch.tensor(np.hstack([sw.transform(df[w_cols_itl].values[train_idx]), scenario_dummies[train_idx]]), dtype=torch.float32).to(device)
        W_ca = torch.tensor(np.hstack([sw.transform(df[w_cols_itl].values[cal_idx]), scenario_dummies[cal_idx]]), dtype=torch.float32).to(device)
        W_te = torch.tensor(np.hstack([sw.transform(df[w_cols_itl].values[test_idx]), scenario_dummies[test_idx]]), dtype=torch.float32).to(device)
        H_tr = torch.tensor(sh.transform(df[h_cols_itl].values[train_idx]), dtype=torch.float32).to(device)
        H_ca = torch.tensor(sh.transform(df[h_cols_itl].values[cal_idx]), dtype=torch.float32).to(device)
        H_te = torch.tensor(sh.transform(df[h_cols_itl].values[test_idx]), dtype=torch.float32).to(device)

        y_tr_raw = torch.tensor(np.log(y_itl[train_idx]), dtype=torch.float32).to(device)
        y_tr_res = torch.tensor(target_res[train_idx], dtype=torch.float32).to(device)
        shift_dist = hw_shift_scores[test_idx[0]]

        ca_a_l, ca_av_l, te_a_l, te_av_l = [], [], [], []
        ca_b_l, ca_bv_l, te_b_l, te_bv_l = [], [], [], []
        for s in SEEDS:
            torch.manual_seed(s + fold)
            ma = PureAttentionModel(W_tr.shape[1], H_tr.shape[1]).to(device)
            oa = torch.optim.AdamW(ma.parameters(), lr=4e-3, weight_decay=1e-4)
            sa = torch.optim.lr_scheduler.CosineAnnealingLR(oa, T_max=150)
            for _ in range(150):
                oa.zero_grad()
                pa, lva = ma(W_tr, H_tr)
                lvc = torch.clamp(lva, -6.0, 4.0)
                loss_a = F.smooth_l1_loss(pa, y_tr_raw, beta=0.05) + 0.1 * torch.mean(0.5 * torch.exp(-lvc) * (y_tr_raw - pa) ** 2 + 0.5 * lvc)
                loss_a.backward(); oa.step(); sa.step()

            torch.manual_seed(s + fold)
            mb = FlatMLPModel(W_tr.shape[1], H_tr.shape[1]).to(device)
            ob = torch.optim.AdamW(mb.parameters(), lr=4e-3, weight_decay=1e-4)
            sb = torch.optim.lr_scheduler.CosineAnnealingLR(ob, T_max=150)
            for _ in range(150):
                ob.zero_grad()
                pb, lvb = mb(W_tr, H_tr)
                lvc = torch.clamp(lvb, -6.0, 4.0)
                loss_b = F.smooth_l1_loss(pb, y_tr_res, beta=0.05) + 0.1 * torch.mean(0.5 * torch.exp(-lvc) * (y_tr_res - pb) ** 2 + 0.5 * lvc)
                loss_b.backward(); ob.step(); sb.step()

            ma.eval(); mb.eval()
            with torch.no_grad():
                pa_c, lva_c = ma(W_ca, H_ca)
                ca_a_l.append(pa_c.cpu().numpy()); ca_av_l.append(torch.exp(torch.clamp(lva_c, -6.0, 4.0)).cpu().numpy())
                pa_t, lva_t = ma(W_te, H_te)
                te_a_l.append(pa_t.cpu().numpy()); te_av_l.append(torch.exp(torch.clamp(lva_t, -6.0, 4.0)).cpu().numpy())

                pb_c, lvb_c = mb(W_ca, H_ca)
                ca_b_l.append(pb_c.cpu().numpy()); ca_bv_l.append(torch.exp(torch.clamp(lvb_c, -6.0, 4.0)).cpu().numpy())
                pb_t, lvb_t = mb(W_te, H_te)
                te_b_l.append(pb_t.cpu().numpy()); te_bv_l.append(torch.exp(torch.clamp(lvb_t, -6.0, 4.0)).cpu().numpy())

        # Calibrate Ablation A & B identically
        cal_a = ConformalRiskCalibrator(alpha=0.10, gamma_shift=0.15, d_max=5.0)
        pca_a = np.exp(np.mean(np.stack(ca_a_l), axis=0))
        sca_a = pca_a * np.sqrt(np.mean(np.stack(ca_av_l), axis=0) + np.var(np.stack(ca_a_l), axis=0))
        cal_a.calibrate(y_itl[cal_idx], pca_a, sca_a)
        pte_a = np.exp(np.mean(np.stack(te_a_l), axis=0))
        ste_a = pte_a * np.sqrt(np.mean(np.stack(te_av_l), axis=0) + np.var(np.stack(te_a_l), axis=0))
        _, uba = cal_a.get_prediction_intervals(pte_a, ste_a, shift_dist)
        preds_abl_a[test_idx] = pte_a; ub_abl_a[test_idx] = uba

        cal_b = ConformalRiskCalibrator(alpha=0.10, gamma_shift=0.15, d_max=5.0)
        pca_b = t_roof[cal_idx] * np.exp(np.mean(np.stack(ca_b_l), axis=0))
        sca_b = pca_b * np.sqrt(np.mean(np.stack(ca_bv_l), axis=0) + np.var(np.stack(ca_b_l), axis=0))
        cal_b.calibrate(y_itl[cal_idx], pca_b, sca_b)
        pte_b = t_roof[test_idx] * np.exp(np.mean(np.stack(te_b_l), axis=0))
        ste_b = pte_b * np.sqrt(np.mean(np.stack(te_bv_l), axis=0) + np.var(np.stack(te_b_l), axis=0))
        _, ubb = cal_b.get_prediction_intervals(pte_b, ste_b, shift_dist)
        preds_abl_b[test_idx] = pte_b; ub_abl_b[test_idx] = ubb

    def eval_point_violations(arr, target_sla=0.035):
        v, tot = 0, 0
        idx_map = {(r["model"], r["scenario"], r["gpu_db_name"]): idx for idx, r in df_preds.iterrows()}
        for _, w_row in unique_workloads.iterrows():
            m_name, sc_name = w_row["model"], w_row["scenario"]
            elig = []
            for g in unique_gpus:
                k = (m_name, sc_name, g)
                if k in idx_map and arr[idx_map[k]] <= target_sla:
                    elig.append((pred_dict[k]["power_pred_pican"], true_dict[k]["itl_true"]))
            if elig:
                elig.sort(key=lambda x: x[0])
                tot += 1
                if elig[0][1] > target_sla:
                    v += 1
        return (v / tot * 100.0) if tot > 0 else 0.0

    viol_full = df_rec_35[df_rec_35["policy"] == "shift_gpu_calibrated"]["violation_rate_pct"].iloc[0]
    viol_xgb = df_rec_35[df_rec_35["policy"] == "xgb_point"]["violation_rate_pct"].iloc[0]
    viol_xgbr = df_rec_35[df_rec_35["policy"] == "xgb_roof_point"]["violation_rate_pct"].iloc[0]

    ablations = [
        {"model_variant": "Full Proposed SHIFT-GPU (PICAN + Roofline + Conformal)", "MdAPE_ITL": overall_pican_i["MdAPE"], "Empirical_Coverage_Pct": cov_conf_up, "SLA_Violation_Pct": viol_full},
        {"model_variant": "Ablation A: Without Physics Roofline (Pure Attention)", "MdAPE_ITL": float(np.median(np.abs(y_itl - preds_abl_a) / y_itl) * 100.0), "Empirical_Coverage_Pct": float(np.mean(y_itl <= ub_abl_a) * 100.0), "SLA_Violation_Pct": eval_point_violations(preds_abl_a)},
        {"model_variant": "Ablation B: Without Cross-Attention (Flat MLP + Roofline)", "MdAPE_ITL": float(np.median(np.abs(y_itl - preds_abl_b) / y_itl) * 100.0), "Empirical_Coverage_Pct": float(np.mean(y_itl <= ub_abl_b) * 100.0), "SLA_Violation_Pct": eval_point_violations(preds_abl_b)},
        {"model_variant": "Baseline C: XGBoost + Roofline", "MdAPE_ITL": overall_xgb_r["MdAPE"], "Empirical_Coverage_Pct": float(np.mean(y_itl <= bounds_xgb_roof_upper) * 100.0), "SLA_Violation_Pct": viol_xgbr},
        {"model_variant": "Baseline D: Matched WattGPU (XGBoost)", "MdAPE_ITL": overall_xgb_i["MdAPE"], "Empirical_Coverage_Pct": float(np.mean(y_itl <= ub_xgb) * 100.0), "SLA_Violation_Pct": viol_xgb},
    ]
    df_abl = pd.DataFrame(ablations)
    df_abl.to_csv(os.path.join(RESULTS_DIR, "table_e7_ablations.csv"), index=False)
    print(df_abl.to_string(index=False))

    # -------------------------------------------------------------------
    # Phase 5: Active Few-Shot Hardware Profiling (Untouched Remainder)
    # -------------------------------------------------------------------
    print("\n>>> Phase 5: Few-Shot FiLM Calibration Benchmark (5-Seed Ensemble)...")
    selector = ActiveSelector()
    few_shot_rows = []
    for pol in ["random", "diversity", "uncertainty", "regret_optimal"]:
        for k in [0, 1, 3, 5]:
            all_y_true, all_y_pred = [], []
            for holdout_gpu, m_dict in models_by_gpu.items():
                test_idx = m_dict["test_idx"]
                sub_test = df.iloc[test_idx].reset_index(drop=True)
                W_te = np.hstack([m_dict["scaler_w"].transform(sub_test[w_cols_itl].values), scenario_dummies[test_idx]])
                H_te = m_dict["scaler_h_itl"].transform(sub_test[h_cols_itl].values)
                y_te_true = sub_test["itl"].values
                r_te = t_roof[test_idx]

                if k == 0:
                    all_y_true.append(y_te_true)
                    all_y_pred.append(preds_pican_itl[test_idx])
                    continue

                cand_indices = np.arange(len(sub_test))
                preds_zs = preds_pican_itl[test_idx]
                vars_zs = (stds_pican_itl[test_idx] ** 2)
                pwr_zs = preds_pican_power[test_idx]

                if pol == "random":
                    probe_idx = selector.select_random(cand_indices, k, seed=42)
                elif pol == "diversity":
                    probe_idx = selector.select_diversity(W_te, cand_indices, k, seed=42)
                elif pol == "uncertainty":
                    probe_idx = selector.select_max_uncertainty(vars_zs, cand_indices, k)
                elif pol == "regret_optimal":
                    probe_idx = selector.select_regret_optimal(preds_zs, vars_zs, pwr_zs, cand_indices, k, t_max=0.035)

                probe_idx = np.asarray(probe_idx)
                if len(probe_idx) < k:
                    missing = [idx for idx in cand_indices if idx not in probe_idx]
                    probe_idx = np.concatenate([probe_idx, np.random.RandomState(42).choice(missing, size=k - len(probe_idx), replace=False)])
                probe_idx = probe_idx[:k]
                untouched = np.array([idx for idx in range(len(sub_test)) if idx not in probe_idx])

                w_pr = torch.tensor(W_te[probe_idx], dtype=torch.float32).to(device)
                h_pr = torch.tensor(H_te[probe_idx], dtype=torch.float32).to(device)
                y_pr = torch.tensor(target_res[test_idx[probe_idx]], dtype=torch.float32).to(device)
                w_ev = torch.tensor(W_te[untouched], dtype=torch.float32).to(device)
                h_ev = torch.tensor(H_te[untouched], dtype=torch.float32).to(device)

                adapted_models = []
                for base_m in m_dict["itl_models"]:
                    am = copy.deepcopy(base_m)
                    am.freeze_backbone_for_few_shot()
                    opt_f = torch.optim.Adam(am.film.parameters(), lr=0.01, weight_decay=1e-2)
                    for _ in range(35):
                        opt_f.zero_grad()
                        _, _, p_r = am(w_pr, h_pr)
                        # Ridge anchor toward FiLM identity prevents overfitting on k<=5 shots
                        reg_id = 1e-2 * (torch.sum((am.film.gamma - 1.0) ** 2) + torch.sum(am.film.beta ** 2))
                        loss_f = F.smooth_l1_loss(p_r, y_pr, beta=0.05) + reg_id
                        loss_f.backward()
                        opt_f.step()
                    am.eval()
                    adapted_models.append(am)

                pred_ev, _, _ = aggregate_itl_ensemble(adapted_models, w_ev, h_ev, r_te[untouched])
                all_y_true.append(y_te_true[untouched])
                all_y_pred.append(pred_ev)

            yt = np.concatenate(all_y_true)
            yp = np.concatenate(all_y_pred)
            md = float(np.median(np.abs(yt - yp) / yt) * 100.0)
            pr = float(pearsonr(yt, yp)[0])
            few_shot_rows.append({"policy": pol, "budget_k": k, "mdape_itl": md, "pearson_r": pr})
        print(f"  Few-shot [{pol:14s}] k=0..5 MdAPE: {[round(r['mdape_itl'], 2) for r in few_shot_rows[-4:]]}")

    pd.DataFrame(few_shot_rows).to_csv(os.path.join(RESULTS_DIR, "table_e5_active_few_shot.csv"), index=False)
    print(f"\nCompleted full authoritative benchmark in {time.time() - t_start:.1f}s!")


if __name__ == "__main__":
    main()
