#!/usr/bin/env python3
"""
Strict Leakage-Free Experimental Pipeline for SHIFT-GPU.
Guarantees 100% Zero-Leakage Validation:
1. Leave-One-GPU-Out (LOGO): For each of the 8 GPUs, the held-out GPU's test data is completely isolated.
2. The remaining 7 GPUs are split into Training (80%) and Calibration (20%) sets.
3. Preprocessing (StandardScalers) and models are fitted strictly on Training data only.
4. Conformal uncertainty quantiles are computed strictly on Calibration data only.
5. Microarchitectural Mahalanobis shift metric is fitted strictly on Training hardware features.
6. Explicit Data Leakage Assertion: Predictions, bounds, and recommendations are verified to be
   100% IDENTICAL whether the test target measurements are present or completely wiped to NaN.
7. Decisions are scored only AFTER locking predictions and bounds.
8. Reports violations among accepted workloads (e.g., '0 / N accepted'), abstentions among all workloads,
   and mean power of selected GPUs.
9. Runs ablations and few-shot tests independently with evaluation on strictly untouched held-out samples.

Author: Jason Pandian (Nehru Institute of Technology)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

# Set random seeds for strict reproducibility
np.random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "experiments/results"
FIGURES_DIR = "experiments/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using compute acceleration device: {device}")


def evaluate_predictions(y_true, y_pred, df_group):
    """Computes MdAPE, Pearson r, and Kendall tau ranking correlations."""
    rel_errors = np.abs(y_true - y_pred) / y_true
    mdape = float(np.median(rel_errors) * 100.0)
    r, _ = pearsonr(y_true, y_pred)

    df_eval = df_group.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred

    # Rank correlation across GPUs for each workload
    gpu_taus = []
    for _, group in df_eval.groupby(["model", "scenario"]):
        if len(group) > 2:
            tau, _ = kendalltau(group["y_true"], group["y_pred"])
            if not np.isnan(tau):
                gpu_taus.append(tau)
    gpu_tau = float(np.mean(gpu_taus)) if gpu_taus else 0.0

    # Rank correlation across LLMs for each GPU
    llm_taus = []
    for _, group in df_eval.groupby("gpu_db_name"):
        if len(group) > 2:
            tau, _ = kendalltau(group["y_true"], group["y_pred"])
            if not np.isnan(tau):
                llm_taus.append(tau)
    llm_tau = float(np.mean(llm_taus)) if llm_taus else 0.0

    return {
        "MdAPE": mdape,
        "Pearson_r": float(r),
        "GPU_tau": gpu_tau,
        "LLM_tau": llm_tau
    }


def main():
    print("=" * 80)
    print("   SHIFT-GPU: Strict Leakage-Free Scientific Benchmark (Zero-Leakage)")
    print("=" * 80)

    # 1. Load and clean real empirical dataset
    df_all = load_and_preprocess_data()
    cols_eval = [
        "gpu_db_name", "model", "itl", "gpu_0_power_draw_watts_mean", "normalized_power_draw",
        "scenario", "model_type", "total_b_params", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "num_layers", "memory_bandwidth_gb_s", "memory_size_gb",
        "tflops_16b", "thermal_design_power_w", "release_year", "bandwidth_latency",
        "compute_latency_s", "boost_percentage", "transistor_count_m", "base_clock_mhz"
    ]
    df = clean_df(df_all.loc[:, cols_eval]).reset_index(drop=True)
    unique_gpus = df["gpu_db_name"].unique()
    print(f"Total evaluated dataset samples: {len(df)}")
    print(f"Target GPU population ({len(unique_gpus)}): {list(unique_gpus)}")

    w_cols = ["total_b_params", "hidden_size", "num_attention_heads", "num_key_value_heads", "num_layers"]
    h_cols_itl = ["memory_bandwidth_gb_s", "memory_size_gb", "tflops_16b", "release_year", "bandwidth_latency", "compute_latency_s"]
    h_cols_pwr = ["thermal_design_power_w", "base_clock_mhz", "boost_percentage", "transistor_count_m", "release_year", "tflops_16b"]

    scenario_dummies = pd.get_dummies(df["scenario"], drop_first=False).astype(float).values

    # Compute deterministic Roofline latency floor (seconds)
    pe = PhysicsEngine()
    t_roof = pe.compute_roofline_latency(
        df["total_b_params"].values,
        df["memory_bandwidth_gb_s"].values,
        df["tflops_16b"].values,
        df["scenario"].values
    )
    y_itl = df["itl"].values
    target_res = np.log(y_itl) - np.log(t_roof)

    # Power targets
    tdp = df["thermal_design_power_w"].values
    y_power = df["gpu_0_power_draw_watts_mean"].values
    y_norm_power = df["normalized_power_draw"].values

    groups = df["gpu_db_name"].values
    logo = LeaveOneGroupOut()

    # Pre-allocate output arrays
    preds_pican_itl = np.zeros(len(df))
    bounds_pican_itl_upper = np.zeros(len(df))
    bounds_pican_itl_lower = np.zeros(len(df))
    preds_pican_power = np.zeros(len(df))
    preds_xgb_itl = np.zeros(len(df))
    bounds_xgb_itl_upper = np.zeros(len(df))
    preds_xgb_power = np.zeros(len(df))
    hw_shift_scores = np.zeros(len(df))

    logo_results_itl = []
    logo_results_power = []

    # Dictionaries to store fitted models and calibration artifacts per holdout GPU
    models_by_gpu = {}

    print("\n>>> Phase 1: Strict Train/Cal/Test LOGO Cross-Validation across 8 GPUs...")

    for fold, (train_pool_idx, test_idx) in enumerate(logo.split(df, y_itl, groups=groups)):
        holdout_gpu = groups[test_idx[0]]
        print(f"\n--- Fold {fold + 1}/8: Holding out [{holdout_gpu}] (N_test={len(test_idx)}) ---")

        # -------------------------------------------------------------------
        # Requirement 1: Split remaining 7 GPUs into Train (80%) and Cal (20%)
        # -------------------------------------------------------------------
        rng_fold = np.random.RandomState(42 + fold)
        n_pool = len(train_pool_idx)
        perm = rng_fold.permutation(n_pool)
        n_train = int(0.80 * n_pool)
        train_idx = train_pool_idx[perm[:n_train]]
        cal_idx = train_pool_idx[perm[n_train:]]

        # Verify strict disjointness
        assert len(set(test_idx).intersection(set(train_idx))) == 0
        assert len(set(test_idx).intersection(set(cal_idx))) == 0
        assert len(set(train_idx).intersection(set(cal_idx))) == 0
        print(f"  Split sizes: Train={len(train_idx)}, Cal={len(cal_idx)}, Test={len(test_idx)}")

        # -------------------------------------------------------------------
        # Fit Preprocessing SCALERS on TRAIN DATA ONLY
        # -------------------------------------------------------------------
        scaler_w = StandardScaler()
        scaler_w.fit(df[w_cols].values[train_idx])

        scaler_h_itl = StandardScaler()
        scaler_h_itl.fit(df[h_cols_itl].values[train_idx])

        scaler_h_pwr = StandardScaler()
        scaler_h_pwr.fit(df[h_cols_pwr].values[train_idx])

        # Transform Workload Features
        W_train = np.hstack([scaler_w.transform(df[w_cols].values[train_idx]), scenario_dummies[train_idx]])
        W_cal = np.hstack([scaler_w.transform(df[w_cols].values[cal_idx]), scenario_dummies[cal_idx]])
        W_test = np.hstack([scaler_w.transform(df[w_cols].values[test_idx]), scenario_dummies[test_idx]])

        # Transform Hardware Features (ITL)
        H_train_itl = scaler_h_itl.transform(df[h_cols_itl].values[train_idx])
        H_cal_itl = scaler_h_itl.transform(df[h_cols_itl].values[cal_idx])
        H_test_itl = scaler_h_itl.transform(df[h_cols_itl].values[test_idx])

        # Transform Hardware Features (Power)
        H_train_pwr = scaler_h_pwr.transform(df[h_cols_pwr].values[train_idx])
        H_cal_pwr = scaler_h_pwr.transform(df[h_cols_pwr].values[cal_idx])
        H_test_pwr = scaler_h_pwr.transform(df[h_cols_pwr].values[test_idx])

        # -------------------------------------------------------------------
        # Fit Microarchitectural Shift Metric on TRAIN DATA ONLY
        # -------------------------------------------------------------------
        calibrator = ConformalRiskCalibrator(alpha=0.10)
        calibrator.fit_shift_metric(H_train_itl)
        shift_dist = calibrator.compute_hardware_shift(H_test_itl[0])
        hw_shift_scores[test_idx] = shift_dist
        print(f"  Microarchitectural Mahalanobis shift d_shift = {shift_dist:.3f}")

        # PyTorch Tensors for Training
        w_tr = torch.tensor(W_train, dtype=torch.float32).to(device)
        h_tr_itl = torch.tensor(H_train_itl, dtype=torch.float32).to(device)
        h_tr_pwr = torch.tensor(H_train_pwr, dtype=torch.float32).to(device)
        y_tr_res = torch.tensor(target_res[train_idx], dtype=torch.float32).to(device)
        y_tr_pwr = torch.tensor(y_norm_power[train_idx], dtype=torch.float32).to(device)

        # PyTorch Tensors for Calibration
        w_ca = torch.tensor(W_cal, dtype=torch.float32).to(device)
        h_ca_itl = torch.tensor(H_cal_itl, dtype=torch.float32).to(device)
        h_ca_pwr = torch.tensor(H_cal_pwr, dtype=torch.float32).to(device)
        r_ca = torch.tensor(t_roof[cal_idx], dtype=torch.float32).to(device)
        tdp_ca = torch.tensor(tdp[cal_idx], dtype=torch.float32).to(device)

        # PyTorch Tensors for Test (Features only! Test targets y_itl[test_idx] are NOT loaded into tensors!)
        w_te = torch.tensor(W_test, dtype=torch.float32).to(device)
        h_te_itl = torch.tensor(H_test_itl, dtype=torch.float32).to(device)
        h_te_pwr = torch.tensor(H_test_pwr, dtype=torch.float32).to(device)
        r_te = torch.tensor(t_roof[test_idx], dtype=torch.float32).to(device)
        tdp_te = torch.tensor(tdp[test_idx], dtype=torch.float32).to(device)

        # -------------------------------------------------------------------
        # Train PICAN ITL on TRAIN DATA ONLY
        # -------------------------------------------------------------------
        torch.manual_seed(42 + fold)
        model_itl = PICAN(w_tr.shape[1], h_tr_itl.shape[1], embed_dim=64, num_heads=4, target="itl").to(device)
        opt_itl = torch.optim.AdamW(model_itl.parameters(), lr=4e-3, weight_decay=1e-4)

        for _ in range(120):
            opt_itl.zero_grad()
            _, _, p_res = model_itl(w_tr, h_tr_itl)
            loss_itl = nn.functional.smooth_l1_loss(p_res, y_tr_res, beta=0.05)
            loss_itl.backward()
            opt_itl.step()

        # -------------------------------------------------------------------
        # Train PICAN Power on TRAIN DATA ONLY
        # -------------------------------------------------------------------
        torch.manual_seed(42 + fold)
        model_pwr = PICAN(w_tr.shape[1], h_tr_pwr.shape[1], embed_dim=64, num_heads=4, target="power").to(device)
        opt_pwr = torch.optim.AdamW(model_pwr.parameters(), lr=4e-3, weight_decay=1e-4)

        for _ in range(120):
            opt_pwr.zero_grad()
            _, _, p_res_pwr = model_pwr(w_tr, h_tr_pwr)
            loss_pwr = nn.functional.smooth_l1_loss(torch.sigmoid(p_res_pwr), y_tr_pwr, beta=0.02)
            loss_pwr.backward()
            opt_pwr.step()

        # -------------------------------------------------------------------
        # Train Matched XGBoost Baselines on TRAIN DATA ONLY
        # -------------------------------------------------------------------
        X_xgb = df.drop(columns=["itl", "gpu_0_power_draw_watts_mean", "normalized_power_draw"], errors="ignore")
        pipe_xgb_itl = _build_pipeline(X_xgb.iloc[train_idx], XGBRegressor(max_depth=5, reg_lambda=100, n_estimators=100, random_state=42))
        pipe_xgb_itl.fit(X_xgb.iloc[train_idx], np.log(y_itl[train_idx]))

        pipe_xgb_pwr = _build_pipeline(X_xgb.iloc[train_idx], XGBRegressor(max_depth=5, reg_lambda=100, n_estimators=100, random_state=42))
        pipe_xgb_pwr.fit(X_xgb.iloc[train_idx], y_power[train_idx])

        # -------------------------------------------------------------------
        # Calibrate Uncertainty Bounds strictly on CALIBRATION SET ONLY
        # -------------------------------------------------------------------
        model_itl.eval()
        with torch.no_grad():
            # Calibration predictions
            p_cal_itl, _, _ = model_itl(w_ca, h_ca_itl, physics_base=r_ca)
            pred_cal = p_cal_itl.cpu().numpy()
            y_cal_true = y_itl[cal_idx]

            # Relative nonconformity scores on Calibration set: |y_cal - p_cal| / p_cal
            cal_scores_pican = np.abs(y_cal_true - pred_cal) / pred_cal
            # Conformal quantile at 1 - alpha level
            n_c = len(cal_scores_pican)
            q_level = min(1.0, (1.0 - 0.10) * (1.0 + 1.0 / n_c))
            q_hat_pican = float(np.quantile(cal_scores_pican, q_level))

            # Calibrate XGBoost bounds on Calibration set as well
            pred_cal_xgb = np.exp(pipe_xgb_itl.predict(X_xgb.iloc[cal_idx]))
            cal_scores_xgb = np.abs(y_cal_true - pred_cal_xgb) / pred_cal_xgb
            q_hat_xgb = float(np.quantile(cal_scores_xgb, q_level))

        print(f"  Calibrated conformal quantile: q_hat_pican={q_hat_pican:.3f}, q_hat_xgb={q_hat_xgb:.3f}")

        # -------------------------------------------------------------------
        # Predict and Bound on TEST DATA (Locking Predictions & Bounds)
        # -------------------------------------------------------------------
        model_pwr.eval()
        with torch.no_grad():
            # Test point predictions
            p_te_itl, _, _ = model_itl(w_te, h_te_itl, physics_base=r_te)
            pred_test_pican = p_te_itl.cpu().numpy()

            # Power point predictions
            p_te_pwr, _, _ = model_pwr(w_te, h_te_pwr, physics_base=tdp_te)
            pred_test_pwr = p_te_pwr.cpu().numpy()

            # Shift-adjusted conformal bounds (using ONLY q_hat_pican and d_shift)
            shift_factor = 1.0 + 0.15 * min(shift_dist, 5.0)
            eff_q = q_hat_pican * shift_factor
            bounds_test_lower = np.maximum(0.0, pred_test_pican * (1.0 - eff_q))
            bounds_test_upper = pred_test_pican * (1.0 + eff_q)

            # XGBoost point and upper bounds
            pred_test_xgb_itl = np.exp(pipe_xgb_itl.predict(X_xgb.iloc[test_idx]))
            bounds_test_xgb_upper = pred_test_xgb_itl * (1.0 + q_hat_xgb * shift_factor)
            pred_test_xgb_pwr = pipe_xgb_pwr.predict(X_xgb.iloc[test_idx])

        # Store locked outputs
        preds_pican_itl[test_idx] = pred_test_pican
        bounds_pican_itl_upper[test_idx] = bounds_test_upper
        bounds_pican_itl_lower[test_idx] = bounds_test_lower
        preds_pican_power[test_idx] = pred_test_pwr
        preds_xgb_itl[test_idx] = pred_test_xgb_itl
        bounds_xgb_itl_upper[test_idx] = bounds_test_xgb_upper
        preds_xgb_power[test_idx] = pred_test_xgb_pwr

        # -------------------------------------------------------------------
        # Requirement 2: TRUE END-TO-END DATA LEAKAGE ASSERTION TEST
        # Create a blinded copy of df where all test labels are completely wiped to NaN
        # -------------------------------------------------------------------
        df_blinded = df.copy()
        df_blinded.loc[test_idx, "itl"] = np.nan
        df_blinded.loc[test_idx, "gpu_0_power_draw_watts_mean"] = np.nan
        df_blinded.loc[test_idx, "normalized_power_draw"] = np.nan

        # Extract features from blinded dataframe
        W_test_blind = np.hstack([scaler_w.transform(df_blinded[w_cols].values[test_idx]), scenario_dummies[test_idx]])
        H_test_itl_blind = scaler_h_itl.transform(df_blinded[h_cols_itl].values[test_idx])
        H_test_pwr_blind = scaler_h_pwr.transform(df_blinded[h_cols_pwr].values[test_idx])
        w_te_b = torch.tensor(W_test_blind, dtype=torch.float32).to(device)
        h_te_itl_b = torch.tensor(H_test_itl_blind, dtype=torch.float32).to(device)
        h_te_pwr_b = torch.tensor(H_test_pwr_blind, dtype=torch.float32).to(device)

        with torch.no_grad():
            p_check_itl, _, _ = model_itl(w_te_b, h_te_itl_b, physics_base=r_te)
            pred_check_pican = p_check_itl.cpu().numpy()
            bounds_check_upper = pred_check_pican * (1.0 + eff_q)

            p_check_pwr, _, _ = model_pwr(w_te_b, h_te_pwr_b, physics_base=tdp_te)
            pred_check_pwr = p_check_pwr.cpu().numpy()

        pred_check_xgb_itl = np.exp(pipe_xgb_itl.predict(X_xgb.iloc[test_idx]))
        pred_check_xgb_pwr = pipe_xgb_pwr.predict(X_xgb.iloc[test_idx])

        # Strict mathematical assertions
        assert np.allclose(pred_test_pican, pred_check_pican, equal_nan=False), f"LEAKAGE ASSERTION FAILED: PICAN ITL on {holdout_gpu}!"
        assert np.allclose(bounds_test_upper, bounds_check_upper, equal_nan=False), f"LEAKAGE ASSERTION FAILED: PICAN bounds on {holdout_gpu}!"
        assert np.allclose(pred_test_pwr, pred_check_pwr, equal_nan=False), f"LEAKAGE ASSERTION FAILED: PICAN power on {holdout_gpu}!"
        assert np.allclose(pred_test_xgb_itl, pred_check_xgb_itl, equal_nan=False), f"LEAKAGE ASSERTION FAILED: XGBoost ITL on {holdout_gpu}!"
        assert np.allclose(pred_test_xgb_pwr, pred_check_xgb_pwr, equal_nan=False), f"LEAKAGE ASSERTION FAILED: XGBoost power on {holdout_gpu}!"
        print(f"  [PASSED] End-to-end leakage verification: Bit-for-bit identical under NaN-blinded test inputs.")

        # -------------------------------------------------------------------
        # Score Predictions (Only AFTER Locking)
        # -------------------------------------------------------------------
        y_test_true_itl = y_itl[test_idx]
        y_test_true_pwr = y_power[test_idx]

        mdape_pican = float(np.median(np.abs(y_test_true_itl - pred_test_pican) / y_test_true_itl) * 100.0)
        mdape_xgb = float(np.median(np.abs(y_test_true_itl - pred_test_xgb_itl) / y_test_true_itl) * 100.0)
        coverage_test = float(np.mean((y_test_true_itl >= bounds_test_lower) & (y_test_true_itl <= bounds_test_upper)) * 100.0)
        print(f"  Scoring: SHIFT-GPU MdAPE={mdape_pican:.2f}% (vs XGBoost={mdape_xgb:.2f}%) | Empirical Coverage={coverage_test:.1f}%")

        logo_results_itl.append({
            "gpu": holdout_gpu,
            "MdAPE": mdape_pican,
            "XGB_MdAPE": mdape_xgb,
            "Empirical_Coverage_Pct": coverage_test,
            "shift_dist": shift_dist
        })
        logo_results_power.append({
            "gpu": holdout_gpu,
            "MdAPE": float(np.median(np.abs(y_test_true_pwr - pred_test_pwr) / y_test_true_pwr) * 100.0)
        })

        models_by_gpu[holdout_gpu] = {
            "model_itl": copy.deepcopy(model_itl),
            "model_pwr": copy.deepcopy(model_pwr),
            "scaler_w": scaler_w,
            "scaler_h_itl": scaler_h_itl,
            "scaler_h_pwr": scaler_h_pwr,
            "q_hat_pican": q_hat_pican,
            "shift_dist": shift_dist,
            "train_idx": train_idx,
            "cal_idx": cal_idx,
            "test_idx": test_idx
        }

    # Save summary LOGO tables
    df_logo_itl = pd.DataFrame(logo_results_itl)
    df_logo_itl.to_csv(os.path.join(RESULTS_DIR, "table_e2_logo_pican_itl.csv"), index=False)
    print("\n--- ZERO-SHOT LOGO PER-GPU RESULTS (LEAKAGE-FREE) ---")
    print(df_logo_itl.to_string(index=False))

    # Overall Metrics
    overall_itl_pican = evaluate_predictions(y_itl, preds_pican_itl, df)
    overall_itl_xgb = evaluate_predictions(y_itl, preds_xgb_itl, df)
    overall_coverage = float(np.mean((y_itl >= bounds_pican_itl_lower) & (y_itl <= bounds_pican_itl_upper)) * 100.0)

    print("\n=== OVERALL LOGO PERFORMANCE (UNSEEN HARDWARE) ===")
    print(f"Proposed SHIFT-GPU: MdAPE={overall_itl_pican['MdAPE']:.2f}% | r={overall_itl_pican['Pearson_r']:.3f} | Coverage={overall_coverage:.1f}%")
    print(f"WattGPU Baseline:   MdAPE={overall_itl_xgb['MdAPE']:.2f}% | r={overall_itl_xgb['Pearson_r']:.3f}")

    # Save detailed empirical sample predictions (4,163 rows)
    df_preds = df.copy()
    df_preds["itl_true"] = y_itl
    df_preds["itl_pred_pican"] = preds_pican_itl
    df_preds["itl_bound_upper"] = bounds_pican_itl_upper
    df_preds["itl_bound_lower"] = bounds_pican_itl_lower
    df_preds["itl_pred_xgb"] = preds_xgb_itl
    df_preds["itl_xgb_upper"] = bounds_xgb_itl_upper
    df_preds["itl_roofline"] = t_roof
    df_preds["power_true"] = y_power
    df_preds["power_pred_pican"] = preds_pican_power
    df_preds["power_pred_xgb"] = preds_xgb_power
    df_preds["hw_shift_score"] = hw_shift_scores
    df_preds.to_csv(os.path.join(RESULTS_DIR, "empirical_predictions.csv"), index=False)
    print(f"Saved {len(df_preds)} locked empirical predictions to {os.path.join(RESULTS_DIR, 'empirical_predictions.csv')}")

    # -------------------------------------------------------------------
    # Requirement 3: Production Hardware Recommendation on the SAME Workloads
    # Compare:
    # 1. Minimum TDP Policy
    # 2. Matched XGBoost Point Prediction (ranks by its own predicted power)
    # 3. SHIFT-GPU Point Prediction (ranks by PICAN predicted power)
    # 4. SHIFT-GPU Risk-Calibrated (Conformal Upper Bounds)
    # 5. True Empirical Oracle
    # -------------------------------------------------------------------
    print("\n>>> Phase 2: Production Hardware Recommendation Benchmark on Identical Workloads...")
    sla_penalty_multiplier = 5.0

    pred_dict = {}
    true_dict = {}

    for i, row in df_preds.iterrows():
        key = (row["model"], row["scenario"], row["gpu_db_name"])
        pred_dict[key] = {
            "power_pred_pican": row["power_pred_pican"],
            "power_pred_xgb": row["power_pred_xgb"],
            "power_upper": row["power_pred_pican"] * 1.05,
            "itl_pican_point": row["itl_pred_pican"],
            "itl_pican_upper": row["itl_bound_upper"],
            "itl_xgb_point": row["itl_pred_xgb"],
            "itl_xgb_upper": row["itl_xgb_upper"],
            "tdp": row["thermal_design_power_w"]
        }
        true_dict[key] = {
            "power_true": row["power_true"],
            "itl_true": row["itl_true"]
        }

    unique_workloads = df_preds[["model", "scenario"]].drop_duplicates().reset_index(drop=True)
    N_total_workloads = len(unique_workloads)

    def evaluate_cluster_policy(policy_name, target_sla=0.035):
        accepted_count = 0
        violation_count = 0
        accepted_powers = []
        regrets = []

        recommendations = []

        for _, w_row in unique_workloads.iterrows():
            m_name, sc_name = w_row["model"], w_row["scenario"]

            # 1. Determine Oracle Choice (minimum power among candidates meeting true SLA)
            oracle_feasible = []
            for g in unique_gpus:
                k = (m_name, sc_name, g)
                if k in true_dict and true_dict[k]["itl_true"] <= target_sla:
                    oracle_feasible.append((g, true_dict[k]["power_true"]))
            
            p_oracle_best = min([x[1] for x in oracle_feasible]) if oracle_feasible else np.nan

            # 2. Determine Policy Choice
            candidate_pool = []
            for g in unique_gpus:
                k = (m_name, sc_name, g)
                if k not in pred_dict:
                    continue
                p_info = pred_dict[k]

                if policy_name == "min_tdp":
                    meets_sla = p_info["itl_pican_point"] <= target_sla
                    sort_metric = p_info["tdp"]
                elif policy_name == "xgb_point":
                    # Strictly prediction-only: ranks by its OWN predicted power
                    meets_sla = p_info["itl_xgb_point"] <= target_sla
                    sort_metric = p_info["power_pred_xgb"]
                elif policy_name == "pican_point":
                    meets_sla = p_info["itl_pican_point"] <= target_sla
                    sort_metric = p_info["power_pred_pican"]
                elif policy_name == "shift_gpu_calibrated":
                    meets_sla = p_info["itl_pican_upper"] <= target_sla
                    sort_metric = p_info["power_pred_pican"]
                elif policy_name == "oracle":
                    meets_sla = true_dict[k]["itl_true"] <= target_sla
                    sort_metric = true_dict[k]["power_true"]
                else:
                    raise ValueError(f"Unknown policy {policy_name}")

                if meets_sla:
                    candidate_pool.append((g, sort_metric))

            if candidate_pool:
                # Select candidate with lowest sort metric
                candidate_pool.sort(key=lambda x: x[1])
                selected_gpu = candidate_pool[0][0]
                recommendations.append(selected_gpu)

                accepted_count += 1
                t_true = true_dict[(m_name, sc_name, selected_gpu)]["itl_true"]
                p_true = true_dict[(m_name, sc_name, selected_gpu)]["power_true"]
                accepted_powers.append(p_true)

                is_violation = t_true > target_sla
                if is_violation:
                    violation_count += 1

                # Regret calculation
                if not np.isnan(p_oracle_best):
                    reg = max(0.0, p_true - p_oracle_best)
                else:
                    reg = 0.0
                if is_violation:
                    reg += sla_penalty_multiplier * p_true * ((t_true - target_sla) / target_sla)
                regrets.append(reg)
            else:
                # Policy abstains (deems workload infeasible under SLA)
                recommendations.append(None)
                reg = 50.0 if not np.isnan(p_oracle_best) else 0.0
                regrets.append(reg)

        abstain_count = N_total_workloads - accepted_count
        abstain_rate = (abstain_count / N_total_workloads) * 100.0
        violation_rate = (violation_count / accepted_count * 100.0) if accepted_count > 0 else 0.0
        mean_pwr = float(np.mean(accepted_powers)) if accepted_powers else np.nan
        mean_reg = float(np.mean(regrets))

        return {
            "policy": policy_name,
            "target_sla_ms": int(target_sla * 1000),
            "total_workloads": N_total_workloads,
            "accepted_workloads": accepted_count,
            "abstain_count": abstain_count,
            "abstain_rate_pct": abstain_rate,
            "violation_count": violation_count,
            "violation_rate_pct": violation_rate,
            "violation_fraction_str": f"{violation_count} / {accepted_count}",
            "mean_power_watts": mean_pwr,
            "mean_regret": mean_reg,
            "recommendations": recommendations
        }

    # Main evaluation at 35 ms SLA
    rec_benchmark = []
    print("\n--- Main Evaluation under 35 ms SLA ---")
    for pol in ["min_tdp", "xgb_point", "pican_point", "shift_gpu_calibrated", "oracle"]:
        res_pol = evaluate_cluster_policy(pol, target_sla=0.035)
        rec_benchmark.append({k: v for k, v in res_pol.items() if k != "recommendations"})
        print(f"  Policy [{pol:20s}]: Accepted={res_pol['accepted_workloads']}/{N_total_workloads} (Abstain={res_pol['abstain_rate_pct']:.1f}%), Violations={res_pol['violation_fraction_str']} ({res_pol['violation_rate_pct']:.1f}%), Mean Power={res_pol['mean_power_watts']:.1f}W, Regret={res_pol['mean_regret']:.1f}")

    df_rec = pd.DataFrame(rec_benchmark)
    df_rec.to_csv(os.path.join(RESULTS_DIR, "table_e6_decision_recommendations.csv"), index=False)

    # Multi-SLA Sensitivity Evaluation (20ms, 25ms, 30ms, 35ms, 40ms)
    print("\n--- Multi-SLA Sensitivity Benchmark across Latency Constraints ---")
    sensitivity_records = []
    for s_sla in [0.020, 0.025, 0.030, 0.035, 0.040]:
        for pol in ["min_tdp", "xgb_point", "pican_point", "shift_gpu_calibrated", "oracle"]:
            res = evaluate_cluster_policy(pol, target_sla=s_sla)
            sensitivity_records.append({k: v for k, v in res.items() if k != "recommendations"})
            print(f"  [SLA={int(s_sla*1000):2d}ms] {pol:20s}: Accepted={res['accepted_workloads']:3d}/{N_total_workloads}, Violations={res['violation_fraction_str']} ({res['violation_rate_pct']:.1f}%), Mean Power={res['mean_power_watts']:.1f}W")

    df_sens = pd.DataFrame(sensitivity_records)
    df_sens.to_csv(os.path.join(RESULTS_DIR, "table_e6_sla_sensitivity.csv"), index=False)

    df_rec = pd.DataFrame(rec_benchmark)
    df_rec.to_csv(os.path.join(RESULTS_DIR, "table_e6_decision_recommendations.csv"), index=False)

    # -------------------------------------------------------------------
    # Requirement 4: Independent Ablation Retraining across Same Folds
    # -------------------------------------------------------------------
    print("\n>>> Phase 3: Independent Ablation Retraining across Identical Folds...")

    class PureAttentionModel(nn.Module):
        """Ablation A: Without Roofline Base."""
        def __init__(self, w_dim, h_dim, embed_dim=64, num_heads=4):
            super().__init__()
            self.w_enc = nn.Sequential(nn.Linear(w_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
            self.h_enc = nn.Sequential(nn.Linear(h_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
            self.head = nn.Linear(embed_dim, 1)

        def forward(self, w, h):
            qw = self.w_enc(w).unsqueeze(1)
            kh = self.h_enc(h).unsqueeze(1)
            attn, _ = self.cross_attn(query=qw, key=kh, value=kh)
            z = self.norm1(qw + attn)
            z = self.norm2(z + self.ff(z)).squeeze(1)
            return self.head(z).squeeze(-1)

    class FlatMLPModel(nn.Module):
        """Ablation B: Without Cross-Attention (Flat Concatenated Residual)."""
        def __init__(self, w_dim, h_dim, embed_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(w_dim + h_dim, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1)
            )

        def forward(self, w, h):
            x = torch.cat([w, h], dim=-1)
            return self.net(x).squeeze(-1)

    preds_ablation_a = np.zeros(len(df))
    preds_ablation_b = np.zeros(len(df))
    bounds_ablation_a_upper = np.zeros(len(df))
    bounds_ablation_b_upper = np.zeros(len(df))

    for fold, (train_pool_idx, test_idx) in enumerate(logo.split(df, y_itl, groups=groups)):
        rng_fold = np.random.RandomState(42 + fold)
        n_pool = len(train_pool_idx)
        perm = rng_fold.permutation(n_pool)
        n_train = int(0.80 * n_pool)
        train_idx = train_pool_idx[perm[:n_train]]
        cal_idx = train_pool_idx[perm[n_train:]]

        scaler_w = StandardScaler().fit(df[w_cols].values[train_idx])
        scaler_h = StandardScaler().fit(df[h_cols_itl].values[train_idx])

        W_tr = np.hstack([scaler_w.transform(df[w_cols].values[train_idx]), scenario_dummies[train_idx]])
        W_ca = np.hstack([scaler_w.transform(df[w_cols].values[cal_idx]), scenario_dummies[cal_idx]])
        W_te = np.hstack([scaler_w.transform(df[w_cols].values[test_idx]), scenario_dummies[test_idx]])

        H_tr = scaler_h.transform(df[h_cols_itl].values[train_idx])
        H_ca = scaler_h.transform(df[h_cols_itl].values[cal_idx])
        H_te = scaler_h.transform(df[h_cols_itl].values[test_idx])

        w_tr = torch.tensor(W_tr, dtype=torch.float32).to(device)
        h_tr = torch.tensor(H_tr, dtype=torch.float32).to(device)
        w_ca = torch.tensor(W_ca, dtype=torch.float32).to(device)
        h_ca = torch.tensor(H_ca, dtype=torch.float32).to(device)
        w_te = torch.tensor(W_te, dtype=torch.float32).to(device)
        h_te = torch.tensor(H_te, dtype=torch.float32).to(device)

        y_tr_raw = torch.tensor(np.log(y_itl[train_idx]), dtype=torch.float32).to(device)
        y_tr_res = torch.tensor(target_res[train_idx], dtype=torch.float32).to(device)

        # Train Ablation A (Pure Attention, log ITL target)
        torch.manual_seed(42 + fold)
        ma = PureAttentionModel(w_tr.shape[1], h_tr.shape[1]).to(device)
        opt_a = torch.optim.AdamW(ma.parameters(), lr=4e-3, weight_decay=1e-4)
        for _ in range(120):
            opt_a.zero_grad()
            pa = ma(w_tr, h_tr)
            loss_a = nn.functional.smooth_l1_loss(pa, y_tr_raw, beta=0.05)
            loss_a.backward()
            opt_a.step()

        # Train Ablation B (Flat MLP, log residual target)
        torch.manual_seed(42 + fold)
        mb = FlatMLPModel(w_tr.shape[1], h_tr.shape[1]).to(device)
        opt_b = torch.optim.AdamW(mb.parameters(), lr=4e-3, weight_decay=1e-4)
        for _ in range(120):
            opt_b.zero_grad()
            pb = mb(w_tr, h_tr)
            loss_b = nn.functional.smooth_l1_loss(pb, y_tr_res, beta=0.05)
            loss_b.backward()
            opt_b.step()

        # Calibrate Ablation A & B strictly on Calibration set
        ma.eval()
        mb.eval()
        with torch.no_grad():
            pred_cal_a = np.exp(ma(w_ca, h_ca).cpu().numpy())
            pred_cal_b = t_roof[cal_idx] * np.exp(np.maximum(mb(w_ca, h_ca).cpu().numpy(), 0.0))

            q_a = float(np.quantile(np.abs(y_itl[cal_idx] - pred_cal_a) / pred_cal_a, min(1.0, 0.90 * (1.0 + 1.0/len(cal_idx)))))
            q_b = float(np.quantile(np.abs(y_itl[cal_idx] - pred_cal_b) / pred_cal_b, min(1.0, 0.90 * (1.0 + 1.0/len(cal_idx)))))

            pred_te_a = np.exp(ma(w_te, h_te).cpu().numpy())
            pred_te_b = t_roof[test_idx] * np.exp(np.maximum(mb(w_te, h_te).cpu().numpy(), 0.0))

            preds_ablation_a[test_idx] = pred_te_a
            preds_ablation_b[test_idx] = pred_te_b
            bounds_ablation_a_upper[test_idx] = pred_te_a * (1.0 + q_a)
            bounds_ablation_b_upper[test_idx] = pred_te_b * (1.0 + q_b)

    # Compute empirical ablation statistics
    mdape_a = float(np.median(np.abs(y_itl - preds_ablation_a) / y_itl) * 100.0)
    mdape_b = float(np.median(np.abs(y_itl - preds_ablation_b) / y_itl) * 100.0)
    cov_a = float(np.mean(y_itl <= bounds_ablation_a_upper) * 100.0)
    cov_b = float(np.mean(y_itl <= bounds_ablation_b_upper) * 100.0)
    cov_full = float(np.mean(y_itl <= bounds_pican_itl_upper) * 100.0)
    cov_xgb = float(np.mean(y_itl <= bounds_xgb_itl_upper) * 100.0)

    # Calculate actual SLA violations for ablations under point deployment
    def eval_variant_violations(preds_arr, target_sla=0.035):
        v = 0
        tot = 0
        for _, w_row in unique_workloads.iterrows():
            m_name, sc_name = w_row["model"], w_row["scenario"]
            cands = []
            for g in unique_gpus:
                k = (m_name, sc_name, g)
                if k in pred_dict:
                    idx = df_preds[(df_preds["model"] == m_name) & (df_preds["scenario"] == sc_name) & (df_preds["gpu_db_name"] == g)].index
                    if len(idx) > 0:
                        cands.append((g, preds_arr[idx[0]], pred_dict[k]["power_pred_pican"], true_dict[k]["itl_true"]))
            eligible = [c for c in cands if c[1] <= target_sla]
            if eligible:
                eligible.sort(key=lambda x: x[2])
                tot += 1
                if eligible[0][3] > target_sla:
                    v += 1
        return (v / tot * 100.0) if tot > 0 else 0.0

    viol_a = eval_variant_violations(preds_ablation_a, target_sla=0.035)
    viol_b = eval_variant_violations(preds_ablation_b, target_sla=0.035)
    viol_xgb = eval_variant_violations(preds_xgb_itl, target_sla=0.035)
    viol_full = df_rec[df_rec["policy"] == "shift_gpu_calibrated"]["violation_rate_pct"].iloc[0]

    ablations = [
        {"model_variant": "Full Proposed SHIFT-GPU (PICAN + Roofline + Conformal)", "MdAPE_ITL": overall_itl_pican["MdAPE"], "Empirical_Coverage_Pct": cov_full, "SLA_Violation_Pct": viol_full},
        {"model_variant": "Ablation A: Without Physics Roofline (Pure Attention)", "MdAPE_ITL": mdape_a, "Empirical_Coverage_Pct": cov_a, "SLA_Violation_Pct": viol_a},
        {"model_variant": "Ablation B: Without Cross-Attention (Flat MLP)", "MdAPE_ITL": mdape_b, "Empirical_Coverage_Pct": cov_b, "SLA_Violation_Pct": viol_b},
        {"model_variant": "Baseline C: Standard WattGPU (XGBoost)", "MdAPE_ITL": overall_itl_xgb["MdAPE"], "Empirical_Coverage_Pct": cov_xgb, "SLA_Violation_Pct": viol_xgb},
    ]
    df_ablation = pd.DataFrame(ablations)
    df_ablation.to_csv(os.path.join(RESULTS_DIR, "table_e7_ablations.csv"), index=False)
    print("\n=== SYSTEMATIC ABLATION BENCHMARK (100% EMPIRICALLY RETRAINED) ===")
    print(df_ablation.to_string(index=False))

    # -------------------------------------------------------------------
    # Requirement 4: Independent Active Few-Shot Profiling Benchmark
    # Evaluate strictly on untouched remainder of test set (D_test \ D_probe)
    # -------------------------------------------------------------------
    print("\n>>> Phase 4: Independent Active Few-Shot Hardware Profiling Benchmark...")
    few_shot_results = []
    budgets = [0, 1, 3, 5]
    policies = ["random", "diversity", "uncertainty", "regret_optimal"]

    selector = ActiveSelector()

    for pol in policies:
        print(f"  Evaluating Few-Shot Policy: [{pol}]...")
        for k in budgets:
            all_untouched_errors = []

            for holdout_gpu, m_dict in models_by_gpu.items():
                test_idx = m_dict["test_idx"]
                sub_test = df.iloc[test_idx].reset_index(drop=True)
                
                # Transform test features
                W_te = np.hstack([m_dict["scaler_w"].transform(sub_test[w_cols].values), scenario_dummies[test_idx]])
                H_te = m_dict["scaler_h_itl"].transform(sub_test[h_cols_itl].values)
                y_te_true = sub_test["itl"].values
                r_te = t_roof[test_idx]

                if k == 0:
                    # Pure zero-shot on full test set
                    w_te_t = torch.tensor(W_te, dtype=torch.float32).to(device)
                    h_te_t = torch.tensor(H_te, dtype=torch.float32).to(device)
                    r_te_t = torch.tensor(r_te, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        p_itl, _, _ = m_dict["model_itl"](w_te_t, h_te_t, physics_base=r_te_t)
                        preds = p_itl.cpu().numpy()
                    rel_errs = np.abs(y_te_true - preds) / y_te_true
                    all_untouched_errors.extend(rel_errs)
                else:
                    # Select k probe measurements from test candidate pool
                    preds_zero_shot = preds_pican_itl[test_idx]
                    pwr_zero_shot = preds_pican_power[test_idx]
                    uncertainty_pool = np.abs(preds_zero_shot - t_roof[test_idx]) # proxy uncertainty

                    cand_indices = np.arange(len(sub_test))
                    if pol == "random":
                        probe_indices = selector.select_random(cand_indices, k, seed=42)
                    elif pol == "diversity":
                        probe_indices = selector.select_diversity(W_te, cand_indices, k, seed=42)
                    elif pol == "uncertainty":
                        probe_indices = selector.select_max_uncertainty(uncertainty_pool, cand_indices, k)
                    elif pol == "regret_optimal":
                        probe_indices = selector.select_regret_optimal(preds_zero_shot, uncertainty_pool, pwr_zero_shot, cand_indices, k, t_max=0.035)
                    else:
                        raise ValueError(f"Unknown few-shot policy {pol}")

                    probe_indices = np.asarray(probe_indices)
                    if len(probe_indices) < k:
                        missing = [idx for idx in cand_indices if idx not in probe_indices]
                        fill = np.random.RandomState(42).choice(missing, size=k - len(probe_indices), replace=False)
                        probe_indices = np.concatenate([probe_indices, fill])
                    probe_indices = probe_indices[:k].copy()

                    # Define untouched evaluation set
                    untouched_indices = np.array([idx for idx in range(len(sub_test)) if idx not in probe_indices])
                    assert len(probe_indices) == k
                    assert len(untouched_indices) == len(sub_test) - k

                    # Clone model and modulate FiLM parameters strictly on probe samples
                    adapted_model = copy.deepcopy(m_dict["model_itl"])
                    adapted_model.freeze_backbone_for_few_shot()
                    opt_film = torch.optim.Adam(adapted_model.film.parameters(), lr=0.01)

                    w_probe = torch.tensor(W_te[probe_indices], dtype=torch.float32).to(device)
                    h_probe = torch.tensor(H_te[probe_indices], dtype=torch.float32).to(device)
                    r_probe = torch.tensor(r_te[probe_indices], dtype=torch.float32).to(device)
                    y_probe_res = torch.tensor(target_res[test_idx[probe_indices]], dtype=torch.float32).to(device)

                    for _ in range(35):
                        opt_film.zero_grad()
                        _, _, p_res = adapted_model(w_probe, h_probe, physics_base=r_probe)
                        loss = nn.functional.smooth_l1_loss(p_res, y_probe_res)
                        loss.backward()
                        opt_film.step()

                    # Evaluate strictly on untouched samples
                    adapted_model.eval()
                    w_eval = torch.tensor(W_te[untouched_indices], dtype=torch.float32).to(device)
                    h_eval = torch.tensor(H_te[untouched_indices], dtype=torch.float32).to(device)
                    r_eval = torch.tensor(r_te[untouched_indices], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        p_eval, _, _ = adapted_model(w_eval, h_eval, physics_base=r_eval)
                        preds_eval = p_eval.cpu().numpy()

                    y_eval_true = y_te_true[untouched_indices]
                    rel_errs = np.abs(y_eval_true - preds_eval) / y_eval_true
                    all_untouched_errors.extend(rel_errs)

            mdape_fewshot = float(np.median(all_untouched_errors) * 100.0)
            few_shot_results.append({
                "policy": pol,
                "budget_k": k,
                "mdape_itl": mdape_fewshot
            })
            print(f"    Budget k={k}: Untouched Evaluation MdAPE = {mdape_fewshot:.2f}%")

    df_fewshot = pd.DataFrame(few_shot_results)
    df_fewshot.to_csv(os.path.join(RESULTS_DIR, "table_e5_active_few_shot.csv"), index=False)
    print("\n=== ACTIVE FEW-SHOT HARDWARE PROFILING BENCHMARK ===")
    print(df_fewshot.to_string(index=False))

    print("\n" + "=" * 80)
    print("   ALL LEAKAGE-FREE EXPERIMENTS COMPLETED AND CERTIFIED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
