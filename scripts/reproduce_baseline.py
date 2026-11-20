"""
Script to reproduce the official WattGPU baseline (IJCAI 2026 workshop paper)
Evaluates Mean Power Draw and Inter-Token Latency (ITL) under:
- 5-Fold Cross Validation (CV)
- Leave-One-GPU-Out (LOGO)
- Leave-One-LLM-Out (LOLO)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_FOLDER = "wattgpu_upstream/data"

def convert_column_to_list(df, column):
    df[column] = df[column].apply(lambda x: eval(x) if isinstance(x, str) else x)

def compute_itl_per_request(row):
    e2e = np.array(row['e2e_request_latency_seconds_events'])
    ttft = np.array(row['time_to_first_token_seconds_events'])
    tokens = np.array(row['request_generation_tokens'])
    valid = tokens >= 2
    if not valid.any():
        return np.nan
    itl_per_req = (e2e[valid] - ttft[valid]) / (tokens[valid] - 1)
    return itl_per_req.mean()

def load_and_preprocess_data():
    print("Loading raw data...")
    df_all = pd.read_csv(os.path.join(DATA_FOLDER, "watt_counts_subset.csv"), index_col=0)
    for col in ["time_to_first_token_seconds_events",
                "e2e_request_latency_seconds_events",
                "request_generation_tokens"]:
        convert_column_to_list(df_all, col)

    df_all["throughput"] = df_all["generation_tokens_total"] / df_all["measurement_duration"]
    df_all["energy_per_token"] = df_all["gpu_energy"] / df_all["generation_tokens_total"]

    df_models = pd.read_csv(os.path.join(DATA_FOLDER, "model_features.csv"), sep=";", index_col=0)
    df_models = df_models[df_models.columns[df_models.isna().sum() / len(df_models) < 0.1]]
    df_all = df_all.merge(df_models, left_on='model', right_index=True, how='left')

    df_gpus = pd.read_csv(os.path.join(DATA_FOLDER, "gpu_features.csv"), index_col=0, sep=";")
    df_all = df_all.merge(df_gpus, left_on='gpu_type', right_index=True, how='left')

    df_all.loc[df_all.lambda_qps < 0.2, "scenario"] = "server_low"
    df_all.loc[df_all.lambda_qps > 0.2, "scenario"] = "server_high"
    df_all.loc[df_all.lambda_qps.isna(), "scenario"] = "offline"
    df_all = df_all[df_all.generation_tokens_total != 0]

    gpu_tflops = {
        "Tesla V100 SXM2 32 GB": 125,
        "Tesla T4": 65,
        "A100 SXM4 40 GB": 312,
        "A30 PCIe": 165,
        "L40S": 362,
        "L4": 121,
        "H200 NVL": 835,
        "H100 NVL 94 GB": 835,
    }
    for gpu_name, tflops in gpu_tflops.items():
        df_all.loc[df_all.gpu_db_name == gpu_name, "tflops_16b"] = tflops

    # Exclude MoE
    df_all = df_all[df_all.total_b_params == df_all.active_b_params].copy()

    # Features
    df_all["normalized_power_draw"] = df_all["gpu_0_power_draw_watts_mean"] / df_all["thermal_design_power_w"]
    df_all["boost_percentage"] = df_all["boost_clock_mhz"] / df_all["base_clock_mhz"]
    df_all["log_total_b_params"] = np.log(df_all.total_b_params)
    df_all["bandwidth_latency"] = (df_all["total_b_params"] * 2) / df_all["memory_bandwidth_gb_s"]
    flops_per_token = 2 * df_all["total_b_params"] * 1e9
    compute_flops_per_s = df_all["tflops_16b"] * 1e12
    df_all["compute_latency_s"] = flops_per_token / compute_flops_per_s

    # Memory features
    df_all["estimated_free_memory"] = df_all["memory_size_gb"] - df_all["total_b_params"] * 2
    df_all["estimated_available_memory"] = df_all["memory_size_gb"] * 0.9 - df_all["total_b_params"] * 2

    # ITL
    df_all.loc[df_all.scenario == "offline", "lambda_qps"] = 1000
    offline_mask = df_all.scenario == "offline"
    df_all.loc[offline_mask, "itl"] = (
        df_all.loc[offline_mask, "measurement_duration"]
        / df_all.loc[offline_mask, "generation_tokens_total"]
    )
    server_mask = ~offline_mask
    df_all.loc[server_mask, "itl"] = df_all.loc[server_mask].apply(compute_itl_per_request, axis=1)

    print(f"Preprocessed dataset ready with {len(df_all)} samples.")
    return df_all

def clean_df(df):
    def is_scalar_column(col):
        return df[col].dropna().apply(
            lambda x: isinstance(x, (int, float, str, bool, np.number))
        ).all()
    scalar_cols = [col for col in df.columns if is_scalar_column(col)]
    df_clean = df[scalar_cols].dropna(axis=1, how='all')
    df_clean = df_clean.loc[:, df_clean.nunique(dropna=True) > 1]
    df_clean = df_clean.dropna(axis=0, how='any')
    return df_clean

def _build_pipeline(X, regressor):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

def run_cv_experiment(df, group_col, target_col, regressor, log_transform_y=False):
    drop_keys = ["gpu_db_name", "model"]
    feature_cols = [c for c in df.columns if c not in drop_keys and c != target_col]
    X = df[feature_cols]
    y = df[target_col].copy()
    if log_transform_y:
        y = np.log(y)

    if group_col is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X))
    else:
        cv = LeaveOneGroupOut()
        splits = list(cv.split(X, groups=df[group_col]))

    predictions = pd.DataFrame(index=df.index)
    predictions["y_true"] = df[target_col]
    predictions["y_pred"] = np.nan
    predictions["gpu_db_name"] = df["gpu_db_name"]
    predictions["model"] = df["model"]
    predictions["scenario"] = df["scenario"]

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        pipe = _build_pipeline(X_train, regressor)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        if log_transform_y:
            pred = np.exp(pred)
        predictions.iloc[test_idx, predictions.columns.get_loc("y_pred")] = pred

    return predictions

def compute_metrics(df_preds):
    y_true = df_preds["y_true"].values
    y_pred = df_preds["y_pred"].values
    
    # MdAPE
    ape = np.abs((y_true - y_pred) / y_true) * 100
    mdape = np.median(ape)
    
    # Pearson r
    r, _ = pearsonr(y_true, y_pred)
    
    # Kendalls
    # GPU ranking per model
    gpu_taus = []
    for _, g in df_preds.groupby("model"):
        if len(g) > 1 and g["y_true"].nunique() > 1 and g["y_pred"].nunique() > 1:
            tau, _ = kendalltau(g["y_true"], g["y_pred"])
            if not np.isnan(tau):
                gpu_taus.append(tau)
    gpu_tau = np.mean(gpu_taus) if gpu_taus else np.nan

    # LLM ranking per GPU
    llm_taus = []
    for _, g in df_preds.groupby("gpu_db_name"):
        if len(g) > 1 and g["y_true"].nunique() > 1 and g["y_pred"].nunique() > 1:
            tau, _ = kendalltau(g["y_true"], g["y_pred"])
            if not np.isnan(tau):
                llm_taus.append(tau)
    llm_tau = np.mean(llm_taus) if llm_taus else np.nan

    return {"MdAPE": mdape, "Pearson_r": r, "GPU_tau": gpu_tau, "LLM_tau": llm_tau}

if __name__ == "__main__":
    df_all = load_and_preprocess_data()
    
    # Power Draw Evaluation
    cols_power = [
        'gpu_db_name', 'model', 'normalized_power_draw', 'scenario',
        'log_total_b_params', 'model_type', 'num_key_value_heads', 'num_layers',
        'memory_bandwidth_gb_s', 'boost_percentage', 'transistor_count_m',
        'base_clock_mhz', 'release_year', 'bandwidth_latency'
    ]
    df_power = clean_df(df_all.loc[:, cols_power])
    xgb_power = XGBRegressor(max_depth=6, reg_lambda=150, n_estimators=200, random_state=42)
    
    print("\n================ EVALUATING POWER DRAW (XGBoost) ================")
    for strategy, grp in [("5-Fold CV", None), ("LOGO (Unseen GPU)", "gpu_db_name"), ("LOLO (Unseen LLM)", "model")]:
        preds = run_cv_experiment(df_power, grp, "normalized_power_draw", xgb_power, log_transform_y=False)
        m = compute_metrics(preds)
        print(f"Power [{strategy}]: MdAPE={m['MdAPE']:.2f}%, Pearson r={m['Pearson_r']:.3f}, GPU tau={m['GPU_tau']:.2f}, LLM tau={m['LLM_tau']:.2f}")

    # ITL Evaluation
    cols_itl = [
        'gpu_db_name', 'model', 'itl', 'scenario', 'model_type',
        'total_b_params', 'hidden_size', 'num_attention_heads', 'num_key_value_heads',
        'num_layers', 'memory_bandwidth_gb_s', 'memory_size_gb', 'memory_type',
        'tflops_16b', 'release_year', 'bandwidth_latency', 'compute_latency_s'
    ]
    df_itl = clean_df(df_all.loc[:, cols_itl])
    xgb_itl = XGBRegressor(max_depth=5, reg_lambda=100, n_estimators=100, random_state=42)
    
    print("\n================ EVALUATING ITL (XGBoost) ================")
    for strategy, grp in [("5-Fold CV", None), ("LOGO (Unseen GPU)", "gpu_db_name"), ("LOLO (Unseen LLM)", "model")]:
        preds = run_cv_experiment(df_itl, grp, "itl", xgb_itl, log_transform_y=True)
        m = compute_metrics(preds)
        print(f"ITL [{strategy}]: MdAPE={m['MdAPE']:.2f}%, Pearson r={m['Pearson_r']:.3f}, GPU tau={m['GPU_tau']:.2f}, LLM tau={m['LLM_tau']:.2f}")
