"""
Data loading and feature decomposition module for SHIFT-GPU.
Disentangles raw LLM metadata, GPU microarchitecture parameters,
operational workloads, and target metrics (Power, ITL).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

DATA_DIR = "wattgpu_upstream/data"

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

def get_processed_dataset(data_dir=DATA_DIR):
    df_all = pd.read_csv(os.path.join(data_dir, "watt_counts_subset.csv"), index_col=0)
    for col in ["time_to_first_token_seconds_events",
                "e2e_request_latency_seconds_events",
                "request_generation_tokens"]:
        convert_column_to_list(df_all, col)

    df_all["throughput"] = df_all["generation_tokens_total"] / df_all["measurement_duration"]
    df_all["energy_per_token"] = df_all["gpu_energy"] / df_all["generation_tokens_total"]

    df_models = pd.read_csv(os.path.join(data_dir, "model_features.csv"), sep=";", index_col=0)
    df_models = df_models[df_models.columns[df_models.isna().sum() / len(df_models) < 0.1]]
    df_all = df_all.merge(df_models, left_on='model', right_index=True, how='left')

    df_gpus = pd.read_csv(os.path.join(data_dir, "gpu_features.csv"), index_col=0, sep=";")
    df_all = df_all.merge(df_gpus, left_on='gpu_type', right_index=True, how='left')

    # Operational scenario
    df_all.loc[df_all.lambda_qps < 0.2, "scenario"] = "server_low"
    df_all.loc[df_all.lambda_qps > 0.2, "scenario"] = "server_high"
    df_all.loc[df_all.lambda_qps.isna(), "scenario"] = "offline"
    df_all = df_all[df_all.generation_tokens_total != 0]

    # Explicit FP16 TFLOPS specifications
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

    # Focus on dense LLMs with verified parameter active == total
    df_all = df_all[df_all.total_b_params == df_all.active_b_params].copy()

    # Derived physics features
    df_all["normalized_power_draw"] = df_all["gpu_0_power_draw_watts_mean"] / df_all["thermal_design_power_w"]
    df_all["boost_percentage"] = df_all["boost_clock_mhz"] / df_all["base_clock_mhz"]
    df_all["log_total_b_params"] = np.log(df_all.total_b_params)
    df_all["bandwidth_latency"] = (df_all["total_b_params"] * 2) / df_all["memory_bandwidth_gb_s"]
    flops_per_token = 2 * df_all["total_b_params"] * 1e9
    compute_flops_per_s = df_all["tflops_16b"] * 1e12
    df_all["compute_latency_s"] = flops_per_token / compute_flops_per_s

    # KV Cache bytes calculation
    head_dim = df_all["hidden_size"] / df_all["num_attention_heads"]
    kv_per_token = (
        2
        * df_all["num_key_value_heads"]
        * head_dim
        * 2
        * df_all["num_layers"]
    )
    df_all["kv_cache_size_bytes"] = kv_per_token * (512 + 128)

    # ITL extraction
    df_all.loc[df_all.scenario == "offline", "lambda_qps"] = 1000
    offline_mask = df_all.scenario == "offline"
    df_all.loc[offline_mask, "itl"] = (
        df_all.loc[offline_mask, "measurement_duration"]
        / df_all.loc[offline_mask, "generation_tokens_total"]
    )
    server_mask = ~offline_mask
    df_all.loc[server_mask, "itl"] = df_all.loc[server_mask].apply(compute_itl_per_request, axis=1)

    # Drop missing target values
    df_all = df_all.dropna(subset=["normalized_power_draw", "itl", "gpu_0_power_draw_watts_mean"])
    return df_all

# Explicit feature sets
WORKLOAD_FEATURES = [
    "total_b_params",
    "log_total_b_params",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "num_layers",
    "kv_cache_size_bytes",
    "scenario",
    "model_type",
]

HARDWARE_FEATURES = [
    "memory_bandwidth_gb_s",
    "memory_size_gb",
    "tflops_16b",
    "thermal_design_power_w",
    "base_clock_mhz",
    "boost_percentage",
    "transistor_count_m",
    "release_year",
    "memory_type",
    "architecture",
]

PHYSICAL_INTERACTION_FEATURES = [
    "bandwidth_latency",
    "compute_latency_s",
]
