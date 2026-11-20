"""
Physics-Guided Execution Layer for SHIFT-GPU.
Implements non-parametric microarchitectural Roofline latency bounds
and physical power baseline approximations.
"""

import numpy as np
import pandas as pd
import torch

class PhysicsEngine:
    @staticmethod
    def compute_roofline_latency(total_b_params, memory_bandwidth_gb_s, tflops_16b, scenario_or_batch=1.0):
        """
        Computes the fundamental Roofline lower bound for generating one token (seconds).
        For offline batched inference in vLLM, default max sequence parallelism B = 256.
        For server scenarios (low/high Poisson arrivals), B = 1.0.
        T_mem = (2 bytes * total_params) / (Memory_Bandwidth * B)
        """
        if isinstance(scenario_or_batch, (str, list, np.ndarray, pd.Series)):
            if isinstance(scenario_or_batch, str):
                b = 256.0 if scenario_or_batch == "offline" else 1.0
            else:
                b = np.where(np.array(scenario_or_batch) == "offline", 256.0, 1.0)
        else:
            b = float(scenario_or_batch)

        if isinstance(total_b_params, torch.Tensor):
            b_tensor = torch.as_tensor(b, dtype=torch.float32, device=total_b_params.device)
            t_mem = (2.0 * total_b_params) / (torch.clamp(memory_bandwidth_gb_s, min=1.0) * b_tensor)
            flops_per_token = 2.0 * total_b_params * 1e9
            peak_flops = torch.clamp(tflops_16b, min=1.0) * 1e12
            t_comp = flops_per_token / peak_flops
            return torch.maximum(t_mem, t_comp)
        else:
            t_mem = (2.0 * np.array(total_b_params)) / (np.maximum(np.array(memory_bandwidth_gb_s), 1.0) * b)
            flops_per_token = 2.0 * np.array(total_b_params) * 1e9
            peak_flops = np.maximum(np.array(tflops_16b), 1.0) * 1e12
            t_comp = flops_per_token / peak_flops
            return np.maximum(t_mem, t_comp)

    @staticmethod
    def compute_power_base(tdp_watts, scenario="server_high"):
        """
        Coarse physical baseline power estimation based on thermal envelopes
        and operational scenario duty cycles.
        """
        if scenario == "offline":
            fraction = 0.88  # Saturation near peak TDP in continuous batching
        elif scenario == "server_high":
            fraction = 0.65  # Dynamic load with active GPU idling between requests
        else:
            fraction = 0.40  # Low Poisson arrival rate
        return tdp_watts * fraction
