"""
Decision Engine for Risk-Constrained Hardware Recommendation.
Evaluates deployment decisions under SLA constraints:
min_g P_hat(x, g)  s.t.  T_hat_upper(x, g) <= T_max
"""

import numpy as np
import pandas as pd

class DecisionEngine:
    def __init__(self, t_max=0.030, sla_penalty_factor=5.0):
        """
        t_max: Latency Service Level Agreement (seconds per token, e.g. 30ms or 50ms)
        sla_penalty_factor: Multiplier penalizing SLA violations in regret calculation
        """
        self.t_max = t_max
        self.sla_penalty_factor = sla_penalty_factor

    def evaluate_recommendations(self, df_workloads, candidate_gpus, pred_dict, true_dict, policy="risk_calibrated"):
        """
        df_workloads: DataFrame of workloads (unique model + scenario)
        candidate_gpus: List of GPU names available in cluster
        pred_dict: Dict mapping (model, scenario, gpu) -> {
            'power_pred': float, 'power_upper': float,
            'itl_pred': float, 'itl_upper': float
        }
        true_dict: Dict mapping (model, scenario, gpu) -> {
            'power_true': float, 'itl_true': float
        }
        policy: 'point_estimate', 'risk_calibrated', 'oracle', 'min_tdp'
        """
        results = []

        for _, row in df_workloads.iterrows():
            m = row["model"]
            scen = row["scenario"]

            feasible_candidates = []

            # Find Oracle best
            oracle_feasible = []
            for g in candidate_gpus:
                key = (m, scen, g)
                if key in true_dict:
                    t_val = true_dict[key]["itl_true"]
                    p_val = true_dict[key]["power_true"]
                    if t_val <= self.t_max:
                        oracle_feasible.append((g, p_val, t_val))
            
            oracle_best_g = None
            oracle_best_p = np.inf
            if oracle_feasible:
                oracle_feasible.sort(key=lambda x: x[1]) # min power
                oracle_best_g = oracle_feasible[0][0]
                oracle_best_p = oracle_feasible[0][1]

            # Evaluate selected policy
            selected_g = None
            for g in candidate_gpus:
                key = (m, scen, g)
                if key not in pred_dict or key not in true_dict:
                    continue

                p_info = pred_dict[key]
                
                if policy == "point_estimate":
                    t_check = p_info["itl_pred"]
                    p_score = p_info["power_pred"]
                elif policy == "risk_calibrated":
                    t_check = p_info["itl_upper"]
                    p_score = p_info["power_upper"]
                elif policy == "oracle":
                    t_check = true_dict[key]["itl_true"]
                    p_score = true_dict[key]["power_true"]
                elif policy == "min_tdp":
                    t_check = p_info["itl_pred"]
                    p_score = p_info["tdp"]
                else:
                    raise ValueError(f"Unknown policy {policy}")

                if t_check <= self.t_max:
                    feasible_candidates.append((g, p_score))

            if feasible_candidates:
                feasible_candidates.sort(key=lambda x: x[1]) # Pick lowest power
                selected_g = feasible_candidates[0][0]

            if selected_g is not None:
                true_p = true_dict[(m, scen, selected_g)]["power_true"]
                true_t = true_dict[(m, scen, selected_g)]["itl_true"]
                violated = true_t > self.t_max

                # Regret vs Oracle
                if oracle_best_p < np.inf:
                    regret = max(0.0, true_p - oracle_best_p)
                else:
                    regret = 0.0

                if violated:
                    regret += self.sla_penalty_factor * true_p # Penalty for violation

                results.append({
                    "model": m,
                    "scenario": scen,
                    "selected_gpu": selected_g,
                    "true_power": true_p,
                    "true_itl": true_t,
                    "violation": violated,
                    "regret": regret,
                    "abstained": False
                })
            else:
                # Abstention or deemed infeasible
                results.append({
                    "model": m,
                    "scenario": scen,
                    "selected_gpu": None,
                    "true_power": np.nan,
                    "true_itl": np.nan,
                    "violation": False,
                    "regret": 100.0 if oracle_best_g is not None else 0.0, # Regret for missing feasible option
                    "abstained": True
                })

        res_df = pd.DataFrame(results)
        valid = res_df[~res_df["abstained"]]
        
        violation_rate = valid["violation"].mean() * 100 if len(valid) > 0 else 0.0
        mean_power = valid["true_power"].mean() if len(valid) > 0 else np.nan
        mean_regret = res_df["regret"].mean()
        abstain_rate = res_df["abstained"].mean() * 100

        return {
            "policy": policy,
            "violation_rate_pct": violation_rate,
            "mean_power_watts": mean_power,
            "mean_regret": mean_regret,
            "abstain_rate_pct": abstain_rate,
            "total_workloads": len(res_df),
            "serviced_workloads": len(valid)
        }
