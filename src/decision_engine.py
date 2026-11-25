"""
Decision Engine for Risk-Constrained Hardware Recommendation.
Evaluates deployment decisions under SLA constraints:
min_g P_hat(x, g)  s.t.  T_hat_upper(x, g) <= T_max
and scores operational regret according to Eq. (2):
R(g* | x) = max(0, P(x, g*) - P(x, g_oracle)) + lambda_sla * P(x, g*) * I(T(x, g*) > T_max)
with fixed abstention penalty when feasible candidates exist under oracle knowledge.
"""

import numpy as np
import pandas as pd

# Table mapping policy keys to (feasibility latency field, ranking cost field)
_POLICY_SELECTORS = {
    "min_tdp_naive": (None, "tdp"),
    "min_tdp_roofline": ("itl_roofline", "tdp"),
    "xgb_point": ("itl_xgb_point", "power_pred_xgb"),
    "xgb_roof_point": ("itl_xgb_roof_point", "power_pred_xgb"),
    "pican_point": ("itl_pican_point", "power_pred_pican"),
    "shift_gpu_calibrated": ("itl_pican_upper", "power_pred_pican"),
    "risk_calibrated": ("itl_pican_upper", "power_pred_pican"),
    "point_estimate": ("itl_pican_point", "power_pred_pican"),
}


class DecisionEngine:
    def __init__(self, t_max=0.035, sla_penalty_factor=5.0, abstention_penalty=50.0):
        """
        t_max: Latency Service Level Agreement (seconds per token)
        sla_penalty_factor: Asymmetric multiplier lambda_SLA penalizing SLA violations
        abstention_penalty: Operational penalty when abstaining despite oracle feasibility
        """
        self.t_max = t_max
        self.sla_penalty_factor = sla_penalty_factor
        self.abstention_penalty = abstention_penalty

    def evaluate_recommendations(self, df_workloads, candidate_gpus, pred_dict, true_dict, policy="shift_gpu_calibrated"):
        """
        Evaluates hardware selection policies across identical workload sets.
        """
        if policy != "oracle" and policy not in _POLICY_SELECTORS:
            raise ValueError(f"Unknown policy {policy}")

        results = []
        for _, row in df_workloads.iterrows():
            m = row["model"]
            scen = row["scenario"]

            oracle_feasible = [
                (g, true_dict[(m, scen, g)]["power_true"])
                for g in candidate_gpus
                if (m, scen, g) in true_dict and true_dict[(m, scen, g)]["itl_true"] <= self.t_max
            ]
            oracle_best_g = None
            oracle_best_p = np.inf
            if oracle_feasible:
                oracle_feasible.sort(key=lambda x: x[1])
                oracle_best_g, oracle_best_p = oracle_feasible[0]

            feasible_candidates = []
            for g in candidate_gpus:
                key = (m, scen, g)
                if key not in pred_dict or key not in true_dict:
                    continue

                if policy == "oracle":
                    meets_sla = true_dict[key]["itl_true"] <= self.t_max
                    score = true_dict[key]["power_true"]
                else:
                    lat_key, pwr_key = _POLICY_SELECTORS[policy]
                    p_info = pred_dict[key]
                    meets_sla = True if lat_key is None else (p_info[lat_key] <= self.t_max)
                    score = p_info[pwr_key]

                if meets_sla:
                    feasible_candidates.append((g, score))

            selected_g = None
            if feasible_candidates:
                feasible_candidates.sort(key=lambda x: x[1])
                selected_g = feasible_candidates[0][0]

            if selected_g is not None:
                true_p = true_dict[(m, scen, selected_g)]["power_true"]
                true_t = true_dict[(m, scen, selected_g)]["itl_true"]
                violated = true_t > self.t_max
                regret = 0.0 if np.isinf(oracle_best_p) else max(0.0, true_p - oracle_best_p)
                if violated:
                    regret += self.sla_penalty_factor * true_p

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
                results.append({
                    "model": m,
                    "scenario": scen,
                    "selected_gpu": None,
                    "true_power": np.nan,
                    "true_itl": np.nan,
                    "violation": False,
                    "regret": self.abstention_penalty if oracle_best_g is not None else 0.0,
                    "abstained": True
                })

        res_df = pd.DataFrame(results)
        valid = res_df[~res_df["abstained"]]
        total = len(res_df)
        accepted = len(valid)
        abstain_count = total - accepted
        violation_count = int(valid["violation"].sum()) if accepted > 0 else 0

        return {
            "policy": policy,
            "target_sla_ms": int(round(self.t_max * 1000)),
            "total_workloads": total,
            "accepted_workloads": accepted,
            "abstain_count": abstain_count,
            "abstain_rate_pct": (abstain_count / total * 100.0) if total > 0 else 0.0,
            "violation_count": violation_count,
            "violation_rate_pct": (violation_count / accepted * 100.0) if accepted > 0 else 0.0,
            "violation_fraction_str": f"{violation_count} / {accepted}",
            "mean_power_watts": float(valid["true_power"].mean()) if accepted > 0 else np.nan,
            "mean_regret": float(res_df["regret"].mean()),
            "serviced_workloads": accepted,
        }
