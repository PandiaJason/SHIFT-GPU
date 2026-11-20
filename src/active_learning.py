"""
Active Few-Shot Acquisition Strategies for SHIFT-GPU.
Implements:
1. Uniform Random Selection
2. Workload Representation K-Means Diversity Selection
3. Maximum Predictive Uncertainty Selection
4. Proposed Decision-Theoretic Regret-Optimal Active Acquisition (ROAA)
"""

import numpy as np
import torch
from sklearn.cluster import KMeans

class ActiveSelector:
    @staticmethod
    def select_random(candidate_indices, budget_k, seed=42):
        rng = np.random.RandomState(seed)
        return rng.choice(candidate_indices, size=min(budget_k, len(candidate_indices)), replace=False)

    @staticmethod
    def select_diversity(workload_embeddings, candidate_indices, budget_k, seed=42):
        """
        K-Means clustering on workload latent space to select structurally diverse workloads.
        """
        if budget_k >= len(candidate_indices):
            return candidate_indices
        embeddings = workload_embeddings[candidate_indices]
        kmeans = KMeans(n_clusters=budget_k, random_state=seed, n_init=10)
        kmeans.fit(embeddings)
        
        selected = []
        for center in kmeans.cluster_centers_:
            dists = np.linalg.norm(embeddings - center, axis=1)
            best_idx = np.argmin(dists)
            selected.append(candidate_indices[best_idx])
        return np.unique(selected)

    @staticmethod
    def select_max_uncertainty(predicted_vars, candidate_indices, budget_k):
        """
        Selects configurations with highest epistemic/aleatoric uncertainty.
        """
        vars_cand = predicted_vars[candidate_indices]
        sorted_order = np.argsort(vars_cand)[::-1]
        return candidate_indices[sorted_order[:budget_k]]

    @staticmethod
    def select_regret_optimal(pred_latencies, pred_vars, pred_powers, candidate_indices, budget_k, t_max=0.05):
        """
        Regret-Optimal Active Acquisition (ROAA).
        Prioritizes workloads located in the critical ambiguity region near the SLA boundary:
        Score = sigma(x) * exp( - (T_hat(x) - T_max)^2 / (2 * sigma^2) ) * (P_hat(x) / TDP)
        
        Avoids wasting profiling shots on:
        - Workloads that are trivially safe (T_hat << T_max)
        - Workloads that are grossly infeasible (T_hat >> T_max)
        """
        t_cand = pred_latencies[candidate_indices]
        var_cand = np.maximum(pred_vars[candidate_indices], 1e-8)
        std_cand = np.sqrt(var_cand)
        p_cand = pred_powers[candidate_indices]

        # Critical boundary proximity factor
        boundary_dist = (t_cand - t_max) / (std_cand + 1e-6)
        boundary_weight = np.exp(-0.5 * (boundary_dist ** 2))
        
        # Energy impact weighting
        power_weight = p_cand / (np.max(p_cand) + 1e-6)
        
        # Acquisition score
        acquisition_score = std_cand * (boundary_weight + 0.1) * (power_weight + 0.1)
        sorted_order = np.argsort(acquisition_score)[::-1]
        return candidate_indices[sorted_order[:budget_k]]
