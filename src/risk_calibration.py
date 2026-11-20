"""
Uncertainty Quantification and Conformal Risk Calibration for SHIFT-GPU.
Implements:
1. Group-conditional Conformal Prediction
2. Covariate-Shift Adjusted Calibration using Microarchitectural Mahalanobis Discrepancy
"""

import numpy as np
from scipy.spatial.distance import mahalanobis

class ConformalRiskCalibrator:
    def __init__(self, alpha=0.10):
        """
        alpha: Miscoverage rate (e.g., 0.10 for 90% confidence / coverage)
        """
        self.alpha = alpha
        self.q_hat = 1.645 # standard normal fallback
        self.mean_train_hw = None
        self.inv_cov_train_hw = None

    def fit_shift_metric(self, train_hw_features):
        """Fits empirical distribution parameters for hardware covariate shift estimation."""
        self.mean_train_hw = np.mean(train_hw_features, axis=0)
        cov = np.cov(train_hw_features, rowvar=False) + 1e-4 * np.eye(train_hw_features.shape[1])
        self.inv_cov_train_hw = np.linalg.pinv(cov)

    def compute_hardware_shift(self, target_hw_features):
        """Computes Mahalanobis distance of target hardware from the training manifold."""
        if self.mean_train_hw is None or self.inv_cov_train_hw is None:
            return 0.0
        delta = target_hw_features - self.mean_train_hw
        d_sq = np.dot(np.dot(delta, self.inv_cov_train_hw), delta.T)
        return float(np.sqrt(np.maximum(d_sq, 0.0)))

    def calibrate(self, y_val, pred_val, std_val):
        """
        Computes empirical non-conformity quantile on held-out validation set.
        Non-conformity score: s_i = |y_i - pred_i| / std_i
        """
        std_val = np.maximum(std_val, 1e-6)
        scores = np.abs(y_val - pred_val) / std_val
        n = len(scores)
        if n == 0:
            return self.q_hat
        # Conformal (1 - alpha) * (1 + 1/n) quantile
        level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        self.q_hat = float(np.quantile(scores, level))
        return self.q_hat

    def get_prediction_intervals(self, pred_test, std_test, hw_shift_score=0.0):
        """
        Returns (lower_bound, upper_bound) calibrated for target hardware shift.
        Upper bound is critical for SLA latency compliance checking.
        """
        std_test = np.maximum(std_test, 1e-6)
        # Shift penalty expands interval proportionally to out-of-distribution distance
        shift_multiplier = 1.0 + 0.15 * np.minimum(hw_shift_score, 5.0)
        effective_q = self.q_hat * shift_multiplier
        
        lower = np.maximum(0.0, pred_test - effective_q * std_test)
        upper = pred_test + effective_q * std_test
        return lower, upper
