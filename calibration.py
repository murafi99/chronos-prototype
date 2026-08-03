"""
uncertainty/calibration.py

Working implementation of the calibration metric specified in manuscript
Section 7.3 / Table 4: Expected Calibration Error (ECE), adapted so that
`acc(S_b)` is an expert-panel agreement rate rather than ground-truth
correctness (since counterfactual ground truth is unobservable by
definition -- Section 3.7).
"""
from __future__ import annotations
import numpy as np


def expected_calibration_error(confidences: np.ndarray, expert_agreement: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    confidences: array of the system's stated confidence for each branch, in [0, 1]
    expert_agreement: array of the same length, the fraction (or boolean) of an
        expert panel that judged that branch plausible (Section 9, Table 4,
        "Expert agreement" row) -- stands in for ground-truth accuracy.
    """
    confidences = np.asarray(confidences, dtype=float)
    expert_agreement = np.asarray(expert_agreement, dtype=float)
    assert confidences.shape == expert_agreement.shape

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = expert_agreement[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def bootstrap_ci(samples: np.ndarray, statistic_fn=np.mean, n_boot: int = 2000,
                  ci: float = 0.95, seed: int = 0):
    """Generic bootstrap confidence interval, used throughout the manuscript
    (Table 2's branch-probability CIs)."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    boot_stats = np.array([
        statistic_fn(samples[rng.integers(0, n, n)]) for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_stats, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)
