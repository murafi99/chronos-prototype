"""
causal_engine/propagate.py

Working implementation of ALGORITHM: CHRONOS-Propagate (manuscript Section 7.4),
built on top of scm.py. Adds:
  - a pluggable consistency-rule engine (contradiction detection + resolution)
  - bootstrap-CI branch-probability reporting (feeds evaluation/calibration.py)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
from .scm import SCM


@dataclass
class ConsistencyRule:
    name: str
    # violated(values: Dict[str, np.ndarray]) -> boolean mask over samples
    violated: Callable[[Dict[str, np.ndarray]], np.ndarray]
    # resolve(values, mask) -> mutates `values` in place to fix violated samples
    resolve: Callable[[Dict[str, np.ndarray], np.ndarray], None]


@dataclass
class PropagationResult:
    values: Dict[str, np.ndarray]
    contradiction_log: Dict[str, float] = field(default_factory=dict)  # rule_name -> pre-resolution violation rate


def chronos_propagate(scm: SCM, do: Dict[str, Any], n_samples: int,
                       rules: Optional[List[ConsistencyRule]] = None,
                       seed: Optional[int] = None) -> PropagationResult:
    """Implements ALGORITHM: CHRONOS-Propagate exactly as specified in
    manuscript Section 7.4, steps 1-12."""
    rules = rules or []
    values = scm.sample(n_samples, do=do, seed=seed)  # steps 1-7 (graph surgery + Monte Carlo)

    contradiction_log = {}
    for rule in rules:  # steps 8-9
        mask = rule.violated(values)
        rate = float(np.mean(mask))
        contradiction_log[rule.name] = rate
        if rate > 0:
            rule.resolve(values, mask)

    return PropagationResult(values=values, contradiction_log=contradiction_log)  # steps 10-12


def branch_distribution(outcome_array: np.ndarray, n_categories: int,
                         n_boot: int = 2000, seed: int = 0):
    """Empirical distribution + 95% bootstrap CIs over a categorical outcome
    array, used to populate manuscript Table 2-style output."""
    rng = np.random.default_rng(seed)
    n = len(outcome_array)
    point = np.array([np.mean(outcome_array == i) for i in range(n_categories)])
    boot = np.zeros((n_boot, n_categories))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = outcome_array[idx]
        boot[b] = [np.mean(sample == i) for i in range(n_categories)]
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return point, lo, hi
