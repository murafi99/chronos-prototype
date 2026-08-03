"""
combinatorial_solver/classical_solver.py

Default, working classical backend for the "partition N agents into
discrete coalitions" sub-problem (Section 8.1). Provides exact brute-force
search (small N) and a hill-climbing heuristic (larger N), matching the
methodology used to produce manuscript Table 3.
"""
from __future__ import annotations
import numpy as np
from itertools import product
from typing import List, Tuple


def cut_value(bitstring, edges: List[Tuple[int, int, float]]) -> float:
    val = 0.0
    for i, j, w in edges:
        if bitstring[i] != bitstring[j]:
            val += w
    return val


def brute_force_maxcut(n_nodes: int, edges: List[Tuple[int, int, float]]):
    """Exact solution. Only tractable for small n_nodes (<= ~20)."""
    best_val, best_bs = -1.0, None
    for bits in product([0, 1], repeat=n_nodes):
        v = cut_value(bits, edges)
        if v > best_val:
            best_val, best_bs = v, bits
    return best_val, best_bs


def hill_climb_maxcut(n_nodes: int, edges: List[Tuple[int, int, float]],
                       n_restarts: int = 200, seed: int = 0) -> float:
    """Classical local-search heuristic; scales to arbitrarily large n_nodes."""
    rng = np.random.default_rng(seed)
    best = -1.0
    for _ in range(n_restarts):
        bits = list(rng.integers(0, 2, n_nodes))
        improved = True
        while improved:
            improved = False
            cur = cut_value(bits, edges)
            for k in range(n_nodes):
                bits[k] ^= 1
                v = cut_value(bits, edges)
                if v > cur:
                    cur, improved = v, True
                else:
                    bits[k] ^= 1
        best = max(best, cut_value(bits, edges))
    return best
