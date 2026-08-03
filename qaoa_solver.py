"""
combinatorial_solver/qaoa_solver.py

Conditionally-invoked quantum backend (Section 8) for the coalition-
partitioning sub-problem. This is an EXACT CLASSICAL SIMULATION of a QAOA
circuit (statevector method) -- not a call to real quantum hardware -- used
to produce manuscript Table 3. Only practical for small n_nodes (statevector
size = 2**n_nodes); a real deployment would call out to actual quantum
hardware/SDKs (Qiskit, PennyLane, Braket, D-Wave Ocean) behind this same
function signature, per the strategy-pattern design in Section 10.2.

Per manuscript Sections 3.5, 3.9, and 8.3: this module is provided for
completeness and reproducibility of Table 3, NOT because it currently
outperforms classical_solver.py. It does not.
"""
from __future__ import annotations
import numpy as np
from itertools import product
from typing import List, Tuple
from scipy.optimize import minimize
from .classical_solver import cut_value


def qaoa_maxcut(n_nodes: int, edges: List[Tuple[int, int, float]], depth: int,
                 n_restarts: int = 8, seed: int = 0):
    """Exact statevector simulation of QAOA for MaxCut. Returns
    (best_expected_cut, best_params). O(2**n_nodes) memory/time -- only
    suitable for small n_nodes, exactly as flagged in manuscript Section 8.2."""
    rng = np.random.default_rng(seed)
    basis_states = list(product([0, 1], repeat=n_nodes))
    cost_diag = np.array([cut_value(bs, edges) for bs in basis_states])

    def statevector(gammas, betas):
        dim = 2 ** n_nodes
        psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
        for layer in range(depth):
            psi = psi * np.exp(-1j * gammas[layer] * cost_diag)
            rx = np.array([[np.cos(betas[layer]), -1j * np.sin(betas[layer])],
                           [-1j * np.sin(betas[layer]), np.cos(betas[layer])]])
            U = rx
            for _ in range(n_nodes - 1):
                U = np.kron(U, rx)
            psi = U @ psi
        return psi

    def expected_cut(params):
        gammas, betas = params[:depth], params[depth:]
        psi = statevector(gammas, betas)
        probs = np.abs(psi) ** 2
        return np.sum(probs * cost_diag)

    best_val, best_params = -1.0, None
    for _ in range(n_restarts):
        x0 = rng.uniform(0, np.pi, size=2 * depth)
        res = minimize(lambda p: -expected_cut(p), x0, method="COBYLA", options={"maxiter": 300})
        val = -res.fun
        if val > best_val:
            best_val, best_params = val, res.x
    return best_val, best_params
