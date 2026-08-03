"""
causal_engine/scm.py

Working implementation of a generic Structural Causal Model (SCM) with
do-calculus graph surgery, per Section 7.3 of the CHRONOS manuscript.

This is domain-agnostic: it takes a directed acyclic graph of node names,
a dict of structural-equation callables f_i(parent_values, noise) -> value,
and a dict of exogenous noise samplers, and supports:
  - .sample(n)                          observational Monte Carlo sampling
  - .sample(n, do={'X': value})         interventional (do-calculus) sampling

See experiments/sim1_scm_propagation.py for a full worked example (the same
synthetic 9-node graph used to produce Table 2 of the manuscript).
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Callable, Dict, Any, Optional


class SCM:
    def __init__(self):
        self.nodes: list[str] = []
        self.parents: Dict[str, list[str]] = {}
        self.equations: Dict[str, Callable[[Dict[str, Any], Any], Any]] = {}
        self.noise_samplers: Dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {}

    def add_node(self, name: str, parents: list[str],
                 equation: Callable[[Dict[str, Any], Any], Any],
                 noise_sampler: Optional[Callable[[np.random.Generator, int], np.ndarray]] = None):
        """Register a node with its parents, structural equation, and noise sampler.

        `equation(parent_values: dict, noise: np.ndarray) -> np.ndarray`
        `noise_sampler(rng, n) -> np.ndarray` defaults to standard normal.
        """
        self.nodes.append(name)
        self.parents[name] = list(parents)
        self.equations[name] = equation
        self.noise_samplers[name] = noise_sampler or (lambda rng, n: rng.normal(0, 1, n))

    def _topo_order(self, severed: set) -> list[str]:
        """Topological sort, treating nodes in `severed` as having no parents
        (this implements do-calculus graph surgery: incoming edges deleted)."""
        indegree = {n: (0 if n in severed else len(self.parents[n])) for n in self.nodes}
        adj = {n: [] for n in self.nodes}
        for n in self.nodes:
            if n not in severed:
                for p in self.parents[n]:
                    adj[p].append(n)
        queue = deque([n for n in self.nodes if indegree[n] == 0])
        order = []
        indegree_copy = dict(indegree)
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in adj[n]:
                indegree_copy[m] -= 1
                if indegree_copy[m] == 0:
                    queue.append(m)
        if len(order) != len(self.nodes):
            raise ValueError("Graph is not a DAG (cycle detected) after intervention surgery.")
        return order

    def sample(self, n_samples: int, do: Optional[Dict[str, Any]] = None,
               seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Monte Carlo sample the SCM.

        do: optional dict of {node_name: fixed_value} implementing the
            do-operator (Pearl, 2009) -- incoming edges to these nodes are
            severed and their value is fixed for every sample.
        """
        do = do or {}
        rng = np.random.default_rng(seed)
        order = self._topo_order(severed=set(do.keys()))
        values: Dict[str, np.ndarray] = {}
        for node in order:
            if node in do:
                values[node] = np.full(n_samples, do[node])
                continue
            noise = self.noise_samplers[node](rng, n_samples)
            parent_vals = {p: values[p] for p in self.parents[node]}
            values[node] = self.equations[node](parent_vals, noise)
        return values


def average_causal_effect(scm: SCM, treatment: str, outcome: str,
                           treatment_on_value, treatment_off_value,
                           n_samples: int = 20000, seed: int = 0) -> float:
    """ACE = E[Y | do(X=on)] - E[Y | do(X=off)], per Section 7.3 Eq. (ACE)."""
    on = scm.sample(n_samples, do={treatment: treatment_on_value}, seed=seed)
    off = scm.sample(n_samples, do={treatment: treatment_off_value}, seed=seed + 1)
    return float(np.mean(on[outcome]) - np.mean(off[outcome]))
