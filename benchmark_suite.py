"""
evaluation/benchmark_suite.py

INTERFACE SCAFFOLD (Section 10.1) -- not a working implementation of a
full benchmark run (that requires a populated KG, real multi-agent
simulation, and a real expert panel -- Section 12.2, Limitation 1).

What IS implemented and working here: the metric functions themselves,
which experiments/ already exercises on toy data (contradiction rate via
causal_engine.propagate, ECE via uncertainty.calibration). This module
just defines the harness shape a full evaluation run would use to combine
them into the Table 4 report format.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EvaluationReport:
    contradiction_rate_pre: float
    contradiction_rate_post: float
    ece: float
    fleiss_kappa: float | None       # requires a real expert panel; None until then
    reproducibility_variance: float
    path_trace_completeness: float
    domain_realism_scores: Dict[str, float]  # per Table 4's last row


def run_benchmark_suite(*args, **kwargs) -> EvaluationReport:
    raise NotImplementedError(
        "Full-suite evaluation requires a populated knowledge graph, a running "
        "multi-agent simulation, and a real expert-historian panel -- none of "
        "which are in scope for this paper's prototype (see manuscript Section "
        "12.2, Limitation 1, and Section 12.3, Future Work item (ii))."
    )
