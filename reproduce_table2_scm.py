"""
experiments/reproduce_table2_scm.py

Reproduces manuscript Table 2 (toy SCM propagation) using the actual
causal_engine package modules (scm.py + propagate.py), rather than the
standalone sim1_scm_propagation.py script included with the paper's
supplementary files. Run with: python -m experiments.reproduce_table2_scm
"""
import numpy as np
from causal_engine.scm import SCM
from causal_engine.propagate import chronos_propagate, ConsistencyRule, branch_distribution


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build_synthetic_graph() -> SCM:
    scm = SCM()
    scm.add_node("Trigger", [], lambda pv, u: (u > 0).astype(int),
                 noise_sampler=lambda rng, n: rng.random(n) - 0.5)
    scm.add_node("Alliance", ["Trigger"],
                 lambda pv, u: (u < sigmoid(2.0 * pv["Trigger"] - 0.5)).astype(int),
                 noise_sampler=lambda rng, n: rng.random(n))
    scm.add_node("Mobilization", ["Trigger", "Alliance"],
                 lambda pv, u: (u < sigmoid(1.5 * pv["Trigger"] + 1.2 * pv["Alliance"] - 0.8)).astype(int),
                 noise_sampler=lambda rng, n: rng.random(n))
    scm.add_node("EconShock", ["Mobilization"],
                 lambda pv, u: (u < sigmoid(1.8 * pv["Mobilization"] - 0.6)).astype(int),
                 noise_sampler=lambda rng, n: rng.random(n))
    scm.add_node("Diplomacy", ["Alliance", "Mobilization"],
                 lambda pv, u: (u < sigmoid(-1.4 * pv["Mobilization"] + 0.5 * pv["Alliance"] - 0.2)).astype(int),
                 noise_sampler=lambda rng, n: rng.random(n))
    scm.add_node("ThirdParty", ["Mobilization", "Diplomacy"],
                 lambda pv, u: (u < sigmoid(1.1 * pv["Mobilization"] - 1.6 * pv["Diplomacy"] - 0.3)).astype(int),
                 noise_sampler=lambda rng, n: rng.random(n))
    scm.add_node("Escalation", ["Mobilization", "EconShock", "ThirdParty"],
                 lambda pv, u: (u < sigmoid(1.0 * pv["Mobilization"] + 0.8 * pv["EconShock"]
                                             + 1.3 * pv["ThirdParty"] - 1.0)).astype(int),
                 noise_sampler=lambda rng, n: rng.random(n))
    scm.add_node("Settlement", ["Diplomacy"],
                 lambda pv, u: (u < sigmoid(1.7 * pv["Diplomacy"] - 0.7)).astype(int),
                 noise_sampler=lambda rng, n: rng.random(n))

    def outcome_eq(pv, u):
        outcome = np.full(len(pv["Escalation"]), 2)  # Frozen_Stalemate
        outcome = np.where((pv["Escalation"] == 1) & (pv["Settlement"] == 0), 0, outcome)
        outcome = np.where((pv["Settlement"] == 1) & (pv["Escalation"] == 0), 1, outcome)
        outcome = np.where((pv["Escalation"] == 1) & (pv["Settlement"] == 1), 0, outcome)  # priority rule
        return outcome
    scm.add_node("Outcome", ["Escalation", "Settlement"], outcome_eq)
    return scm


def contradiction_rule() -> ConsistencyRule:
    def violated(values):
        return (values["Escalation"] == 1) & (values["Settlement"] == 1)

    def resolve(values, mask):
        values["Outcome"] = np.where(mask, 0, values["Outcome"])  # causal-priority override

    return ConsistencyRule(name="escalation_settlement_exclusivity", violated=violated, resolve=resolve)


def main():
    scm = build_synthetic_graph()
    rules = [contradiction_rule()]

    for label, do in [("Observational", {}), ("do(Trigger=1)", {"Trigger": 1}), ("do(Trigger=0)", {"Trigger": 0})]:
        result = chronos_propagate(scm, do=do, n_samples=20000, rules=rules, seed=42)
        point, lo, hi = branch_distribution(result.values["Outcome"], n_categories=3)
        labels = ["Escalated_Conflict", "Negotiated_Settlement", "Frozen_Stalemate"]
        print(f"=== {label} ===")
        for l, p, a, b in zip(labels, point, lo, hi):
            print(f"  {l:24s} {p:.3f}  95% CI [{a:.3f}, {b:.3f}]")
        print(f"  contradiction rate (pre-resolution): {result.contradiction_log}")
        print()


if __name__ == "__main__":
    main()
