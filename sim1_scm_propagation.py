"""
Simulation 1 (toy / synthetic demonstration): Monte Carlo propagation of a
discrete intervention through a small structural causal model (SCM) built
over a SYNTHETIC, abstracted historical event graph. This is a mechanism
demonstration for the CHRONOS counterfactual-propagation algorithm
(Section 7.4 of the manuscript) -- it is NOT a claim about any real
historical episode. Node names are deliberately generic/abstract categories
(political, military, economic, diplomatic) rather than named real events,
to avoid misrepresenting a toy illustration as validated historiography.

Method:
 - 9-node causal DAG, binary/ternary variables.
 - Each non-root node's conditional distribution is a logistic function of
   its parents' states plus independent exogenous noise (standard SCM
   formulation, Pearl 2009).
 - We compare P(Outcome | do(Trigger=1)) against the observational baseline
   P(Outcome) via 20,000-sample Monte Carlo simulation with bootstrap 95% CIs.
 - A simple consistency checker flags "contradictory" samples (e.g. terminal
   state reached before an intermediate node has fired), analogous to the
   consistency constraints CHRONOS is meant to enforce.
"""
import numpy as np

rng = np.random.default_rng(42)
N_SAMPLES = 20000
N_BOOT = 2000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate(intervene_trigger=None, n=N_SAMPLES):
    # Exogenous noise
    U = rng.normal(0, 1, size=(n, 9))

    Trigger = (rng.random(n) < 0.5).astype(int) if intervene_trigger is None else np.full(n, intervene_trigger)

    # Node 1: Political_Alliance_Shift <- Trigger
    p1 = sigmoid(2.0 * Trigger - 0.5 + 0.3 * U[:, 0])
    Alliance = (rng.random(n) < p1).astype(int)

    # Node 2: Military_Mobilization <- Trigger, Alliance
    p2 = sigmoid(1.5 * Trigger + 1.2 * Alliance - 0.8 + 0.3 * U[:, 1])
    Mobilization = (rng.random(n) < p2).astype(int)

    # Node 3: Economic_Shock <- Mobilization
    p3 = sigmoid(1.8 * Mobilization - 0.6 + 0.4 * U[:, 2])
    EconShock = (rng.random(n) < p3).astype(int)

    # Node 4: Public_Opinion_Shift <- EconShock, Alliance  (0=stable,1=hostile,2=pacifist)
    score = 1.3 * EconShock - 0.9 * Alliance + 0.5 * U[:, 3]
    PublicOpinion = np.where(score > 0.6, 1, np.where(score < -0.6, 2, 0))

    # Node 5: Diplomatic_Intervention <- Alliance, Mobilization (inverse relation)
    p5 = sigmoid(-1.4 * Mobilization + 0.5 * Alliance - 0.2 + 0.4 * U[:, 4])
    Diplomacy = (rng.random(n) < p5).astype(int)

    # Node 6: Third_Party_Entry <- Mobilization, Diplomacy
    p6 = sigmoid(1.1 * Mobilization - 1.6 * Diplomacy - 0.3 + 0.4 * U[:, 5])
    ThirdParty = (rng.random(n) < p6).astype(int)

    # Node 7: Escalation_Pressure <- Mobilization, EconShock, ThirdParty
    p7 = sigmoid(1.0 * Mobilization + 0.8 * EconShock + 1.3 * ThirdParty - 1.0 + 0.4 * U[:, 6])
    Escalation = (rng.random(n) < p7).astype(int)

    # Node 8: Negotiated_Settlement_Pressure <- Diplomacy, PublicOpinion==2
    p8 = sigmoid(1.7 * Diplomacy + 1.1 * (PublicOpinion == 2).astype(int) - 0.7 + 0.4 * U[:, 7])
    Settlement = (rng.random(n) < p8).astype(int)

    # Terminal node: Outcome in {Escalated_Conflict, Negotiated_Settlement, Frozen_Stalemate}
    outcome = np.full(n, 2)  # default: Frozen_Stalemate
    outcome = np.where((Escalation == 1) & (Settlement == 0), 0, outcome)  # Escalated_Conflict
    outcome = np.where((Settlement == 1) & (Escalation == 0), 1, outcome)  # Negotiated_Settlement
    # contradictory samples: both fired strongly -> flagged, resolved by causal priority (escalation dominates)
    contradiction = (Escalation == 1) & (Settlement == 1)
    outcome = np.where(contradiction, 0, outcome)

    return {
        "Trigger": Trigger, "Alliance": Alliance, "Mobilization": Mobilization,
        "EconShock": EconShock, "PublicOpinion": PublicOpinion, "Diplomacy": Diplomacy,
        "ThirdParty": ThirdParty, "Escalation": Escalation, "Settlement": Settlement,
        "Outcome": outcome, "Contradiction": contradiction.astype(int)
    }

def branch_probs_with_ci(outcome_array, n_boot=N_BOOT):
    labels = ["Escalated_Conflict", "Negotiated_Settlement", "Frozen_Stalemate"]
    n = len(outcome_array)
    point = [np.mean(outcome_array == i) for i in range(3)]
    boot = np.zeros((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = outcome_array[idx]
        boot[b] = [np.mean(sample == i) for i in range(3)]
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return labels, point, lo, hi

if __name__ == "__main__":
    baseline = simulate(intervene_trigger=None)
    do1 = simulate(intervene_trigger=1)
    do0 = simulate(intervene_trigger=0)

    print("=== P(Outcome) -- observational baseline (Trigger unforced) ===")
    labels, point, lo, hi = branch_probs_with_ci(baseline["Outcome"])
    for l, p, a, b in zip(labels, point, lo, hi):
        print(f"  {l:24s}  {p:.3f}  95% CI [{a:.3f}, {b:.3f}]")
    print(f"  Contradiction rate (pre-resolution): {baseline['Contradiction'].mean():.3f}")

    print("\n=== P(Outcome | do(Trigger=1)) -- counterfactual intervention ===")
    labels, point1, lo1, hi1 = branch_probs_with_ci(do1["Outcome"])
    for l, p, a, b in zip(labels, point1, lo1, hi1):
        print(f"  {l:24s}  {p:.3f}  95% CI [{a:.3f}, {b:.3f}]")
    print(f"  Contradiction rate (pre-resolution): {do1['Contradiction'].mean():.3f}")

    print("\n=== P(Outcome | do(Trigger=0)) -- counterfactual suppression ===")
    labels, point0, lo0, hi0 = branch_probs_with_ci(do0["Outcome"])
    for l, p, a, b in zip(labels, point0, lo0, hi0):
        print(f"  {l:24s}  {p:.3f}  95% CI [{a:.3f}, {b:.3f}]")
    print(f"  Contradiction rate (pre-resolution): {do0['Contradiction'].mean():.3f}")

    print("\n=== Average Causal Effect of do(Trigger=1) vs do(Trigger=0) on P(Escalated_Conflict) ===")
    ace = point1[0] - point0[0]
    print(f"  ACE = {ace:.3f}")
