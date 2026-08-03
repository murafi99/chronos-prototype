"""
Simulation 2 (toy / synthetic demonstration): full statevector QAOA simulation
for a small MaxCut instance, used here as a stand-in for the "partition N
strategic agents into two competing coalitions to maximize scenario-branch
diversity" combinatorial sub-problem described in Section 8 of the
manuscript. This is a genuine, locally-run quantum-circuit simulation (not
run on real quantum hardware, and not a claim about quantum advantage) --
its purpose is to empirically illustrate, on a problem this paper can
actually compute, the honest literature finding (Section 3.5) that QAOA at
shallow depth does not reliably beat classical heuristics on small
instances, and that classical optimization of QAOA's own parameters is hard.

Graph: 6-node random weighted graph (fixed seed), i.e. 6 qubits, dim=64.
Compares: brute-force optimum, classical local-search heuristic, and QAOA at
circuit depths p = 1, 2, 3.
"""
import numpy as np
from itertools import product
from scipy.optimize import minimize

rng = np.random.default_rng(7)
N = 6

# Random weighted graph (adjacency matrix), symmetric, sparse-ish
edges = []
W = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        if rng.random() < 0.55:
            w = round(rng.uniform(0.5, 2.0), 2)
            W[i, j] = W[j, i] = w
            edges.append((i, j, w))

def cut_value(bitstring, edges):
    val = 0.0
    for i, j, w in edges:
        if bitstring[i] != bitstring[j]:
            val += w
    return val

# --- Brute force optimum ---
best_val = -1
best_bs = None
all_vals = []
for bits in product([0, 1], repeat=N):
    v = cut_value(bits, edges)
    all_vals.append(v)
    if v > best_val:
        best_val = v
        best_bs = bits

# --- Classical local search heuristic (bit-flip hill climbing, 200 random restarts) ---
def hill_climb():
    bits = list(rng.integers(0, 2, N))
    improved = True
    while improved:
        improved = False
        cur = cut_value(bits, edges)
        for k in range(N):
            bits[k] ^= 1
            v = cut_value(bits, edges)
            if v > cur:
                cur = v
                improved = True
            else:
                bits[k] ^= 1
    return cut_value(bits, edges)

classical_best = max(hill_climb() for _ in range(200))

# --- QAOA statevector simulation ---
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_n(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

# Diagonal cost Hamiltonian values for each basis state (vectorized)
basis_states = list(product([0, 1], repeat=N))
cost_diag = np.array([cut_value(bs, edges) for bs in basis_states])
# Map bit convention: 0/1 -> +1/-1 spins for cut cost is already handled via cut_value (bit differs)

def qaoa_statevector(gammas, betas, p):
    dim = 2 ** N
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)  # |+>^N
    for layer in range(p):
        # phase separator: exp(-i*gamma*C) diagonal
        phase = np.exp(-1j * gammas[layer] * cost_diag)
        psi = psi * phase
        # mixer: exp(-i*beta*X) on each qubit = RX rotations, apply via tensor structure
        rx = np.array([[np.cos(betas[layer]), -1j * np.sin(betas[layer])],
                       [-1j * np.sin(betas[layer]), np.cos(betas[layer])]])
        mats = [rx] * N
        U = kron_n(mats)
        psi = U @ psi
    return psi

def expected_cut(params, p):
    gammas = params[:p]
    betas = params[p:]
    psi = qaoa_statevector(gammas, betas, p)
    probs = np.abs(psi) ** 2
    return np.sum(probs * cost_diag)

results = {}
for p in [1, 2, 3]:
    best_exp = -1
    best_params = None
    for trial in range(8):  # multiple random restarts, since QAOA landscape is known to be hard (Section 3.5)
        x0 = rng.uniform(0, np.pi, size=2 * p)
        res = minimize(lambda params: -expected_cut(params, p), x0, method="COBYLA",
                        options={"maxiter": 300})
        val = -res.fun
        if val > best_exp:
            best_exp = val
            best_params = res.x
    approx_ratio = best_exp / best_val
    results[p] = (best_exp, approx_ratio)

if __name__ == "__main__":
    print(f"Graph: N={N} nodes, {len(edges)} weighted edges")
    print(f"Brute-force optimum cut value: {best_val:.3f} (exhaustive search over {2**N} states)")
    print(f"Classical hill-climbing best (200 restarts): {classical_best:.3f}  "
          f"(ratio to optimum: {classical_best/best_val:.3f})")
    print()
    print("QAOA (statevector-simulated, COBYLA-optimized parameters, 8 restarts each):")
    for p, (val, ratio) in results.items():
        print(f"  p={p}:  best expected cut = {val:.3f}   approximation ratio = {ratio:.3f}")
