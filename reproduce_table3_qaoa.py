"""
experiments/reproduce_table3_qaoa.py

Reproduces manuscript Table 3 (toy QAOA vs. classical MaxCut benchmark)
using the actual combinatorial_solver package modules. Run with:
    python -m experiments.reproduce_table3_qaoa
"""
import numpy as np
from combinatorial_solver.classical_solver import brute_force_maxcut, hill_climb_maxcut
from combinatorial_solver.qaoa_solver import qaoa_maxcut


def build_graph(seed=7, n=6, p_edge=0.55):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                w = round(rng.uniform(0.5, 2.0), 2)
                edges.append((i, j, w))
    return n, edges


def main():
    n, edges = build_graph()
    print(f"Graph: N={n} nodes, {len(edges)} weighted edges")

    best_val, _ = brute_force_maxcut(n, edges)
    print(f"Brute-force optimum: {best_val:.3f}")

    classical_best = hill_climb_maxcut(n, edges, n_restarts=200)
    print(f"Classical hill-climbing (200 restarts): {classical_best:.3f} "
          f"(ratio {classical_best/best_val:.3f})")

    for depth in [1, 2, 3]:
        val, _ = qaoa_maxcut(n, edges, depth=depth, n_restarts=8)
        print(f"QAOA p={depth}: best expected cut = {val:.3f}  "
              f"approximation ratio = {val/best_val:.3f}")


if __name__ == "__main__":
    main()
