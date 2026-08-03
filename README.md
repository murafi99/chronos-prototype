# Supplementary scripts (as submitted with the manuscript)

These two files are the original, self-contained scripts referenced in the
manuscript's supplementary files and in Section 10.1 (Table 5):

- `sim1_scm_propagation.py` → produced **Table 2**. Its logic was refactored
  into the reusable package at [`causal_engine/`](../causal_engine/) (see
  [`experiments/reproduce_table2_scm.py`](../experiments/reproduce_table2_scm.py)
  for the package-based equivalent, which gives the same results).
- `sim2_qaoa_maxcut.py` → produced **Table 3**. Its logic was refactored into
  [`combinatorial_solver/`](../combinatorial_solver/) (see
  [`experiments/reproduce_table3_qaoa.py`](../experiments/reproduce_table3_qaoa.py)
  for the package-based equivalent).

Both are single-file, dependency-light (`numpy`, `scipy`) and can be run
directly:

```bash
python supplementary/sim1_scm_propagation.py
python supplementary/sim2_qaoa_maxcut.py
```

They're kept here unmodified for provenance/citation purposes — if you just
want to reproduce the paper's tables, use `experiments/` instead, which
exercises the actual package code.
