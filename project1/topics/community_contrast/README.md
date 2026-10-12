# Community contrast

**Question:** How does p_in/p_out affect recovery at fixed expected mean degree?
Corresponds to “Density of graph within- and between communities” (course PDF, p. 43).

**Setup:** 180 nodes, sizes 100/50/30, expected mean degree 12, known k=3.
Sweep 11 ratios from 1 to 20; 50 independent graphs per ratio; paired L/Lsym;
20 k-means initializations. Baseline r=5 was calibrated for partial recovery with Lsym.
Parameters come from `configs/baseline.json`; change `RATIOS` in `run.py` for the scan.
The overall expected degree is fixed; community-specific degrees vary with the ratio.

**Results:** Mean misclassification rate across 50 graphs:

| Ratio | L | Lsym |
|---|---:|---:|
| 3 | 42.9% | 30.7% |
| 5 | 38.8% | 8.4% |
| 8 | 32.2% | 3.1% |
| 12 | 24.8% | 1.4% |

Lsym recovers 91.6% of nodes at baseline, despite zero completely recovered graphs.
Exact recovery does not measure partial success. These are finite simulation results.

**Outputs:** [overview/](results/overview/) contains recovery and eigengap curves,
supplementary exact-recovery/connectivity plots, configurations and CSV scores.
Bands show mean ± sample SD; exact recovery uses pointwise 95% Wilson intervals.
[r5/](results/r5/) groups scatter, graph, eigenvalue/eigenvector diagnostics and source data
for repeat 0. Scatter and graph share a 2D PCA display; clustering uses all three features.
Black rings mark errors. All figures are PNG.

Run from `project1/`: `python topics/community_contrast/run.py`.
Use `--plot-only` to redraw saved results.
