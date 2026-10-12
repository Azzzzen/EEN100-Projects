# Project 2 — Lasso Simulations

Model: `y = X beta + epsilon`, with independent original noise `epsilon_i ~ N(0,1)`. Predictor columns are centered and scaled to mean square one; the response is centered. The software penalty is `alpha` in `RSS/(2n) + alpha ||b||_1`; the paper convention is `lambda = 2n alpha`.

## Experiment settings

### Low-dimensional

- `p=3`, `q=2`, `beta=(1,1,0)`.
- Gaussian predictors with `rho12=0.2`, `rho13=rho23=r`; cases use `r=0.3` and `r=0.7`.
- Sample sizes: 50, 100, 200, 400, 600, 800; 50 repetitions per setting.
- Predetermined penalty: `alpha_n=0.75 n^(-0.15)`; also check the complete Lasso path.
- Paired-noise illustration: `n=50`, comparing `epsilon` and `-epsilon` for the same design and penalty.

### High-dimensional

The first `q` coefficients are +1; all others are zero. Active predictors are independent. In correlated cases, one inactive proxy has the stated correlation with each active predictor; all remaining inactive predictors are independent.

| Case | p | q | Population correlation structure | IC | eta |
|---|---|---|---|---:|---:|
| A — Independent | 4n | 5 | Independent predictors | 0 | 1 |
| B — Correlated | 4n | 5 | One inactive proxy; correlation 0.18 | 0.9 | 0.1 |
| C — Difficult | 8n | 10 | One inactive proxy; correlation 0.09 | 0.9 | 0.1 |

- Sample sizes: 50, 80, 120, 180, 270, 400.
- Per setting: 80 tuning datasets and 160 independent evaluation datasets.
- Grid: 49 values of `alpha=k sqrt(2 log(p)/n)`, with k from 0.15 to 8.
- Selection: minimum mean sign-error count; ties use coefficient squared error, then smaller alpha.
- Population margin: `eta=1-IC`; sample IC is computed for each dataset.

## Run

Use Python 3.12 and run from `project2/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python low_dimensional/run.py
.venv/bin/python low_dimensional/visualize.py
.venv/bin/python low_dimensional/explain_finite_samples.py
.venv/bin/python low_dimensional/validate.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python high_dimensional/run.py
.venv/bin/python high_dimensional/visualize.py
.venv/bin/python high_dimensional/validate.py
```
