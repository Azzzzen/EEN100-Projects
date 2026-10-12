# Project 1: Spectral Clustering

Python experiments comparing L and Lsym on stochastic block models (SBM).

## Setup and run

Run from `project1/` with Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m unittest discover -s tests -q
python -m spectral_project --config configs/baseline.json --output results/baseline
python topics/community_contrast/run.py
```

## Parameters and files

- Edit [configs/build_baseline.py](configs/build_baseline.py), then run `python configs/build_baseline.py` to generate the shared config.
- Baseline: sizes **100/50/30**, expected mean degree **12**, p_in/p_out **5**, **50** graphs, **20** k-means initializations.
- [src/spectral_project/](src/spectral_project/): graph generation, clustering, metrics and plotting.
- [topics/community_contrast/](topics/community_contrast/README.md): density investigation; edit `RATIOS` in its `run.py` for the sweep.
- [results/reference/](results/reference/): checked baseline results and calibration record.
- Copy `topics/_template/` to start another topic.

Both methods use the same graphs and known k. Compare eigenvalues, gaps, eigenvectors
and misclassification rates; exact recovery is supplementary. At baseline, mean error
is **38.79% for L** and **8.37% for Lsym**.
