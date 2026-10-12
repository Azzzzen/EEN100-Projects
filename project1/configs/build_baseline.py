"""Edit the parameters below, then run: PYTHONPATH=src python configs/build_baseline.py."""
import json
from pathlib import Path
from spectral_project.experiment import make_config

SIZES = [100, 50, 30]
EXPECTED_DEGREE = 12
CONTRAST = 5                 # p_in / p_out
REPEATS = 50                  # Independent random graphs
SEED = 20260928
N_INIT = 20                    # k-means initializations per fit


def main():
    n = sum(SIZES)
    # Weighted expected degree, excluding self-loops; valid for unequal sizes.
    denominator = sum(s * (CONTRAST * (s - 1) + n - s) for s in SIZES) / n
    p_out = EXPECTED_DEGREE / denominator
    p_in = CONTRAST * p_out
    config = make_config('baseline', SIZES, p_in, p_out,
                         repeats=REPEATS, seed=SEED, n_init=N_INIT)
    path = Path(__file__).with_name('baseline.json')
    path.write_text(json.dumps(config, indent=2) + '\n')
    print(f'Saved {path}: p_in={p_in:.8f}, p_out={p_out:.8f}')


if __name__ == '__main__':
    main()
