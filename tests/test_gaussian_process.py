from typing import List
import numpy as np
from gaussian_process import TQDMGaussianProcess, Space, GaussianProcess
from multiprocessing import cpu_count
from skopt.callbacks import DeltaYStopper


def score(x: List, y: List):
    return -(np.sum(np.cos(x)) + np.sum(np.sin(y)))


def test_gaussian_process():
    space = Space({
        "x": [[-1.0, 1.0] for _ in range(2)],
        "y": [[0.0, 3.0] for _ in range(2)]
    })

    gp = GaussianProcess(score, space)

    n_calls = 50
    results = gp.minimize(
        n_calls=n_calls,
        n_random_starts=5,
        callback=[
            TQDMGaussianProcess(n_calls=n_calls),
            DeltaYStopper(0.001)
        ],
        random_state=42,
        n_jobs=cpu_count()
    )
    assert 4 + results.fun < 0.0001
    gp.clear_cache()
