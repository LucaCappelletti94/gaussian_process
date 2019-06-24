from typing import List
import numpy as np
from gaussian_process import TQDMGaussianProcess, Space, GaussianProcess
from multiprocessing import cpu_count


def score(x: List, y: List):
    return -(np.sum(np.cos(x)) + np.sum(np.sin(y)))


def test_categorical_gp():
    space = Space({
        "x": (-1.0, -0.7, -0.3, 0, 0.3, 0.7, 1.0),
        "y": (0, 0.3, 0.7, 1.3, 1.5, 1.57, 1.6, 1.7, 2.3, 2.7, 3.0)
    })

    gp = GaussianProcess(score, space)

    n_calls = 50
    results = gp.minimize(
        n_calls=n_calls,
        n_random_starts=5,
        callback=[TQDMGaussianProcess(n_calls=n_calls)],
        random_state=42,
        n_jobs=cpu_count()
    )
    assert 2 + results.fun < 0.0001
    gp.clear_cache()
