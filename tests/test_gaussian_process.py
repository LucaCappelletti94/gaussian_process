from typing import List
import numpy as np
from gaussian_process import TQDMGaussianProcess, Space, GaussianProcess
from multiprocessing import cpu_count
from skopt.callbacks import DeltaYStopper


def to_maximize(x: List, y: List):
    return np.sum(np.cos(x)) + np.sum(np.sin(y))

def to_minimize(x: List, y: List):
    return np.sin(x+y).sum()


def test_gaussian_process():
    space = Space({
        "x": [[-3.0, 3.0] for _ in range(2)],
        "y": [[-3.0, 3.0] for _ in range(2)]
    })

    GaussianProcess(to_maximize, space).maximize().plot(path="history_maximize.png")
    GaussianProcess(to_minimize, space).minimize().plot(path="history_minimize.png")