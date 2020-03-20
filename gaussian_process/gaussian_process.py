from typing import Callable, Dict
import os
import shutil
import json
from skopt import gp_minimize
from skopt.utils import use_named_args
from .space import Space
from dict_hash import sha256
from skopt.callbacks import DeltaYStopper
from .utils import TQDMGaussianProcess, History


class GaussianProcess:
    def __init__(
        self,
        score: Callable,
        space: Space
    ):
        """Create a new gaussian process-optimized neural network wrapper

        Parameters
        -----------------------------
        score: Callable,
            Function representing a loss to minimize or a score to maximize.
        space: Space,
            Space with the space to explore and the parameters to pass to the score function.
        """
        self._space = space
        self._score = score
        self._maximization_problem = False

    def _decorate_score(self, score: Callable) -> Callable:
        @use_named_args(self._space.space)
        def wrapper(**kwargs: Dict):
            new_score = score(**self._space.inflate(kwargs))
            if self._maximization_problem:
                return -new_score
            return new_score
        return wrapper

    @property
    def best_parameters(self):
        if self._best_parameters is None:
            raise ValueError("You have not run the Gaussian Process yet!")
        return self._best_parameters

    def _fit(
        self,
        n_calls: int = 100,
        n_random_starts: int = 10,
        random_state: int = 42,
        early_stopping_delta: float = 0.001,
        early_stopping_best_models: int = 5,
        n_jobs: int = -1
    ):
        """Minimize the function score."""
        self._space.rasterize()
        history = History(self._space, self._maximization_problem)
        results = gp_minimize(
            func=self._decorate_score(self._score),
            dimensions=self._space.space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            n_jobs=n_jobs,
            callback=[
                TQDMGaussianProcess(n_calls),
                history,
                DeltaYStopper(
                    early_stopping_delta,
                    n_best=early_stopping_best_models
                )
            ],
            random_state=random_state
        )
        self._best_parameters = self._space.inflate_results(results)
        return history

    def minimize(self, *args, **kwargs):
        self._maximization_problem = False
        return self._fit(*args, **kwargs)

    def maximize(self, *args, **kwargs):
        self._maximization_problem = True
        return self._fit(*args, **kwargs)
