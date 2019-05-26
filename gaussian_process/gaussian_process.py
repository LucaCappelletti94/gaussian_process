from typing import List, Callable, Dict
import os
import shutil
import pickle
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.callbacks import DeltaYStopper
from .space import Space

class GaussianProcess:
    def __init__(self, score:Callable, space:Space, cache:bool=True, cache_dir:str=".gaussian_process"):
        """Create a new gaussian process-optimized neural network wrapper
            score:Callable, function returning a score for the give parameters.
            space:Space, Space with the space to explore and the parameters to pass to the score function.
            cache:bool=True, whetever to use or not cache.
            cache_dir:str=".gaussian_process", directory where to store cache.
        """
        self._space = space
        self._score = self._decorate_score(score)
        self._best_parameters = None
        self._best_optimized_parameters = None
        self._cache, self._cache_dir = cache, cache_dir

    def _params_to_cache_path(self, params:Dict):
        return "{cache_dir}/{hash}.pickle".format(
            cache_dir=self._cache_dir,
            hash=hash(str(params))
        )

    def _load_cached_score(self, path:str)->float:
        with open(path, "rb") as f:
            return pickle.load(f)["score"]

    def _store_cached_score(self, path:str, data:Dict):
        os.makedirs(self._cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            return pickle.dump(data, f)

    def _decorate_score(self, score:Callable)->Callable:
        @use_named_args(self._space)
        def wrapper(**kwargs:Dict):
            params = self._space.inflate(kwargs)
            if self._cache:
                path = self._params_to_cache_path(params)
                if os.path.exists(path):
                    return self._load_cached_score(path)
            value = score(**params)
            if self._cache:
                self._store_cached_score(path, {"score":value})
            return value
        return wrapper

    def _negate(self, func:Callable)->Callable:
        def wrapper(*args, **kwargs):
            return -func(*args, **kwargs)
        return wrapper

    @property
    def best_parameters(self):
        return self._best_parameters

    @property
    def best_optimized_parameters(self):
        return self._best_optimized_parameters

    def minimize(self, **kwargs):
        """Minimize the function score."""
        results = gp_minimize(self._score, self._space, **kwargs)
        self._best_parameters = self._space.inflate_results(results)
        self._best_optimized_parameters = self._space.inflate_results_only(results)
        return results

    def maximize(self, **kwargs):
        """Minimize the maximize score."""
        self._score = self._negate(self._score)
        return self.minimize(**kwargs)

    def clear_cache(self):
        if os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir)