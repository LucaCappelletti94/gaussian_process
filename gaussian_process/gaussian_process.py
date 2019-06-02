from typing import Callable, Dict
import os
import shutil
import json
from skopt import gp_minimize
from skopt.utils import use_named_args
from .space import Space
from dict_hash import sha256


class GaussianProcess:
    def __init__(self, score: Callable, space: Space, cache: bool = True, cache_dir: str = ".gaussian_process"):
        """Create a new gaussian process-optimized neural network wrapper
            score:Callable, function returning a score for the give parameters.
            space:Space, Space with the space to explore and the parameters to pass to the score function.
            cache:bool=True, whetever to use or not cache.
            cache_dir:str=".gaussian_process", directory where to store cache.
        """
        self._space = space
        self._score = score
        self._best_parameters = None
        self._best_optimized_parameters = None
        self._cache, self._cache_dir = cache, cache_dir

    def _params_to_cache_path(self, params: Dict):
        return "{cache_dir}/gp{hash}.json".format(
            cache_dir=self._cache_dir,
            hash=sha256(params)
        )

    @classmethod
    def _load_cached_score(cls, path: str)->float:
        with open(path, "r") as f:
            return json.load(f)["score"]

    def _store_cached_score(self, path: str, data: Dict):
        os.makedirs(self._cache_dir, exist_ok=True)
        with open(path, "w") as f:
            return json.dump(data, f, indent=4, sort_keys=True)

    def _decorate_score(self, score: Callable)->Callable:
        @use_named_args(self._space.space)
        def wrapper(**kwargs: Dict):
            params = self._space.inflate(kwargs)
            if self._cache:
                path = self._params_to_cache_path(params)
                if os.path.exists(path):
                    return self._load_cached_score(path)
            value = score(**params)
            if self._cache:
                self._store_cached_score(path, {
                    "score": value,
                    "parameters": params
                })
            return value
        return wrapper

    @property
    def best_parameters(self):
        return self._best_parameters

    @property
    def best_optimized_parameters(self):
        return self._best_optimized_parameters

    def minimize(self, random_state: int, **kwargs):
        """Minimize the function score."""
        self._space.rasterize()
        results = gp_minimize(self._decorate_score(
            self._score), self._space.space, random_state=random_state, **kwargs)
        self._best_parameters = self._space.inflate_results(results)
        self._best_optimized_parameters = self._space.inflate_results_only(
            results)
        return results

    def clear_cache(self):
        if os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir)
