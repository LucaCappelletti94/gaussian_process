from typing import Dict
import pandas as pd
from plot_keras_history import plot_history
from scipy.optimize import OptimizeResult
from ..space import Space


class History:
    def __init__(self, space: Space, maximization_problem: bool):
        self._history = []
        self._space = space
        self._maximization_problem = maximization_problem

    def __call__(self, results: OptimizeResult):
        self._history.append({
            "parameters": self._space.inflate_results(results),
            "score": -results.fun if self._maximization_problem else results.fun
        })

    def __repr__(self):
        return repr(self.to_dataframe())

    def __str__(self):
        return str(self.to_dataframe())

    def _repr_html_(self):
        return self.to_dataframe()._repr_html_()

    def plot(self, *args, **kwargs):
        plot_history(self.to_dataframe()[["score"]], *args, **kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)
