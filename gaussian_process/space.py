from typing import Dict, Tuple, List
from skopt.space import Categorical, Real, Integer
from deflate_dict import inflate, deflate
from collections import OrderedDict
import numpy as np


class Space(OrderedDict):
    def __init__(self, space: Dict, sep="____"):
        super().__init__(space)
        self._sep = sep
        self._space = self._fixed = self._names = None

    def _parse_categorical(self, value: Tuple, name: str):
        if len(self._to_tuple(value)) > 1:
            self._space.append(Categorical(self._to_tuple(value), name=name))
            self._names.append(name)
        else:
            self._fixed[name] = value

    def _parse_real(self, low: float, high: float, name: str):
        self._space.append(Real(low=low, high=high, name=name))
        self._names.append(name)

    def _parse_integer(self, low: int, high: int, name: str):
        self._space.append(Integer(low=low, high=high, name=name))
        self._names.append(name)

    @classmethod
    def _is_categorical(cls, value)->bool:
        return not (isinstance(value, list) and len(value) == 2 and all([
            isinstance(v, (float, int)) for v in value
        ]))

    @classmethod
    def _is_real(cls, values)->bool:
        return all([isinstance(v, float) for v in values])

    @classmethod
    def _to_tuple(cls, value)->bool:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value, )

    def _parse(self, name: str, value):
        if self._is_categorical(value):
            self._parse_categorical(value, name=name)
        elif self._is_real(value):
            self._parse_real(*value, name=name)
        else:
            self._parse_integer(*value, name=name)

    def rasterize(self):
        self._names = []
        self._fixed = {}
        self._space = []
        for name, value in deflate(self, sep=self._sep).items():
            self._parse(name, value)

    @property
    def space(self)->List:
        return self._space

    @classmethod
    def _sanitize_array(cls, array):
        return [
            np.asscalar(a) if isinstance(a, np.generic) else a for a in array
        ]

    @classmethod
    def _sanitize_dictionary(cls, dictionary):
        return dict([
            (k, np.asscalar(v)) if isinstance(v, np.generic) else (k, v) for k, v in dictionary.items()
        ])

    def inflate(self, deflated_space: Dict)->Dict:
        return inflate(self._sanitize_dictionary({**deflated_space, **self._fixed}), sep=self._sep)

    def inflate_results(self, results: "OptimizeResult")->Dict:
        return self.inflate(dict(zip(self._names, self._sanitize_array(results.x))))

    def inflate_results_only(self, results: "OptimizeResult")->Dict:
        return inflate(dict(zip(self._names, self._sanitize_array(results.x))), sep=self._sep)
