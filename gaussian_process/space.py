from typing import Dict, Tuple
from skopt.space import Categorical, Real, Integer
from deflate_dict import inflate, deflate

class Space(list):
    def __init__(self, space:Dict, sep="____"):
        super().__init__()
        self._sep = sep
        self._space = deflate(space, sep=self._sep)
        self._fixed = {}
        self._names = []
        for name, value in self._space.items():
            self._parse(name, value)

    def _parse_categorical(self, value:Tuple, name:str):
        if len(self._to_tuple(value))>1:
            self.append(Categorical(self._to_tuple(value), name=name))
            self._names.append(name)
        else:
            self._fixed[name] = value

    def _parse_real(self, low:float, high:float, name:str):
        self.append(Real(low=low, high=high, name=name))
        self._names.append(name)

    def _parse_integer(self, low:int, high:int, name:str):
        self.append(Integer(low=low, high=high, name=name))
        self._names.append(name)

    def _is_categorical(self, value)->bool:
        return not (isinstance(value, list) and len(value)==2 and all([
            isinstance(v, (float, int)) for v in value
        ]))

    def _is_real(self, values)->bool:
        return all([
            isinstance(v, float) for v in values
        ])

    def _to_tuple(self, value)->bool:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value, )

    def _parse(self, name:str, value):
        if self._is_categorical(value):
            self._parse_categorical(value, name=name)
        elif self._is_real(value):
            self._parse_real(*value, name=name)
        else:
            self._parse_integer(*value, name=name)

    def inflate(self, deflated_space:Dict)->Dict:
        return inflate({**deflated_space, **self._fixed}, sep=self._sep)

    def inflate_results(self, results:"OptimizeResult")->Dict:
        return self.inflate(dict(zip(self._names, results.x)))

    def inflate_results_only(self, results:"OptimizeResult")->Dict:
        return inflate(dict(zip(self._names, results.x)), sep=self._sep)