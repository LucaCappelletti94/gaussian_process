import silence_tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import boston_housing
from extra_keras_utils import set_seed
from typing import Callable, Dict
import numpy as np
from holdouts_generator import holdouts_generator, random_holdouts
from gaussian_process import TQDMGaussianProcess, Space, GaussianProcess


class MLP:
    def __init__(self, holdouts:Callable):
        self._holdouts = holdouts
    
    def mlp(self, dense_layers:Dict, dropout_rate:float)->Sequential:
        return Sequential([
            *[Dense(**kwargs) for kwargs in dense_layers],
            Dropout(dropout_rate),
            Dense(1, activation="relu"),
        ])

    def model_score(self, train:np.ndarray, test:np.ndarray, structure:Dict, fit:Dict):
        model = self.mlp(**structure)
        model.compile(
            optimizer="nadam",
            loss="mse"
        )

        return model.fit(
            *train,
            epochs=1,
            validation_data=test,
            verbose=0,
            **fit
        ).history["val_loss"][-1]


    def score(self, structure:Dict, fit:Dict):
        return -np.mean([
            self.model_score(training, test, structure, fit) for (training, test), _ in self._holdouts()
        ])

def test_gaussian_process():
    set_seed(42)

    generator = holdouts_generator(
        *boston_housing.load_data()[0],
        holdouts=random_holdouts([0.1], [2])
    )

    mlp = MLP(generator)

    space = Space({
        "structure":{
            "dense_layers":[{
                "units":(8,16,32),
                "activation":("relu", "selu")
            },
            {
                "units":5,
                "activation":("relu", "selu")
            }],
            "dropout_rate":[0.0,1.0]
        },
        "fit":{
            "batch_size":[100,1000]
        }
    })

    gp = GaussianProcess(mlp.score, space)
    
    n_calls = 3
    results = gp.minimize(
        n_calls=n_calls,
        n_random_starts=1,
        callback=[TQDMGaussianProcess(n_calls=n_calls)],
        random_state=42
    )
    results = gp.minimize(
        n_calls=n_calls,
        n_random_starts=1,
        callback=[TQDMGaussianProcess(n_calls=n_calls)],
        random_state=42
    )
    print(gp.best_parameters)
    print(gp.best_optimized_parameters)
    gp.clear_cache()