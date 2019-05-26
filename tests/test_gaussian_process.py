import silence_tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import boston_housing
from extra_keras_utils import set_seed
from typing import List, Callable, Dict
import numpy as np
from holdouts_generator import holdouts_generator, random_holdouts
from gaussian_process import tqdm_gp, Space, GaussianProcess
from pprint import pprint


def mlp(dense_layers:Dict, dropout_rate:float)->Sequential:
    return Sequential([
        *[Dense(**kwargs) for kwargs in dense_layers],
        Dropout(dropout_rate),
        Dense(1, activation="relu"),
    ])

def model_score(train:np.ndarray, test:np.ndarray, structure:Dict, fit:Dict):
    model = mlp(**structure)
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


def score(holdouts:Callable, model:Dict):
    return np.mean([
        model_score(training, test, **model) for (training, test), _ in holdouts()
    ])

space = Space({
    "holdouts": holdouts_generator(
        *boston_housing.load_data()[0],
        holdouts=random_holdouts([0.1], [2])
    ),
    "model": {
        "structure":{
            "dense_layers":[{
                "units":(8,16,32),
                "activation":("relu", "selu")
            },
            {
                "units":(8,16,32),
                "activation":("relu", "selu")
            }],
            "dropout_rate":[0.0,1.0]
        },
        "fit":{
            "batch_size":[100,1000]
        }
    }
})

def test_gaussian_process():
    gp = GaussianProcess(score, space)
    n_calls = 5
    results = gp.maximize(
        n_calls=n_calls,
        n_random_starts=1,
        callback=[tqdm_gp(n_calls=n_calls)]
    )
    pprint(gp.best_parameters)
    pprint(gp.best_optimized_parameters)
    gp.clear_cache()