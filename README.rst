gaussian_process
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy| |code_climate_maintainability| |pip| |downloads|

Wrapper for `"sklearn.gp_minimize"` for a simpler parameter specification using nested dictionaries.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install gaussian_process

Tests Coverage
----------------------------------------------
Since some software handling coverages sometime get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|

Keras model optimization using a gaussian process
-------------------------------------------------------------

.. code:: python

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

    if __name__ == "__main__":
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
                    "units":[8,16,32],
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

.. |travis| image:: https://travis-ci.org/LucaCappelletti94/gaussian_process.png
   :target: https://travis-ci.org/LucaCappelletti94/gaussian_process
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_gaussian_process&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_gaussian_process
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_gaussian_process&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_gaussian_process
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_gaussian_process&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_gaussian_process
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/gaussian_process/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/gaussian_process?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/gaussian_process.svg
    :target: https://badge.fury.io/py/gaussian_process
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/gaussian_process
    :target: https://pepy.tech/badge/gaussian_process
    :alt: Pypi total project downloads 

.. |codacy|  image:: https://api.codacy.com/project/badge/Grade/0a674ed703f44793a27936462ca05080
    :target: https://www.codacy.com/app/LucaCappelletti94/gaussian_process?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LucaCappelletti94/gaussian_process&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/aabe32e918c9ba7cd773/maintainability
    :target: https://codeclimate.com/github/LucaCappelletti94/gaussian_process/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/aabe32e918c9ba7cd773/test_coverage
    :target: https://codeclimate.com/github/LucaCappelletti94/gaussian_process/test_coverage
    :alt: Code Climate Coverate