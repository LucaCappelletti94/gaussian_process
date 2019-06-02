from auto_tqdm import tqdm


class TQDMGaussianProcess(object):
    def __init__(self, n_calls: int, **kwargs):
        self._bar = tqdm(total=n_calls,
                         desc="Gaussian process", **kwargs)

    def __call__(self, res):
        self._bar.update()
