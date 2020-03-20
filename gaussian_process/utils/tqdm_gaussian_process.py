from tqdm.auto import tqdm


class TQDMGaussianProcess:
    def __init__(self, total: int):
        """Create the TQDM bar for the Gaussian Process."""
        self._bar = tqdm(
            total=total,
            desc="Gaussian process",
            dynamic_ncols=True
        )

    def __call__(self, res):
        self._bar.update()
