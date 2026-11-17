import numpy as np


def default_rng(rng=None):
    """Return `rng` if given, else a fresh unseeded Generator."""
    return rng if rng is not None else np.random.default_rng()
