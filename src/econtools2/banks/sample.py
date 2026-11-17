from pathlib import Path

import pandas as pd

from econtools2._rng import default_rng
from econtools2.problems.registry import load_problem


def _read_bank(path):
    path = Path(path)
    return pd.read_pickle(path) if path.suffix == ".pkl" else pd.read_csv(path)


def sample_params(bank_path, rng=None):
    rng = default_rng(rng)
    return dict(_read_bank(bank_path).sample(1, random_state=rng).iloc[0])


def sample_problem(problem_type, bank_path, rng=None, name='generic_problem'):
    """Draw one row from a parameter bank and build the problem.

    Equivalent to the notebook pattern:
        df = pd.read_csv(f"{TYPE}.csv")
        params = dict(df.sample(1, random_state=rng).iloc[0])
        problem = u2.load_problem(TYPE, params, rng=rng)
    """
    rng = default_rng(rng)
    params = sample_params(bank_path, rng=rng)
    return load_problem(problem_type, params, name=name, rng=rng)
