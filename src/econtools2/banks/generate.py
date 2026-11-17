import pandas as pd

from econtools2._rng import default_rng
from econtools2.problems.registry import load_problem


def build_bank(problem_type, param_iter, rng=None):
    """Build a parameter bank: keep every params dict whose problem passes check_solution().

    Replaces the loop pattern in generate_problems.ipynb:
        data.append(problem.params | problem.sol)
    """
    rng = default_rng(rng)
    rows = []
    for params in param_iter:
        problem = load_problem(problem_type, params=params, rng=rng)
        if problem.check_solution():
            rows.append(problem.params | problem.sol)
    return pd.DataFrame(rows)
