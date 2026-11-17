from econtools2._rng import default_rng
from econtools2.problems.base import PROBLEM_TYPES

# Importing the topic modules populates PROBLEM_TYPES via @register.
from econtools2.problems import (  # noqa: F401
    math_review, markets, consumer, labor, production, equilibrium, imperfect, uncertainty,
)


def list_problems():
    return sorted(PROBLEM_TYPES)


def load_problem(problem_str, params=None, name='generic_problem', rng=None):
    rng = default_rng(rng)
    return PROBLEM_TYPES[problem_str](params=params, name=name, rng=rng)


def show_menu(problem_str, params=None, name='generic_problem', rng=None):
    prob = load_problem(problem_str, params, name, rng=rng)
    setup_list = prob.setup_list
    question_list = prob.question_list
    print(prob.params)
    print(prob.sol)
    for i in range(len(setup_list)):
        print(f"{i}: {setup_list[i]['setup']}".replace('\n',' '))
        print("")
    for i in range(len(question_list)):
        print(f"{i}: {question_list[i]['question']}".replace('\n',' '))
        print("")
