from econtools2._rng import default_rng

PROBLEM_TYPES = {}


def register(cls):
    """Class decorator: make a Problem available to load_problem() by its class name."""
    PROBLEM_TYPES[cls.__name__] = cls
    return cls


class GenericProblem:
    def __init__(self, params, default_params, rng=None, name="generic_problem"):
        self.name = name
        self.sol = {}
        if not params:
            params = default_params.copy()
        self.params = {}
        for k in default_params.keys():
            if k in params.keys():
                self.params[k] = params[k]
            else:
                self.params[k] = default_params[k]
    def show_setups(self):
        i=0
        for s in self.setup_list:
            print(f'{i}: ', end='')
            print(s)
            i+=1
    def show_questions(self):
        i=0
        for q in self.question_list:
            print(f'{i}: ', end='')
            print(q)
            i+=1
    def check_solution(self):
        return True
