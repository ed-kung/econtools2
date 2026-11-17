import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2._rng import default_rng
from econtools2.config import get_config
from econtools2.mathutils import *
from econtools2.models import *
from econtools2.graphs import *
from econtools2.documents import *
from econtools2.problems.base import GenericProblem, register


@register
class ExpectedValueProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='expected_value_problem'):
        rng = default_rng(rng)
        default_params = {'x':[1,2,3,4],'p':[0.25,0.25,0.25,0.25]}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        x, p = params['x'], params['p']
        x = np.array(x)
        p = np.array(p)
        EX = np.sum(x*p)
        self.sol['EX'] = EX
        list_x = ', '.join([fr"\(x_{i+1}={x[i]:g}\)" for i in range(len(x))])
        list_p = ', '.join([fr"\(p_{i+1}={p[i]:g}\)" for i in range(len(p))])
        setup_list = []
        setup = fr"""
A random variable \(X\) can take on the following values: {list_x}, with probabilities {list_p}.
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Calculate the expected value of \(X\)."
        answer = EX
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(E[X] = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if not equals(np.sum(self.params['p']), 1): return False
        return True


@register
class InsuranceProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='insurance_problem'):
        rng = default_rng(rng)
        default_params = {'W0':1000, 'D':200, 'p':0.2, 'fun':'ln'}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        W0, D, p, fun = params['W0'], params['D'], params['p'], params['fun']
        insurance = Insurance(W0=W0, D=D, p=p, fun=fun)
        self.sol = insurance.sol
        setup_list = []
        setup = insurance.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Calculate the expected value of \(X\)."
        answer = insurance.sol['EX']
        online_answer = fr"\(E[X] = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the expected utility."
        answer = insurance.sol['EU']
        online_answer = fr"\(E[u(X)] = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the certainty equivalent of \(X\)."
        answer = insurance.sol['CE']
        online_answer = fr"\(CE = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"How much is the person willing to pay to avoid the risk of the accident?"
        answer = insurance.sol['WTP']
        online_answer = fr"\(WTP = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"How much would it cost a risk-neutral insurance company to insure the individual against this risk?"
        answer = insurance.sol['FairCost']
        online_answer = fr"\(C = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list


@register
class PresentValueProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='present_value_problem'):
        rng = default_rng(rng)
        default_params = {'beta':0.95,'x':100,'T':20}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        beta, x, T = params['beta'], params['x'], params['T']
        if T==0:
            x = np.array(x)
            xlist = ', '.join([fr"\(x_{i+1} = {x[i]:,g}\)" for i in range(len(x))])
            question = fr"Calculate the present value of a promise to receive payments {xlist} in periods \(t=1,\ldots,{len(x):g}\), when the time discount factor is \(\beta = {beta:g}\)."
            PV = np.sum(beta**np.arange(1,len(x)+1) * x)
        elif T==np.inf:
            question = fr"Calculate the present value of a promise to receive a constant payment of {x:,g} every period from \(t=1\) to eternity, when the time discount factor is \(\beta = {beta:g}\)"
            PV = beta/(1-beta)*x
        else:
            question = fr"Calculate the present value of a promise to receive a constant payment of {x:,g} every period for \({T:g}\) periods, when the time discount factor is \(\beta = {beta:g}\)" 
            PV = ((1-beta**(T+1))/(1-beta)-1)*x
        self.sol['PV'] = PV
        setup_list = []
        setup = ""
        online_setup = ""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = question
        online_question = question
        answer = PV
        online_answer = fr"\(PV = {answer:,g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list


@register
class SavingsProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='savings_problem'):
        rng = default_rng(rng)
        default_params = {'beta':0.95, 'p':0.95, 'Y':50000}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        beta, p, Y = params['beta'], params['p'], params['Y']
        savings = Savings(Y=Y, beta=beta, p=p)
        self.sol = savings.sol.copy()
        setup_list = []
        setup = savings.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "How many bonds will the person buy in period 1?"
        answer = savings.sol['c2']
        online_answer = fr"\(b = {answer:,g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "How much will the person consume in period 1?"
        answer = savings.sol['c1']
        online_answer = fr"\(c_1 = {answer:,g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "How much will the person consume in period 2?"
        answer = savings.sol['c2']
        online_answer = fr"\(c_2 = {answer:,g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What is the interest rate on bonds?"
        answer = savings.sol['r']
        online_answer = fr"\(r = {answer*100:,g}\%\)"
        answers = generate_distractors(answer,rng=rng)
        answers = [fr"\({ans*100:g}\%\)" for ans in answers]
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['c1']<=0: return False
        if self.sol['c2']<=0: return False
        if self.params['p']>=1: return False
        if self.params['beta']>=1: return False
        return True
