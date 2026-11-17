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
class GeneralEquilibriumProblem(GenericProblem):
    # u(L) = wL - d*L^kw
    # f(L) = A*L^kf
    # u(q) = a*ln(q) - p*q
    def __init__(self, params=None, rng=None, name='general_equilibrium_problem'):
        rng = default_rng(rng)
        default_params = {'A':2,'kf':1/2,'d':1/2,'kw':3/2,'a':12}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        A, kf, d, kw, a = params['A'], params['kf'], params['d'], params['kw'], params['a']
        consumer = LogConsumer(a)
        firm = ExponentialProductionFirm(A,kf)
        worker = Worker(d,kw)
        equilibrium = GeneralEquilibrium(consumer, firm, worker)
        self.sol = equilibrium.eq.copy()
        L = self.sol['L']
        w = self.sol['w']
        p = self.sol['p']
        q = self.sol['q']
        U_consumer = self.sol['U_consumer']
        U_worker = self.sol['U_worker']
        profit = self.sol['profit']
        setup_list = []
        setup = fr"""
{consumer.setup()}

{worker.setup()}

{firm.setup()}
"""
        online_setup = fr"""
<p>{consumer.setup()}</p>
        
<p>{worker.setup()}</p>

<p>{firm.setup()}</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Calculate the equilibrium quantity of labor supplied, \(L\)."
        online_question = question
        answer = L
        online_answer = fr"\(L = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the equilibrium quantity of commodity produced, \(q\)."
        online_question = question
        answer = q
        online_answer = fr"\(q = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the equilibrium wage rate \(w\)."
        online_question = question
        answer = w
        online_answer = fr"\(w = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the equilibrium commodity price \(p\)."
        online_question = question
        answer = p
        online_answer = fr"\(p = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the equilibrium utility of the consumer."
        online_question = question
        answer = U_consumer
        online_answer = fr"\(U_c = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the equilibrium utility of the worker."
        online_question = question
        answer = U_worker
        online_answer = fr"\(U_w = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the equilibrium profit of the firm."
        online_question = question
        answer = profit
        online_answer = fr"\(\Pi = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['p']<=1: return False
        if self.sol['p']>50: return False
        if self.sol['q']<=1: return False
        if self.sol['q']>50: return False
        if self.sol['w']<=1: return False
        if self.sol['w']>50: return False
        if self.sol['L']<=1: return False
        if self.sol['L']>50: return False
        if self.sol['U_consumer']<0: return False
        return True


@register
class ProductivityShockProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='productivity_shock_problem'):
        rng = default_rng(rng)
        default_params = {'A1':2,'A2':1,'kf':1/2,'d':1/2,'kw':3/2,'a':12}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        A1, A2, kf, d, kw, a = params['A1'], params['A2'], params['kf'], params['d'], params['kw'], params['a']
        assert A1!=A2
        consumer = LogConsumer(a)
        firm1 = ExponentialProductionFirm(A1,kf)
        firm2 = ExponentialProductionFirm(A2,kf)
        worker = Worker(d,kw)
        equilibrium1 = GeneralEquilibrium(consumer, firm1, worker)
        equilibrium2 = GeneralEquilibrium(consumer, firm2, worker)
        self.eq1 = equilibrium1
        self.eq2 = equilibrium2
        dU_consumer = equilibrium2.eq['U_consumer'] - equilibrium1.eq['U_consumer']
        dU_worker = equilibrium2.eq['U_worker'] - equilibrium1.eq['U_worker']
        dprofit = equilibrium2.eq['profit'] - equilibrium1.eq['profit']
        assert equals(dprofit, 0)
        assert equals(dU_worker, 0)
        dprofit=0
        dU_worker=0
        self.sol = {'dU_consumer':dU_consumer, 'dU_worker':dU_worker, 'dprofit':dprofit}
        firm_setup = fr"""
A representative, price-taking firm uses labor to produce and sell a commodity at unit price \(p\). The firm hires labor at a constant wage rate \(w\). If the firm employs \(L\) units of labor, it can produce \(f(L)\) units of commodity output, where:

$$ f(L) = A L^{{{asfrac(kf,inline=True)}}} $$
"""
        setup_list = []
        setup = fr"""
{consumer.setup()}

{worker.setup()}

{firm_setup}

Suppose the labor productivity, \(A\), changes from {A1:g} to {A2:g}.
"""
        online_setup = fr"""
<p>{consumer.setup()}</p>
        
<p>{worker.setup()}</p>

<p>{firm_setup}</p>

<p>Suppose the total factor productivity, \(A\), changes from {A1:g} to {A2:g}.</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"How much does the utility of the consumer change?"
        online_question = question
        answer = dU_consumer
        online_answer = fr"\(\Delta U_c = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"How much does the utility of the worker change?"
        online_question = question
        answer = dU_worker
        online_answer = fr"\(\Delta U_w = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"How much does the profit of the firm change?"
        online_question = question
        answer = dprofit
        online_answer = fr"\(\Delta \Pi = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
        question = fr"Who is the greatest beneficiary of the growth in labor productivity?"
        online_question = question
        answer = "consumers"
        online_answer = answer
        answers = ['workers', 'firms', 'consumers']
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,2,shuffle=False,sort=False,horz=True,numerical=False,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.eq1.eq['U_consumer']<0: return False
        if self.eq2.eq['U_consumer']<0: return False
        if np.abs(self.sol['dU_consumer'])<0.1: return False
        return True
