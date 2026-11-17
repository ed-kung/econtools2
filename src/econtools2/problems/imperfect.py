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
class MonopolyProblem(GenericProblem):
    # p = alpha - beta*q
    # c(q) = f + a*q + b*q^2
    def __init__(self, params=None, rng=None, name='quadratic_optimization_problem'):
        rng = default_rng(rng)
        default_params = {'alpha':12,'beta':1,'f':0,'a':0,'b':0.5}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        alpha, beta, f, a, b = params['alpha'], params['beta'], params['f'], params['a'], params['b']
        consumer = LinearConsumer(alpha, beta)
        monopoly = Monopoly(consumer.demand, f, a, b)
        self.consumer = consumer
        self.monopoly = monopoly
        self.sol = monopoly.sol.copy()

        # --- Setups
        setup_list = []
        setup = fr"""
Price-taking consumers in the market for a commodity have a demand curve given by:
$$ q = {consumer.demand.print()} $$
{monopoly.setup()}
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        setup = fr"""
{consumer.setup()}
{monopoly.setup()}
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })

        # --- Questions
        question_list = []
        question = "Calculate the profit maximizing price."
        answer = monopoly.sol['p']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(p = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the profit maximizing quantity."
        answer = monopoly.sol['q']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(q = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the monopolist's profit."
        answer = monopoly.sol['profit']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(\Pi = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the consumer utility."
        answer = monopoly.sol['CS']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(U = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What quantity would a benevolent social planner choose, in order to maximize total surplus?"
        answer = monopoly.sol['q_eff']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(q_e = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What price would a benevolent social planner choose? (Assume the planner chooses price quantity pairs on the demand curve.)"
        answer = monopoly.sol['p_eff']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(p_e = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the monopolist's profit under the benevolent social planner's choices."
        answer = monopoly.sol['profit_eff']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(\Pi_e = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the consumer utility under the benevolent social planner's choices."
        answer = monopoly.sol['CS_eff']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(U_e = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the deadweight loss caused by monopolistic behavior."
        answer = monopoly.sol['DWL']
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(DWL = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['q']<=0: return False
        if self.sol['p']<=0: return False
        if self.sol['q_eff']<=0: return False
        if self.sol['p_eff']<=0: return False
        if self.sol['profit']<=0: return False
        return True


@register
class PriceDiscriminationProblem(GenericProblem):
    # pA = alphaA - betaA*qA
    # pB = alphaB - betaB*qB
    # c(q) = c*q
    def __init__(self, params=None, rng=None, name="price_discrimination_problem"):
        rng = default_rng(rng)
        default_params = {'alphaA':12,'betaA':1,'alphaB':10,'betaB':1,'c':1}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        alphaA, betaA, alphaB, betaB, c = params['alphaA'], params['betaA'], params['alphaB'], params['betaB'], params['c']
        demandA = LinearDemand(a=alphaA, b=betaA)
        demandB = LinearDemand(a=alphaB, b=betaB)
        pdmodel = PriceDiscrimination(demandA, demandB, c)
        self.pdmodel = pdmodel
        qA = pdmodel.monopoly1.sol['q']
        pA = pdmodel.monopoly1.sol['p']
        profitA = pdmodel.monopoly1.sol['profit']
        qB = pdmodel.monopoly2.sol['q']
        pB = pdmodel.monopoly2.sol['p']
        profitB = pdmodel.monopoly2.sol['profit']
        profit_pd = profitA + profitB
        q_nopd = pdmodel.monopoly_both.sol['q']
        p_nopd = pdmodel.monopoly_both.sol['p']
        qA_nopd = demandA.eval_at_p(p_nopd)
        qB_nopd = demandB.eval_at_p(p_nopd)
        profit_nopd = pdmodel.monopoly_both.sol['profit']
        self.sol = {
            'qA': qA,
            'pA': pA,
            'profitA': profitA,
            'qB': qB,
            'pB': pB,
            'profitB': profitB,
            'profit_pd': profit_pd,
            'q_nopd': q_nopd,
            'p_nopd': p_nopd,
            'qA_nopd': qA_nopd,
            'qB_nopd': qB_nopd,
            'profit_nopd': profit_nopd
        }
        setup_list = []
        setup = fr"""
There are two types of consumers in the market, type A and type B. The demand curve for type A consumers is given by:
$$ q_A = {demandA.print()} $$
The demand curve for type B consumers is given by:
$$ q_B = {demandB.print()} $$
{pdmodel.setup()}
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "Calculate the profit maximizing price for type A consumers under third degree price discrimination."
        answer = pA
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(p_A = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the profit maximizing price for type B consumers under third degree price discrimination."
        answer = pB
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(p_B = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the monopolist's total profit under third degree price discrimination."
        answer = profit_pd
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"Profit (with price discrimination) = \({answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the price that the monopolist would charge if it were not able to price discriminate."
        answer = p_nopd
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"\(p = {answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the monopolist's profit if it were not able to price discriminate."
        answer = profit_nopd
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": fr"Profit (without price discrimination) = \({answer:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['pA']<=0: return False
        if self.sol['pB']<=0: return False
        if self.sol['p_nopd']<=0: return False
        if self.sol['qA']<=0: return False
        if self.sol['qB']<=0: return False
        if self.sol['qA_nopd']<=0: return False
        if self.sol['qB_nopd']<=0: return False
        if self.sol['profitA']<=0: return False
        if self.sol['profitB']<=0: return False
        if self.sol['profit_pd']<=0: return False
        if self.sol['profit_nopd']<=0: return False
        if self.sol['profit_pd']<self.sol['profit_nopd']: return False
        return True


@register
class Cournot2Problem(GenericProblem):
    def __init__(self, params=None, rng=None, name='cournot2_problem'):
        rng = default_rng(rng)
        default_params = {'alpha':12,'beta':1,'a1':0,'b1':1,'a2':0,'b2':1}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        alpha, beta, a1, b1, a2, b2 = params['alpha'], params['beta'], params['a1'], params['b1'], params['a2'], params['b2']
        firm1 = QuadraticCostFirm(a=a1, b=b1)
        firm2 = QuadraticCostFirm(a=a2, b=b2)
        demand = LinearDemand(a=alpha, b=beta)
        cournot2 = Cournot2(firm1, firm2, demand)
        self.sol = cournot2.sol.copy()
        self.cournot2 = cournot2
        setup_list = []
        setup = cournot2.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "What quantity does firm 1 produce in the Nash equilibrium?"
        online_question = question
        answer = cournot2.sol['q1']
        online_answer = fr"\(q_1 = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What quantity does firm 2 produce in the Nash equilibrium?"
        online_question = question
        answer = cournot2.sol['q2']
        online_answer = fr"\(q_2 = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What is the Nash equilibrium price in this market?"
        online_question = question
        answer = cournot2.sol['p']
        online_answer = fr"\(p = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What is the Nash equilibrium profit of firm 1?"
        online_question = question
        answer = cournot2.sol['profit1']
        online_answer = fr"\(\Pi_1 = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What is the Nash equilibrium profit of firm 2?"
        online_question = question
        answer = cournot2.sol['profit2']
        online_answer = fr"\(\Pi_2 = {answer:g}\)"
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
    def check_solution(self):
        if not is_divisible(self.sol['p'],1): return False
        if not is_divisible(self.sol['q1'],1): return False
        if not is_divisible(self.sol['q2'],1): return False
        if self.sol['p']<=0: return False
        if self.sol['q1']<=0: return False
        if self.sol['q2']<=0: return False
        return True


@register
class CournotNProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='cournotN_problem'):
        rng = default_rng(rng)
        default_params = {'alpha':12,'beta':1,'mc':3,'N':2}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        alpha, beta, mc, N = params['alpha'], params['beta'], params['mc'], params['N']
        demand = LinearDemand(a=alpha, b=beta)
        cournotN = CournotN(mc, N, demand)
        self.sol = cournotN.sol.copy()
        self.cournotN = cournotN
        setup_list = []
        setup = cournotN.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "What quantity does each firm produce in the Nash equilibrium?"
        online_question = question
        answer = cournotN.sol['q']
        online_answer = fr"\(q_i = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What is the total quantity produced in the Nash equilibrium?"
        online_question = question
        answer = cournotN.sol['Q']
        online_answer = fr"\(Q = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What is the Nash equilibrium price in this market?"
        online_question = question
        answer = cournotN.sol['p']
        online_answer = fr"\(p = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "What is the Nash equilibrium profit of each firm?"
        online_question = question
        answer = cournotN.sol['profit']
        online_answer = fr"\(\Pi_i = {answer:g}\)"
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
    def check_solution(self):
        return True


@register
class NormalFormProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='normal_form_problem'):
        rng = default_rng(rng)
        default_params = {
            'players': ['Player 1', 'Player 2'],
            'strategies': [['A','B'],['A','B']],
            'payoffs': [[[4,4],[10,0]],
                        [[0,10],[6,6]]],
            'gametype': "Prisoner's Dilemma"
        }
        xn = 13
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        players, strategies, payoffs, gametype = params['players'], params['strategies'], params['payoffs'], params['gametype']
        normalform = NormalForm(players=players, strategies=strategies, payoffs=payoffs, gametype=gametype)
        ne = normalform.ne
        ne_distractors = normalform.ne_distractors
        if len(ne)==0:
            ne_sol = fr"No Nash Equilibria"
            _distractors_ = rng.choice(ne_distractors, 2, replace=False)
            ne_d1 = fr"1 Nash Equilibria: {_distractors_[0]}"
            ne_d2 = fr"1 Nash Equilibria: {_distractors_[1]}"
            ne_d3 = fr"2 Nash Equilibria: {', '.join(rng.choice(ne_distractors,2,replace=False))}"
        elif len(ne)==1:
            ne_sol = fr"{len(ne)} Nash Equilibria: {ne[0]}"
            ne_d1 = fr"No Nash Equilibria"
            ne_d2 = fr"1 Nash Equilibria: {rng.choice(ne_distractors)}"
            ne_d3 = fr"2 Nash Equilibria: {', '.join(rng.choice(ne_distractors,2,replace=False))}"
        else:
            ne_sol = fr"{len(ne)} Nash Equilibria: {', '.join(ne)}"
            ne_d1 = fr"No Nash Equilibria"
            ne_d2 = fr"1 Nash Equilibria: {rng.choice(ne_distractors)}"
            ne_d3 = fr"2 Nash Equilibria: {', '.join(rng.choice(ne_distractors,2,replace=False))}"
        self.normalform = normalform
        setup_list = []
        setup = fr"""
Consider the 2-player game described by the normal form below:
{normalform.table_as_latex()}
"""
        online_setup = fr"""
<p>Consider the 2-player game described by the normal form below:</p>
{normalform.table_as_html()}
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "Find all the Nash equilibria."
        online_question = question
        answer = ne_sol
        online_answer = answer
        answers = [ne_sol, ne_d1, ne_d2, ne_d3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=False,shuffle=True,rng=rng)
        })
        question = "Do the strategic dynamics of this game most resemble Prisoner's Dilemma, Stag Hunt, Chicken, or Rock Paper Scissors?"
        online_question = question
        answer = gametype
        online_answer = answer
        answers = ["Prisoner's Dilemma", "Stag Hunt", "Chicken", "Rock Paper Scissors"]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,shuffle=False,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
