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
from econtools2.problems.consumer import *
from econtools2.problems.production import *


@register
class LinearMarketProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='linear_market_problem'):
        rng = default_rng(rng)
        default_params = {'ad':12,'bd':1,'as':0,'bs':1,'xunit':1,'yunit':1,'xn':13,'yn':13}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        demand = LinearDemand(a=params['ad'], b=params['bd'])
        supply = LinearSupply(a=params['as'], b=params['bs'])
        market = LinearMarket(demand, supply)
        axis = Axis(xn=params['xn'],yn=params['yn'],xunit=params['xunit'],yunit=params['yunit'],xlab=r'$q$',ylab=r'$p$')
        demand.line.color = 'blue'
        supply.line.color = 'red'
        axis.add(demand.line, supply.line)
        setup_list = []
        setup = fr"""
Supply and demand in a market are defined by the following equations:
\begin{{align*}}
q_d &= {demand.print()} \\
q_s &= {supply.print()} 
\end{{align*}}
"""
        online_setup = fr"""
Supply and demand in a market are defined by the following equations:
$$\begin{{align}}
q_d &= {demand.print()} \\
q_s &= {supply.print()}
\end{{align}}$$
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "Calculate the equilibrium price."
        ans = market.eq['p']
        answers = generate_distractors(ans,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": ans,
            "online_answer": fr"\(p = {ans:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the equilibrium quantity."
        ans = market.eq['q']
        answers = generate_distractors(ans,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": ans,
            "online_answer": fr"\(q = {ans:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"""
Draw the supply and demand diagram using the provided grid:
\begin{{center}}
\includegraphics[width=3in]{{{name}_blank.png}}
\end{{center}}
"""
        online_question = f'Draw the supply and demand diagram using the provided grid:<p><img src="{get_config().image_url_prefix}/{name}_blank.png"></p>'
        answer = fr"""
\begin{{center}}
\includegraphics[width=3in]{{{name}.png}}
\end{{center}}
"""
        online_answer = f'<img src="{get_config().image_url_prefix}/{name}_sol.png">'
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": None
        })
        question = "Calculate the consumer surplus."
        ans = market.eq['CS']
        answers = generate_distractors(ans,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": ans,
            "online_answer": fr"\(CS = {ans:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the producer surplus."
        ans = market.eq['PS']
        answers = generate_distractors(ans,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": ans,
            "online_answer": fr"\(PS = {ans:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.demand = demand
        self.supply = supply
        self.market = market
        self.sol = market.eq.copy()
        self.axis = axis
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        ok = True
        ok = ok and self.axis.on_grid(self.market.eq['p'], self.market.eq['q'])
        ok = ok and self.axis.on_grid(0, self.demand.a)
        ok = ok and self.axis.on_grid(0, self.supply.a)
        return ok


@register
class ExponentialMarketProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='exponential_market_problem'):
        rng = default_rng(rng)
        default_params = {'ad':6,'kd':-0.5,'as':1,'ks':2}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        demand = ExponentialDemand(a=params['ad'], k=params['kd'])
        supply = ExponentialSupply(a=params['as'], k=params['ks'])
        market = ExponentialMarket(demand, supply)
        setup_list = []
        setup = fr"""
Supply and demand in a market are defined by the following equations:
\begin{{align*}}
q_d &= {demand.print()} \\
q_s &= {supply.print()} 
\end{{align*}}
"""
        online_setup = fr"""
Supply and demand in a market are defined by the following equations:
$$\begin{{align}}
q_d &= {demand.print()} \\
q_s &= {supply.print()}
\end{{align}}$$
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        setup = fr"""
A market's inverse demand curve and inverse supply curve are defined by the following equations:
\begin{{align*}}
p &= {demand.print_inverse(x='q_d')} \\
p &= {supply.print_inverse(x='q_s')} 
\end{{align*}}
"""
        online_setup = fr"""
A market's inverse demand curve and inverse supply curve are defined by the following equations:
$$\begin{{align}}
p &= {demand.print_inverse(x='q_d')} \\
p &= {supply.print_inverse(x='q_s')}
\end{{align}}$$
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "Calculate the equilibrium price."
        ans = market.eq['p']
        answers = generate_distractors(ans,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": ans,
            "online_answer": fr"\(p = {ans:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = "Calculate the equilibrium quantity."
        ans = market.eq['q']
        answers = generate_distractors(ans,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": ans,
            "online_answer": fr"\(q = {ans:g}\)",
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.demand = demand
        self.supply = supply
        self.market = market
        self.sol = market.eq.copy()
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if not is_rational((1/self.demand.a)**(1/self.demand.k)): return False
        if not is_rational((1/self.supply.a)**(1/self.supply.k)): return False
        return True


@register
class LinearCommodityMarketProblem(GenericProblem):
    # u = ac*q - 0.5*bcq^2 - pq
    # pi = p*q - af*q - 0.5*bf*q^2
    def __init__(self, params=None, rng=None, name='linear_commodity_market_problem'):
        rng = default_rng(rng)
        default_params = {'ac':12,'bc':1,'af':0,'bf':1}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        ac, bc, af, bf = params['ac'], params['bc'], params['af'], params['bf']
        consumer = LinearConsumer(ac, bc)
        producer = QuadraticCostFirm(af, bf)
        market = LinearMarket(consumer.demand, producer.supply)
        q = market.eq['q']
        p = market.eq['p']
        U = consumer.utility_at(p, q)
        profit = producer.profit_at(p, q)
        self.sol = {'q':q, 'p':p, 'U':U, 'profit':profit}
        linear_consumer_problem = LinearConsumerProblem(params={'a':ac,'b':bc,'p':p}, rng=rng)
        linear_producer_problem = QuadraticCostFirmProblem(params={'a':af,'b':bf,'p':p}, rng=rng)
        setup_list = []
        setup = fr"""
{consumer.setup()}

{producer.setup()}
"""
        online_setup = fr"""
<p>{consumer.setup()}</p>

<p>{producer.setup()}</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question_list.append(linear_consumer_problem.question_list[1])
        question_list.append(linear_producer_problem.question_list[2])
        question = "Calculate the equilibrium price."
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
        question = "Calculate the equilibrium quantity."
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
        question = "Calculate the consumer utility in equilibrium."
        online_question = question
        answer = U
        online_answer = fr"\(U = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = "Calculate the firm profit in equilibrium."
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
        self.consumer = consumer
        self.producer = producer
        self.market = market
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['q']<=0: return False
        if self.sol['p']<=0: return False
        return True


@register
class ExponentialCommodityMarketProblem(GenericProblem):
    # u = a*ln(q) - pq
    # pi = p*q - 0.5*bf*q^2
    def __init__(self, params=None, rng=None, name='linear_commodity_market_problem'):
        rng = default_rng(rng)
        default_params = {'a':12,'b':1}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b = self.params['a'], self.params['b']
        consumer = LogConsumer(a)
        producer = QuadraticCostFirm(0, b)
        producer.supply = ExponentialSupply(b, 1)
        market = ExponentialMarket(consumer.demand, producer.supply)
        q = market.eq['q']
        p = market.eq['p']
        U = consumer.utility_at(p, q)
        profit = producer.profit_at(p, q)
        self.sol = {'q':q, 'p':p, 'U':U, 'profit':profit}
        consumer_problem = LogConsumerProblem(params={'a':a,'p':p}, rng=rng)
        producer_problem = QuadraticCostFirmProblem(params={'a':0,'b':b,'p':p}, rng=rng)
        setup_list = []
        setup = fr"""
{consumer.setup()}

{producer.setup()}
"""
        online_setup = fr"""
<p>{consumer.setup()}</p>

<p>{producer.setup()}</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question_list.append(consumer_problem.question_list[1])
        question_list.append(producer_problem.question_list[2])
        question = "Calculate the equilibrium price."
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
        question = "Calculate the equilibrium quantity."
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
        question = "Calculate the consumer utility in equilibrium."
        online_question = question
        answer = U
        online_answer = fr"\(U = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = "Calculate the firm profit in equilibrium."
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
        self.consumer = consumer
        self.producer = producer
        self.market = market
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['q']<=0: return False
        if self.sol['p']<=0: return False
        if self.sol['U']<0: return False
        return True
