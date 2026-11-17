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
class QuadraticCostFirmProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='quadratic_cost_firm_problem'):
        rng = default_rng(rng)
        default_params = {'a':0,'b':1,'p':6}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b, p = params['a'], params['b'], params['p']
        producer = QuadraticCostFirm(a,b)
        q = producer.supply.eval_at_p(p)
        profit = producer.profit_at(p,q)
        self.sol = {'q':q, 'profit':profit}

        obj =             fr"\(\Pi = {PolyEq([1,-a,-0.5*b],['pq','q','q'],[1,1,2])} \)"
        obj_distractor1 = fr"\(\Pi = {PolyEq([1,-a,0.5*b],['pq','q','q'],[1,1,2])} \)"
        obj_distractor2 = fr"\(\Pi = {PolyEq([1,-a,0.5*b],['p','q','q'],[1,1,2])} \)"
        obj_distractor3 = fr"\(\Pi = {PolyEq([1,-a,-0.5*b],['p','q','q'],[1,1,2])} \)"
        
        foc =             fr"\({PolyEq([1,-a,-b],['p','','q'],[1,0,1])} = 0 \)"
        foc_distractor1 = fr"\({PolyEq([1,-b],['p',''],[1,0])} = 0\)"
        foc_distractor2 = fr"\({PolyEq([1,-a,b],['p','','q'],[1,0,1])} = 0\)"
        foc_distractor3 = fr"\({PolyEq([1,-a,-b],['pq','','q'],[1,0,1])} = 0\)"

        supply_curve =             fr"\(q_s = {PolyEq([1/b,-a/b],'p',[1,0])}\)"
        supply_curve_distractor1 = fr"\(q_s = {PolyEq([a/b,-1/b],'p',[0,1])}\)"
        supply_curve_distractor2 = fr"\(q_s = {PolyEq([1/b,-a/b],'p',[-1,0])}\)"
        supply_curve_distractor3 = rng.choice([
            fr"\(q_s = {PolyEq([2/b,-2*a/b],'p',[1,0])}\)",
            fr"\(q_s = {PolyEq([0.5/b,-0.5*a/b],'p',[1,0])}\)"
        ])
        
        setup_list = []
        setup = producer.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "Write down the firm's profit function."
        online_question = question
        answer = obj
        online_answer = answer
        answers = [obj, obj_distractor1, obj_distractor2, obj_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = "Write down the firm's first order condition."
        online_question = question
        answer = foc
        online_answer = answer
        answers = [foc, foc_distractor1, foc_distractor2, foc_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = "Write down the firm's supply curve."
        online_question = question
        answer = supply_curve
        online_answer = answer
        answers = [supply_curve, supply_curve_distractor1, supply_curve_distractor2, supply_curve_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Calculate the choice of \(q\) that maximizes profit when price is \(p={p:g}\)."
        online_question = question
        answer = q
        online_answer = fr"\(q = {answer}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the maximum profit attained when price is \(p={p:g}\)."
        online_question = question
        answer = profit
        online_answer = fr"\(U = {answer}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        self.producer = producer
        self.setup_list = setup_list
        self.question_list = question_list


@register
class ExponentialProductionFirmProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='exponential_production_firm_problem'):
        rng = default_rng(rng)
        default_params = {'A':1,'k':1/2,'w':1,'p':1}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        A, k, w, p = params['A'], params['k'], params['w'], params['p']
        firm = ExponentialProductionFirm(A,k)
        labor_demand = firm.get_labor_demand(p)
        L = labor_demand.eval_at_p(w)
        profit = firm.profit_at(p, w, L)
        self.sol = {'L':L, 'profit':profit}

        c_ = k*A
        k_ = k-1
        demand_curve = fr"\(w = {PolyEq([c_],'p L_d',[k_])} \)"
        c_ = A
        k_ = k-1
        demand_curve_distractor1 = fr"\(w = {PolyEq([c_],'p L_d',[k_])} \)"
        c_ = 1/(k*A)
        k_ = 1-k
        demand_curve_distractor2 = fr"\(w = {PolyEq([c_],'p L_d',[k_])} \)"
        c_ = k*A
        k_ = 1/k
        demand_curve_distractor3 = fr"\(w = {PolyEq([c_],'p L_d',[k_])} \)"

        c_ = asfrac(1/A, inline=False, maxdenom=A)
        k_ = asfrac(1/k, inline=True)
        if A==1:
            cost_func = fr"\(c(q) = w q^{{{k_}}}\)"
        else:
            cost_func = fr"\(c(q) = \left({c_}\right)^{{{k_}}} w q^{{{k_}}}\)"
        c_ = asfrac(A/k, inline=False, maxdenom=A)
        k_ = asfrac(1/k, inline=True)
        cost_func_distractor1 = fr"\(c(q) = {c_} w q^{{{k_}}} \)"
        c_ = asfrac(A/k, inline=False, maxdenom=A)
        k_ = asfrac(1/k, inline=True)
        cost_func_distractor2 = fr"\(c(q) = \left({c_}\right)^{{{k_}}} w q^{{{k_}}} \)"
        c_ = asfrac(1/A, inline=False, maxdenom=A)
        k_ = asfrac(k, inline=True)
        if A==1:
            cost_func_distractor3 = fr"\(c(q) = w q^{{{k_}}}\)"
        else:
            cost_func_distractor3 = fr"\(c(q) = \left({c_}\right)^{{{k_}}} w q^{{{k_}}}\)"

        setup_list = []
        setup = firm.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Write down the inverse labor demand curve."
        online_question = question
        answer = demand_curve
        online_answer = answer
        answers = [answer, demand_curve_distractor1, demand_curve_distractor2, demand_curve_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Write down the firm's total cost function, \(c(q)\), in terms of \(w\) and \(q\)."
        online_question = question
        answer = cost_func
        online_answer = answer
        answers = [answer, cost_func_distractor1, cost_func_distractor2, cost_func_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Calculate the choice of \(L\) that maximizes profit when the wage rate is \(w={w:g}\) and the output price is \(p={p:g}\)."
        online_question = question
        answer = L
        online_answer = fr"\(L = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"What is the maximum profit attainable when the wage rate is \(w={w:g}\) and the output price is \(p={p:g}\)?"
        online_question = question
        answer = profit
        online_answer = fr"\(\Pi = {answer:g}\)"
        answers = generate_distractors(answer, rng=rng)
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
        if self.sol['L']<1: return False
        if self.sol['L']>50: return False
        return True


@register
class ReturnsToScaleProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='cobb_douglas_consumer_problem'):
        rng = default_rng(rng)
        default_params = {'A':1,'a':1/2,'b':1/2,'delta':0.5}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        A, a, b, delta = params['A'], params['a'], params['b'], params['delta']
        if delta<0: 
            input_inc_dec = "reduces"
            output_inc_dec = "decrease"
        elif delta>0: 
            input_inc_dec = "increases"
            output_inc_dec = "increase"
        else: 
            input_inc_dec = "neither"
            output_inc_dec = "neither"
        if a+b==1: rts="constant returns to scale"
        elif a+b>1: rts="increasing returns to scale"
        else: rts="decreasing returns to scale"
        self.sol['rts'] = rts
        cb = CobbDouglas(A=A,a=a,b=b,x='L',y='K')
        setup_list = []
        setup = fr"""
A firm uses labor \(L\) and capital \(K\) to produce a commodity output. The firm's production function is:
$$f(L,K) = {cb}$$
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Does the firm exhibit increasing, decreasing, or constant returns to scale?"
        online_question = question
        answer = rts
        online_answer = answer
        answers = ["increasing returns to scale", "decreasing returns to scale", "constant returns to scale"]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,rng=rng)
        })
        question = fr"Suppose the firm {input_inc_dec} both its labor and capital input by \({np.abs(delta)*100:g}\%\). This would cause the firm's output to {output_inc_dec} by a factor of:"
        online_question = fr"Suppose the firm {input_inc_dec} both its labor and capital input by \({np.abs(delta)*100:g}\%\). This would cause the firm's output to {output_inc_dec} by a factor of what?"
        answer = np.abs((1+delta)**(a+b)-1)
        online_answer = fr"\({answer*100:g}\%\)"
        answers = generate_distractors(answer,K=4,delta=1.25,type='mul',rng=rng)
        answers = [fr"\({ans*100:g}\%\)" for ans in answers]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,sort=True,rng=rng)
        })
        question = fr"Suppose the firm {input_inc_dec} its labor input by \({np.abs(delta)*100:g}\%\) without changing its capital input. This would cause the firm's output to {output_inc_dec} by a factor of:"
        online_question = fr"Suppose the firm {input_inc_dec} its labor input by \({np.abs(delta)*100:g}\%\) without changing its capital input. This would cause the firm's output to {output_inc_dec} by a factor of what?"
        answer = np.abs((1+delta)**a-1)
        online_answer = fr"\({answer*100:g}\%\)"
        answers = generate_distractors(answer,K=4,delta=1.25,type='mul',rng=rng)
        answers = [fr"\({ans*100:g}\%\)" for ans in answers]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,sort=True,rng=rng)
        })
        question = fr"Suppose the firm {input_inc_dec} its capital input by \({np.abs(delta)*100:g}\%\) without changing its labor input. This would cause the firm's output to {output_inc_dec} by a factor of:"
        online_question = fr"Suppose the firm {input_inc_dec} its capital input by \({np.abs(delta)*100:g}\%\) without changing its labor input. This would cause the firm's output to {output_inc_dec} by a factor of what?"
        answer = np.abs((1+delta)**b-1)
        online_answer = fr"\({answer*100:g}\%\)"
        answers = generate_distractors(answer,K=4,delta=1.25,type='mul',rng=rng)
        answers = [fr"\({ans*100:g}\%\)" for ans in answers]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,sort=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list


@register
class CobbDouglasFirmProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='cobb_douglas_firm_problem'):
        rng = default_rng(rng)
        default_params = {'A':1, 'a':1/2, 'w':1, 'r':1}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        A, a, w, r = params['A'], params['a'], params['w'], params['r']
        firm = CobbDouglasFirm(A,a)
        L = firm.unit_cost_L(w,r)
        K = firm.unit_cost_K(w,r)
        unit_cost = firm.unit_cost(w,r)
        self.sol = {'L':L, 'K':K, 'unit_cost':unit_cost}
        setup_list = []
        setup = fr"""
A firm has a constant returns to scale production function over labor and capital given by:
$$ f(L,K) = {firm.cb} $$
The unit price of labor is \(w={w:g}\) and the unit price of capital is \(r={r:g}\).
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"What choice of labor \(L\) minimizes the cost to produce one unit of output?"
        online_question = question
        answer = L
        online_answer = fr"\(L = {answer:g}\)"
        answers = generate_distractors(answer, type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What choice of capital \(K\) minimizes the cost to produce one unit of output?"
        online_question = question
        answer = K
        online_answer = fr"\(K = {answer:g}\)"
        answers = generate_distractors(answer, type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the firm's cost to produce one unit?"
        online_question = question
        answer = unit_cost
        online_answer = fr"\(MC = {answer:g}\)"
        answers = generate_distractors(answer, type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list


@register
class CobbDouglasFirmGraphicalProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='cobb_douglas_firm_problem'):
        rng = default_rng(rng)
        default_params = {'L':4, 'K':4, 'w':1, 'r':1, 'xunit':1}
        xn = 13
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        L, K, w, r, xunit = params['L'], params['K'], params['w'], params['r'], params['xunit']
        unit_cost = w*L + r*K
        self.sol['unit_cost'] = unit_cost
        firm = get_cb_firm_from_point(L, K, w, r)
        setup_axis = Axis(xn=13,yn=13,xunit=xunit,yunit=xunit,xlab=r'$L$',ylab=r'$K$')
        passing = [(L+i*xunit, K+i*xunit) for i in np.arange(-xn,xn)]
        contours = LineContours(w, r, passing)
        self.contours = contours
        setup_axis.add(contours)
        setup_axis.add(firm)
        self.setup_axis = setup_axis
        setup_list = []
        setup = fr"""
A firm has a constant returns to scale production function. Its unit isoquant and isocost curves are shown in the diagram below:
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
The unit price of labor is \(w={w:g}\) and the unit price of capital is \(r={r:g}\).
"""
        online_setup = fr"""
A firm has a constant returns to scale production function. Its unit isoquant and isocost curves are shown in the diagram below:
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
The unit price of labor is \(w={w:g}\) and the unit price of capital is \(r={r:g}\).
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"What choice of labor \(L\) minimizes the cost to produce one unit of output?"
        online_question = question
        answer = L
        online_answer = fr"\(L = {answer:g}\)"
        answers = generate_distractors(answer, delta=xunit, type='add', must_positive=True,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What choice of capital \(K\) minimizes the cost to produce one unit of output?"
        online_question = question
        answer = K
        online_answer = fr"\(K = {answer:g}\)"
        answers = generate_distractors(answer, delta=xunit, type='add', must_positive=True,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the firm's cost to produce one unit?"
        online_question = question
        answer = unit_cost
        online_answer = fr"\(MC = {answer:g}\)"
        answers = generate_distractors(answer, type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if not is_divisible(self.params['L'], self.setup_axis.xunit): return False
        if not is_divisible(self.params['K'], self.setup_axis.yunit): return False
        return True


@register
class TechnicalChangeProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='technical_change_problem'):
        rng = default_rng(rng)
        default_params = {'A1':1, 'A2':2, 'a1':1/2, 'a2':1/2}
        xn = 13
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        A1, A2, a1, a2 = params['A1'], params['A2'], params['a1'], params['a2']
        if A2>A1:
            changetype = 'increase in total factor productivity'
        elif A2<A1:
            changetype = 'decrease in total factor productivity'
        elif a2>a1:
            changetype = 'technical change favoring labor'
        elif a2<a1:
            changetype = 'technical change favoring capital'
        else:
            changetype = 'no change'
        self.sol['changetype'] = changetype
        firm1 = CobbDouglasFirm(A=A1, a=a1, label='T1')
        firm2 = CobbDouglasFirm(A=A2, a=a2, color='red', linestyle='dashed', label='T2')
        if equals(A1,A2):
            xunit = (1/A1)/6
            yunit = (1/A1)/6
        else:
            yunit = (1/A1 + 1/A2)/12
            xunit = (1/A1 + 1/A2)/12
        setup_axis = Axis(xn=xn,yn=xn,xunit=xunit,yunit=yunit,noticklabels=True,xlab=r'$L$',ylab=r'$K$')
        setup_axis.add(firm1)
        setup_axis.add(firm2)
        self.setup_axis = setup_axis
        setup_list = []
        setup = fr"""
The diagram below illustrates a technological change in a firm's production function. The black solid line shows the firm's unit isoquant prior to the technical change, and the red dashed line shows the firm's unit isoquant subsequent to the technical change.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
"""
        online_setup = fr"""
The diagram below illustrates a technological change in a firm's production function. The black solid line shows the firm's unit isoquant prior to the technical change, and the red dashed line shows the firm's unit isoquant subsequent to the technical change.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"What type of technical change does this illustrate?"
        online_question = question
        answer = changetype
        online_answer = answer
        answers = [
            'increase in total factor productivity',
            'decrease in total factor productivity',
            'technical change favoring labor',
            'technical change favoring capital'
        ]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=False,shuffle=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        params = self.params
        A1, A2, a1, a2 = params['A1'], params['A2'], params['a1'], params['a2']
        if (A1!=A2) and (a1!=a2): return False
        if equals(A1,A2) and equals(a1,a2): return False
        return True
