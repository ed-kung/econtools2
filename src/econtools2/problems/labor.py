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
from econtools2.problems.production import *


@register
class WorkerProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='worker_problem'):
        rng = default_rng(rng)
        default_params = {'d':0.5,'k':2,'w':2}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        d, k, w = params['d'], params['k'], params['w']
        worker = Worker(d, k)
        L = worker.supply.eval_at_p(w)
        U = worker.utility_at(w, L)
        self.sol = {'L':L, 'U':U}

        c_ = k*d
        k_ = k-1
        supply_curve = fr"\(w = {PolyEq([c_],'L_s',[k_])} \)"
        
        c_ = d
        k_ = k-1
        supply_curve_distractor1 = fr"\(w = {PolyEq([c_],'L_s',[k_])} \)"

        c_ = 1/(k*d)
        k_ = 1-k
        supply_curve_distractor2 = fr"\(w = {PolyEq([c_],'L_s',[k_])} \)"
        
        c_ = k*d
        k_ = 1/k
        supply_curve_distractor3 = fr"\(w = {PolyEq([c_],'L_s',[k_])} \)"

        setup_list = []
        setup = worker.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "Write down the inverse labor supply curve."
        online_question = question
        answer = supply_curve
        online_answer = answer
        answers = [answer, supply_curve_distractor1, supply_curve_distractor2, supply_curve_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Calculate the choice of \(L\) that maximizes utility when the wage rate is \(w={w:g}\)."
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
        question = fr"What is the maximum utility attainable when the wage rate is \(w={w:g}\)?"
        online_question = question
        answer = U
        online_answer = fr"\(U = {answer:g}\)"
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
class CobbDouglasWorkerProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='cobb_douglas_worker_problem'):
        rng = default_rng(rng)
        default_params = {'x':6,'w':15,'yunit':15*5}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        x, w, yunit = self.params['x'], self.params['w'], self.params['yunit']
        T = 60 # time budget
        I = w*T
        y = I-w*x
        bc = BudgetConstraint(w,1,I)
        cb = get_cb_from_point(x, y, bc)
        U = cb.eval_at(x,y)
        self.sol = {'y':y, 'I':I}
        setup_axis = Axis(xn=13, yn=13, xunit=5, yunit=yunit, xlab='Leisure Hours', ylab='Income')
        solution_axis = Axis(xn=13, yn=13, xunit=5, yunit=yunit, xlab='Leisure Hours', ylab='Income')
        levels = get_cb_levels(cb, setup_axis, U)
        contours = CobbDouglasContours(cb, levels)
        self.contours = contours
        setup_axis.add(contours)
        solution_axis.add(contours)
        point = Point(x,y,text='A',position='sw')
        solution_axis.add(bc, point)
        self.setup_axis = setup_axis
        self.solution_axis = solution_axis
        setup_list = []
        setup = fr"""
A worker has up to 60 hours a week to spend on working or on leisure. If he works, he makes a wage rate of \(w={w:g}\) dollars per hour. The worker's 
indifference curves over leisure and income are shown in the diagram below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
"""
        online_setup = fr"""
A worker has up to 60 hours a week to spend on working or on leisure. If he works, he makes a wage rate of \(w={w:g}\) dollars per hour. The worker's 
indifference curves over leisure and income are shown in the diagram below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Draw the worker's budget constraint. Label the optimal point A."
        online_question = question
        answer = fr"\begin{{center}}\includegraphics[width=3in]{{{name}_sol.png}}\end{{center}}"
        online_answer = f'<img src = "{get_config().image_url_prefix}/{name}_sol.png">'
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": None
        })
        question = fr"How many hours per week does the worker choose to work?"
        online_question = question
        answer = T - x
        online_answer = fr"\({answer:g}\) hours"
        answers = generate_distractors(answer,delta=5,type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })        
        question = fr"How income per week does the worker make?"
        online_question = question
        answer = y
        online_answer = fr"\({answer:g}\) dollars"
        answers = generate_distractors(answer,delta=setup_axis.yunit,type='add',rng=rng)
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
        if not is_divisible(self.params['x'], self.setup_axis.xunit): return False
        if not is_divisible(self.sol['y'], self.setup_axis.yunit): return False
        if not is_divisible(self.sol['I'], self.setup_axis.yunit): return False
        if not is_divisible(self.setup_axis.yunit, 1): return False
        return True


@register
class WageChangeProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='wage_change_problem'):
        rng = default_rng(rng)
        default_params = {'x1':6,'w1':15,'x2':6,'w2':30}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        x1, w1, x2, w2 = params['x1'], params['w1'], params['x2'], params['w2']
        T = 60 # time budget
        I1 = w1*T
        I2 = w2*T
        y1 = (I1 - w1*x1)
        y2 = (I2 - w2*x2)
        bc1 = BudgetConstraint(w1,1,I1)
        bc2 = BudgetConstraint(w2,1,I2)
        setup_axis = Axis(xn=13, yn=13, xunit=5, yunit=5*T, xlab='Leisure Hours', ylab='Income')
        solution_axis = Axis(xn=13, yn=13, xunit=5, yunit=5*T, xlab='Leisure Hours', ylab='Income')
        contours = CESContours(x1,y1,x2,y2,bc1,bc2,setup_axis)
        self.contours = contours
        setup_axis.add(contours)
        solution_axis.add(contours)
        point1 = Point(x1,y1,text='A',position='sw')
        point2 = Point(x2,y2,text='B',position='sw')
        solution_axis.add(bc1, bc2, point1, point2)
        self.setup_axis = setup_axis
        self.solution_axis = solution_axis
        if w1>w2:
            wage_inc_or_dec = 'decreases'
        elif w1<w2:
            wage_inc_or_dec = 'increases'
        else:
            wage_inc_or_dec = 'neither'
        if x1>x2:
            labor_inc_or_dec = 'increase'
        elif x1<x2:
            labor_inc_or_dec = 'decrease'
        else:
            labor_inc_or_dec = 'neither increase nor decrease'
        self.sol = {'y1':y1, 'y2':y2, 'I1':I1, 'I2':I2, 'wage_inc_or_dec':wage_inc_or_dec, 'labor_inc_or_dec':labor_inc_or_dec}
        setup_list = []
        setup = fr"""
A worker has up to 60 hours a week to spend on working or on leisure. If he works, he makes a wage rate of \(w={w1:g}\) dollars per hour. The worker's 
indifference curves over leisure and income are shown in the diagram below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
One day, the worker's hourly wage changes to \(w^\prime = {w2:g}\).
"""
        online_setup = fr"""
A worker has up to 60 hours a week to spend on working or on leisure. If he works, he makes a wage rate of \(w={w1:g}\) dollars per hour. The worker's 
indifference curves over leisure and income are shown in the diagram below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
One day, the worker's hourly wage changes to \(w^\prime = {w2:g}\).
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Draw the worker's budget constraint at both the initial wage and the subsequent wage. Label the optimal point at the initial wage A. Label the optimal point at the subsequent wage B."
        online_question = question
        answer = fr"\begin{{center}}\includegraphics[width=3in]{{{name}_sol.png}}\end{{center}}"
        online_answer = f'<img src = "{get_config().image_url_prefix}/{name}_sol.png">'
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": None
        })
        question = fr"How many hours per week does the worker work at the initial wage, \(w\)?"
        online_question = question
        answer = T - x1
        online_answer = fr"\({answer:g}\) hours"
        answers = generate_distractors(answer,delta=5,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })        
        question = fr"How many hours per week does the worker work at the subsequent wage, \(w^\prime\)?"
        online_question = question
        answer = T - x2
        online_answer = fr"\({answer:g}\) hours"
        answers = generate_distractors(answer,delta=5,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"Did the number of hours worked increase or decrease as a result of the wage change?"
        online_question = question
        if labor_inc_or_dec=='increase': answer = "increase"
        elif labor_inc_or_dec=='decrease': answer = "decrease"
        else: answer = "neither increase nor decrease"
        online_answer = answer
        answers = ["increase", "decrease", "neither increase nor decrease"]
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
    def check_solution(self):
        if self.contours.ces.bad: return False
        if self.contours.ces.a<=0: return False
        if self.contours.ces.a>=1: return False
        if equals(self.params['w1'], self.params['w2']): return False
        if not is_divisible(self.setup_axis.yunit, 1): return False
        if not is_divisible(self.sol['I1'], self.setup_axis.yunit): return False
        if not is_divisible(self.sol['I2'], self.setup_axis.yunit): return False
        if not is_divisible(self.sol['y1'], self.setup_axis.yunit): return False
        if not is_divisible(self.sol['y2'], self.setup_axis.yunit): return False
        return True


@register
class IncomeSupportProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='income_support_problem'):
        rng = default_rng(rng)
        default_params = {'x':6,'w':15,'yunit':15*5,'ymin':15*5}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        x, w, yunit, ymin = self.params['x'], self.params['w'], self.params['yunit'], self.params['ymin']
        T = 60 # time budget
        I = w*T
        y = I-w*x
        bc = BudgetConstraint(w,1,I,color='red')
        ibc = IncomeSupportBudget(bc, ymin, color='black')
        cb = get_cb_from_point(x, y, bc)
        setup_axis = Axis(xn=13, yn=13, xunit=5, yunit=yunit, xlab='Leisure Hours', ylab='Income')
        solution_axis = Axis(xn=13, yn=13, xunit=5, yunit=yunit, xlab='Leisure Hours', ylab='Income')
        U_work = cb.eval_at(x,y)
        U_nowork = cb.eval_at(T,ymin)
        if U_work > U_nowork:
            work = 'yes'
            x_choice = x
            y_choice = y
            income_inc_or_dec = 'neither increase nor decrease'
            levels = get_cb_levels(cb, setup_axis, U_work)
        elif U_work < U_nowork:
            work = 'no'
            x_choice = T
            y_choice = ymin
            levels = get_cb_levels(cb, setup_axis, U_work, U_nowork)
            if ymin>y:
                income_inc_or_dec = 'increase'
            elif ymin<y:
                income_inc_or_dec = 'decrease'
            else:
                income_inc_or_dec = 'neither increase nor decrease'
        else:
            work = 'not enough information'
            x_choice = x
            y_choice = y
            income_inc_or_dec = 'not enough information'
            levels = get_cb_levels(cb, setup_axis, U_work)
        self.sol = {'y':y, 'I':I, 'U_work':U_work, 'U_nowork':U_nowork, 'work':work, 'x_choice':x_choice, 'y_choice':y_choice, 'income_inc_or_dec':income_inc_or_dec}
        contours = CobbDouglasContours(cb, levels)
        self.contours = contours
        setup_axis.add(contours)
        solution_axis.add(contours)
        pointB = Point(x_choice,y_choice,text='B',position='ne')
        pointA = Point(x,y,text='A',position='sw')
        solution_axis.add(bc, ibc, pointA, pointB)
        self.setup_axis = setup_axis
        self.solution_axis = solution_axis
        setup_list = []
        setup = fr"""
A worker has up to 60 hours a week to spend on working or on leisure. If he works, he makes a wage rate of \(w={w:g}\) dollars per hour. The worker's 
indifference curves over leisure and income are shown in the diagram below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
In addition, the government provides minimum income support of up to \({ymin:g}\) dollars a week. That is, if someone earns less than \({ymin:g}\) dollars a week, the government supplements their income until it reaches \({ymin:g}\) a week.
"""
        online_setup = fr"""
A worker has up to 60 hours a week to spend on working or on leisure. If he works, he makes a wage rate of \(w={w:g}\) dollars per hour. The worker's 
indifference curves over leisure and income are shown in the diagram below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
In addition, the government provides minimum income support of up to \({ymin:g}\) dollars a week. That is, if someone earns less than \({ymin:g}\) dollars a week, the government supplements their income until it reaches \({ymin:g}\) a week.
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Draw the worker's budget constraint without income support. Label the optimal point A. Draw the worker's budget constraint with the income support. Label the optimal point B."
        online_question = question
        answer = fr"\begin{{center}}\includegraphics[width=3in]{{{name}_sol.png}}\end{{center}}"
        online_answer = f'<img src = "{get_config().image_url_prefix}/{name}_sol.png">'
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": None
        })
        question = fr"If there were no income support, how many hours per week does the worker work?"
        online_question = question
        answer = T-x
        online_answer = fr"{answer:,g} hours"
        answers = generate_distractors(answer,delta=5,type='add',rng=rng)
        sol = 0
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,sort=True,numerical=True,rng=rng)
        })
        question = fr"If there were no income support, how much income per week does the worker make?"
        online_question = question
        answer = y
        online_answer = fr"{answer:,g} dollars"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng)
        sol = 0
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,sort=True,numerical=True,rng=rng)
        })
        question = fr"With the availability of income support, how many hours per week does the worker work?"
        online_question = question
        if work=='yes': answer = T-x
        elif work=='no': answer = 0
        else: raise ValueError("Invalid solution for work")
        online_answer = fr"{answer} hours"
        distractors = [x for x in range(0, 65, 5) if (x!=answer)]
        answers = [answer] + rng.choice(distractors, size=3, replace=False).tolist()
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,sort=True,rng=rng)
        })
        question = fr"With the availability of income support, how much income per week does the worker get?"
        online_question = question
        answer = y_choice
        online_answer = fr"{answer:,g} dollars"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng)
        sol = 0
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,sort=True,numerical=True,rng=rng)
        })
        question = fr"Does the availability of the income support policy increase or decrease the worker's actual income?"
        online_question = question
        answer = income_inc_or_dec
        online_answer = answer
        answers = ['increase', 'decrease', 'neither increase nor decrease']
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
    def check_solution(self):
        if not is_divisible(self.params['x'], self.setup_axis.xunit): return False
        if not is_divisible(self.sol['y'], self.setup_axis.yunit): return False
        if not is_divisible(self.sol['I'], self.setup_axis.yunit): return False
        if not is_divisible(self.params['ymin'], self.setup_axis.yunit): return False
        if not is_divisible(self.setup_axis.yunit, 1): return False
        if np.abs(self.sol['U_work'] - self.sol['U_nowork'])/self.sol['U_work']<0.1: return False
        return True


@register
class ExponentialLaborMarketProblem(GenericProblem):
    # u(L) = wL - d*L^kw
    # f(L) = A*L^kf
    def __init__(self, params=None, rng=None, name='exponential_labor_market_problem'):
        rng = default_rng(rng)
        default_params = {'A':2,'kf':1/2,'d':1/2,'kw':3/2,'p':1}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        A, kf, d, kw, p = params['A'], params['kf'], params['d'], params['kw'], params['p']
        assert kf<1
        assert kw>1
        firm = ExponentialProductionFirm(A,kf)
        worker = Worker(d,kw)
        labor_demand = ExponentialDemand(p*A*kf, kf-1)
        labor_supply = ExponentialSupply(d*kw, kw-1)
        labor_market = ExponentialMarket(labor_demand, labor_supply)
        w = labor_market.eq['p']
        L = labor_market.eq['q']
        U = worker.utility_at(w, L)
        profit = firm.profit_at(p,w,L)
        firm_problem = ExponentialProductionFirmProblem(params={'A':A,'k':kf,'w':w,'p':p}, rng=rng)
        worker_problem = WorkerProblem(params={'d':d,'k':kw,'w':w}, rng=rng)
        self.sol = {'w':w, 'L':L, 'U':U, 'profit':profit}
        setup_list = []
        setup = fr"""
{worker.setup()}

{firm.setup()}

The current commodity price is \(p={p:g}\).
"""
        online_setup = fr"""
<p>{worker.setup()}</p>

<p>{firm.setup()}</p>

<p>The current commodity price is \(p={p:g}\).</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
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
        question = fr"Calculate the equilibrium utility of the worker."
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
        question_list.append(worker_problem.question_list[0])
        question_list.append(firm_problem.question_list[0])
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['L']<=1: return False
        if self.sol['L']>50: return False
        if self.sol['w']<=1: return False
        if self.sol['w']>50: return False
        return True
