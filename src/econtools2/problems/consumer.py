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
class LinearConsumerProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='linear_consumer_problem'):
        rng = default_rng(rng)
        default_params = {'a':12,'b':1,'p':6}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b, p = params['a'], params['b'], params['p']
        consumer = LinearConsumer(a,b)
        q = consumer.demand.eval_at_p(p)
        U = consumer.utility_at(p,q)
        self.sol = {'q':q, 'U':U}
        
        foc =             fr"\( {PolyEq([a,-b],'q',[0,1]).print()} - p = 0 \)"
        foc_distractor1 = fr"\( {PolyEq([a,-0.5*b],'q',[0,1]).print()} - p = 0 \)"
        foc_distractor2 = fr"\( {PolyEq([a,-b],'q',[0,1]).print()} = 0 \)"
        foc_distractor3 = fr"\( {PolyEq([a,b],'q',[0,1]).print()} - p = 0 \)"
        
        demand_curve =             fr"\( q_d = {PolyEq([a/b,-1/b],'p',[0,1])} \)"
        demand_curve_distractor1 = fr"\( q_d = {PolyEq([a/b,1/b],'p',[0,1])} \)"
        demand_curve_distractor2 = fr"\( q_d = {PolyEq([a],'p',[-1])}\)"
        demand_curve_distractor3 = rng.choice([
            fr"\(q_d = {PolyEq([2*a/b,-2/b],'p',[0,1])}\)",
            fr"\(q_d = {PolyEq([0.5*a/b,-0.5/b],'p',[0,1])}\)"
        ])
        
        setup_list = []
        setup = consumer.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "Write down the consumer's first order condition."
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
        question = "Write down the consumer's demand curve."
        online_question = question
        answer = demand_curve
        online_answer = answer
        answers = [demand_curve, demand_curve_distractor1, demand_curve_distractor2, demand_curve_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Calculate the choice of \(q\) that maximizes utility when price is \(p={p:g}\)."
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
        question = fr"Calculate the maximum utility attained when price is \(p={p:g}\)."
        online_question = question
        answer = U
        online_answer = fr"\(U = {answer}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        self.consumer = consumer
        self.setup_list = setup_list
        self.question_list = question_list


@register
class LogConsumerProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='log_consumer_problem'):
        rng = default_rng(rng)
        default_params = {'a':12,'p':2}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, p = params['a'], params['p']
        consumer = LogConsumer(a)
        q = consumer.demand.eval_at_p(p)
        U = consumer.utility_at(p,q)
        self.sol = {'q':q, 'U':U}
        
        foc =             fr"\({PolyEq([a,-1],['q','p'],[-1,1])} = 0 \)"
        foc_distractor1 = fr"\({PolyEq([a,0],['q','p'],[-1,1])} = 0 \)"
        foc_distractor2 = fr"\({PolyEq([a,-1],['q','p'],[1,1])} = 0 \)"
        foc_distractor3 = fr"\({PolyEq([a,-1,-1],['q','q','p'],[0,1,1])} = 0 \)"

        demand_curve =             fr"\(q_d = {PolyEq([a],'p',[-1])}) \)"
        demand_curve_distractor1 = fr"\(q_d = {PolyEq([a,-1],'p',[0,1])} \)"
        demand_curve_distractor2 = fr"\(q_d = {PolyEq([1/a],'p',[-1]).print(maxdenom=a)} \)"
        demand_curve_distractor3 = fr"\(q_d = {PolyEq([a],'e^{-p}',[1])}\)"
        
        demand_curve = fr"\(q_d = {consumer.demand.print()}\)"
        setup_list = []
        setup = consumer.setup()
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = "Write down the consumer's first order condition."
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
        question = "Write down the consumer's demand curve."
        online_question = question
        answer = demand_curve
        online_answer = answer
        answers = [demand_curve, demand_curve_distractor1, demand_curve_distractor2, demand_curve_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Calculate the choice of \(q\) that maximizes utility when price is \(p={p:g}\)."
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
        question = fr"Calculate the maximum utility attained when price is \(p={p:g}\)."
        online_question = question
        answer = U
        online_answer = fr"\(U = {answer}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        self.consumer = consumer
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['U']<=0: return False
        return True


@register
class CobbDouglasConsumerProblem(GenericProblem):
    # u(x,y) = x^a y^b s.t. px x + py y = I
    def __init__(self, params=None, rng=None, name='cobb_douglas_consumer_problem'):
        rng = default_rng(rng)
        default_params = {'a':1/2,'b':1/2,'px':1,'py':1,'I':12,'xunit':1,'yunit':1,'xn':13,'yn':13}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b, px, py, I, xunit, yunit, xn, yn = params['a'], params['b'], params['px'], params['py'], params['I'], params['xunit'], params['yunit'], params['xn'], params['yn']
        cobb_douglas = CobbDouglas(A=1, a=a, b=b)
        budget_constraint = BudgetConstraint(px=px, py=py, I=I)
        consumer = CobbDouglasConsumer(cobb_douglas, budget_constraint)
        setup_axis = Axis(xn=xn,yn=yn,xunit=xunit,yunit=yunit)
        setup_axis2 = Axis(xn=xn,yn=yn,xunit=xunit,yunit=yunit)
        solution_axis = Axis(xn=xn,yn=yn,xunit=xunit,yunit=yunit)
        x = consumer.x
        y = consumer.y
        U = consumer.U
        self.sol = {'x':x, 'y':y, 'U':U}
        self.consumer = consumer
        self.cobb_douglas = cobb_douglas
        self.budget_constraint = budget_constraint
        levels = get_cb_levels(cobb_douglas, setup_axis, U1=U)
        self.levels = levels
        contours = CobbDouglasContours(cobb_douglas, levels)
        point = Point(x,y,text='A',position='sw')
        setup_axis.add(contours)
        setup_axis2.add(contours)
        setup_axis2.add(budget_constraint)
        solution_axis.add(contours)
        solution_axis.add(budget_constraint)
        solution_axis.add(point)
        self.setup_axis = setup_axis
        self.setup_axis2 = setup_axis2
        self.solution_axis = solution_axis
        A_, a_, b_, px_ = a, a-1, b, px
        focx    = fr"\({CobbDouglas(A_,'x',a_,'y',b_)} = {px_:g} \lambda\)"
        A_, a_, b_, px_ = 1, a-1, b, px
        focx_d1 = fr"\({CobbDouglas(A_,'x',a_,'y',b_)} = \lambda\)"
        A_, a_, b_, px_ = a, a-1, b, px
        focx_d2 = fr"\({CobbDouglas(A_,'x',a_,'y',b_)} = {px_:g} \)"
        A_, a_, b_, px_ = b, a, b-1, py
        focx_d3 = fr"\({CobbDouglas(A_,'x',a_,'y',b_)} = {px_:g} \lambda\)"
        A_, a_, b_, py_ = b, a, b-1, py
        focy    = fr"\({CobbDouglas(A_,'x',a_,'y',b_)} = {py_:g} \lambda\)"
        A_, a_, b_, py_ = 1, a, b-1, py
        focy_d1 = fr"\({CobbDouglas(A_,'x',a_,'y',b_)} = \lambda\)"
        A_, a_, b_, py_ = b, a, b-1, py
        focy_d2 = fr"\({CobbDouglas(A_,'x',a_,'y',b_)} = {py_:g} \)"
        A_, a_, b_, py_ = a, a-1, b, px
        focy_d3 = fr"\({CobbDouglas(A_,'x',a_,'y',b_)} = {py_:g} \lambda\)"
        setup_list = []
        setup = fr"""
A consumer with income \(I = {I:g}\) has a utility function over two goods, \(x\) and \(y\):

$$ u(x,y) = {cobb_douglas} $$

The price of good \(x\) is \(p_x = {px:g}\) and the price of good \(y\) is \(p_y = {py:g}\).
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        setup = fr"""
Solve:

$$ \max_{{x,y}} ~ {cobb_douglas} ~ \text{{s.t.}} ~ {PolyEq([px,py],['x','y'],[1,1])} = {I:g}$$
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        setup = fr"""
A consumer with income \(I = {I:g}\) has a utility function over two goods, \(x\) and \(y\), represented
by the indifference curves shown below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
The price of good \(x\) is \(p_x = {px:g}\) and the price of good \(y\) is \(p_y = {py:g}\).
"""
        online_setup = fr"""
A consumer with income \(I = {I:g}\) has a utility function over two goods, \(x\) and \(y\), represented
by the indifference curves shown below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
The price of good \(x\) is \(p_x = {px:g}\) and the price of good \(y\) is \(p_y = {py:g}\).
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        setup = fr"""
A consumer's indifference curves and budget constraint over two goods, \(x\) and \(y\), is shown below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup2.png}}
\end{{center}}
"""
        online_setup = fr"""
A consumer's indifference curves and budget constraint over two goods, \(x\) and \(y\), is shown below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup2.png">
</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Draw the consumer's budget constraint and label the optimal point A."
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
        question = fr"What is the consumer's optimal choice of \(x\)?"
        online_question = question
        answer = x
        online_answer = fr"\(x = {answer:g}\)"
        answers = generate_distractors(answer,delta=xunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the optimal value of \(x\)?"
        online_question = question
        answer = x
        online_answer = fr"\(x = {answer:g}\)"
        answers = generate_distractors(answer,delta=xunit,type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the consumer's optimal choice of \(y\)?"
        online_question = question
        answer = y
        online_answer = fr"\(y = {answer:g}\)"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the optimal value of \(y\)?"
        online_question = question
        answer = y
        online_answer = fr"\(y = {answer:g}\)"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"Calculate the consumer's utility at the optimal choice."
        online_question = question
        answer = U
        online_answer = fr"\(U = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the maximum value of the function?"
        online_question = question
        answer = U
        online_answer = fr"\(f(x,y) = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"Write the first order condition for \(x\)."
        online_question = question
        answer = focx
        online_answer = answer
        answers = [focx, focx_d1, focx_d2, focx_d3]
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Write the first order condition for \(y\)."
        online_question = question
        answer = focy
        online_answer = answer
        answers = [focy, focy_d1, focy_d2, focy_d3]
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if len(self.levels)>=14: return False
        if len(self.levels)<=6: return False
        if not is_divisible(self.sol['x'], self.params['xunit']): return False
        if not is_divisible(self.sol['y'], self.params['yunit']): return False
        if not is_divisible(self.budget_constraint.xint, self.params['xunit']): return False
        if not is_divisible(self.budget_constraint.yint, self.params['yunit']): return False
        if self.sol['x'] >= self.setup_axis.xmax: return False
        if self.sol['y'] >= self.setup_axis.ymax: return False
        if self.budget_constraint.xint >= self.setup_axis.xmax: return False
        if self.budget_constraint.yint >= self.setup_axis.ymax: return False
        return True


@register
class PerfectSubstitutesProblem(GenericProblem):
    # u(x,y) = ax + by s.t. px x + py y = I
    def __init__(self, params=None, rng=None, name='perfect_substitutes_problem'):
        rng = default_rng(rng)
        default_params = {'a':2,'b':1,'px':1,'py':1,'I':12,'xunit':1,'yunit':1,'xn':13,'yn':13}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b, px, py, I, xunit, yunit, xn, yn = params['a'], params['b'], params['px'], params['py'], params['I'], params['xunit'], params['yunit'], params['xn'], params['yn']
        budget_constraint = BudgetConstraint(px=px, py=py, I=I)
        self.budget_constraint = budget_constraint
        if (a*py - b*px)>0:
            x, y = I/px, 0
            passing = [(x_,0) for x_ in np.arange(xunit, xunit*5*xn, xunit)]
        elif (a*py - b*px)<0:
            x, y = 0, I/py
            passing = [(0,y_) for y_ in np.arange(yunit, yunit*5*yn, yunit)]
        else:
            x = 0.5*I/px
            y = 0.5*I/py
            passing = [(i*xunit,i*yunit) for i in np.arange(1, xn)]
        U = a*x + b*y
        self.sol = {'x':x, 'y':y, 'U':U}
        contours = LineContours(a,b,passing)
        point = Point(x,y,text='A',position='ne')
        setup_axis = Axis(xn=xn,yn=yn,xunit=xunit,yunit=yunit)
        solution_axis = Axis(xn=xn,yn=yn,xunit=xunit,yunit=yunit)
        setup_axis.add(contours)
        solution_axis.add(contours)
        solution_axis.add(budget_constraint)
        solution_axis.add(point)
        self.setup_axis = setup_axis
        self.solution_axis = solution_axis
        setup_list = []
        setup = fr"""
A consumer with income \(I = {I:g}\) has a utility function over two goods, \(x\) and \(y\), represented
by the indifference curves shown below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
The price of good \(x\) is \(p_x = {px:g}\) and the price of good \(y\) is \(p_y = {py:g}\).
"""
        online_setup = fr"""
A consumer with income \(I = {I:g}\) has a utility function over two goods, \(x\) and \(y\), represented
by the indifference curves shown below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
The price of good \(x\) is \(p_x = {px:g}\) and the price of good \(y\) is \(p_y = {py:g}\).
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Draw the consumer's budget constraint and label the optimal point A."
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
        question = fr"What is the consumer's optimal choice of \(x\)?"
        online_question = question
        answer = x
        online_answer = fr"\(x = {answer:g}\)"
        answers = generate_distractors(answer,delta=xunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })        
        question = fr"What is the consumer's optimal choice of \(y\)?"
        online_question = question
        answer = y
        online_answer = fr"\(y = {answer:g}\)"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What best describes the relationship between goods \(x\) and \(y\)?"
        online_question = question
        answer = "perfect substitutes"
        online_answer = answer
        answers = [answer, "perfect complements", "inferior goods", "Giffen goods"]
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if (self.sol['x']>0 and self.sol['y']>0): return False
        if not is_divisible(self.sol['x'], self.params['xunit']): return False
        if not is_divisible(self.sol['y'], self.params['yunit']): return False
        if not is_divisible(self.budget_constraint.xint, self.params['xunit']): return False
        if not is_divisible(self.budget_constraint.yint, self.params['yunit']): return False
        if self.sol['x'] >= self.setup_axis.xmax: return False
        if self.sol['y'] >= self.setup_axis.ymax: return False
        if self.budget_constraint.xint >= self.setup_axis.xmax: return False
        if self.budget_constraint.yint >= self.setup_axis.ymax: return False
        return True


@register
class PerfectComplementsProblem(GenericProblem):
    # u(x,y) = min(x,y)
    def __init__(self, params=None, rng=None, name='perfect_substitutes_problem'):
        rng = default_rng(rng)
        default_params = {'px':1,'py':1,'I':12,'xunit':1,'yunit':1,'xn':13,'yn':13}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        px, py, I, xunit, yunit, xn, yn = params['px'], params['py'], params['I'], params['xunit'], params['yunit'], params['xn'], params['yn']
        budget_constraint = BudgetConstraint(px=px, py=py, I=I)
        self.budget_constraint = budget_constraint
        x = I/(px+py)
        y = I/(px+py)
        U = x
        self.sol = {'x':x, 'y':y, 'U':U}
        contours = LeontieffContours(xunit=xunit,yunit=yunit,xn=xn,yn=yn)
        point = Point(x,y,text='A',position='sw')
        setup_axis = Axis(xn=xn,yn=yn,xunit=xunit,yunit=yunit)
        solution_axis = Axis(xn=xn,yn=yn,xunit=xunit,yunit=yunit)
        setup_axis.add(contours)
        solution_axis.add(contours)
        solution_axis.add(budget_constraint)
        solution_axis.add(point)
        self.setup_axis = setup_axis
        self.solution_axis = solution_axis
        setup_list = []
        setup = fr"""
A consumer with income \(I = {I:g}\) has a utility function over two goods, \(x\) and \(y\), represented
by the indifference curves shown below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
The price of good \(x\) is \(p_x = {px:g}\) and the price of good \(y\) is \(p_y = {py:g}\).
"""
        online_setup = fr"""
A consumer with income \(I = {I:g}\) has a utility function over two goods, \(x\) and \(y\), represented
by the indifference curves shown below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
The price of good \(x\) is \(p_x = {px:g}\) and the price of good \(y\) is \(p_y = {py:g}\).
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Draw the consumer's budget constraint and label the optimal point A."
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
        question = fr"What is the consumer's optimal choice of \(x\)?"
        online_question = question
        answer = x
        online_answer = fr"\(x = {answer:g}\)"
        answers = generate_distractors(answer,delta=xunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })        
        question = fr"What is the consumer's optimal choice of \(y\)?"
        online_question = question
        answer = y
        online_answer = fr"\(y = {answer:g}\)"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What best describes the relationship between goods \(x\) and \(y\)?"
        online_question = question
        answer = "perfect complements"
        online_answer = answer
        answers = [answer, "perfect substitutes", "inferior goods", "quasilinear goods"]
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if not is_divisible(self.sol['x'], self.params['xunit']): return False
        if not is_divisible(self.sol['y'], self.params['yunit']): return False
        if not is_divisible(self.budget_constraint.xint, self.params['xunit']): return False
        if not is_divisible(self.budget_constraint.yint, self.params['yunit']): return False
        if self.sol['x'] >= self.setup_axis.xmax: return False
        if self.sol['y'] >= self.setup_axis.ymax: return False
        if self.budget_constraint.xint >= self.setup_axis.xmax: return False
        if self.budget_constraint.yint >= self.setup_axis.ymax: return False
        return True


@register
class PriceChangeProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='cobb_douglas_consumer_problem'):
        rng = default_rng(rng)
        default_params = {'x1':7,'px1':1,'py1':1,'x2':8,'px2':1,'py2':2,'I':12,'xunit':1,'xn':13}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        x1, px1, py1, x2, px2, py2, I, xunit, xn = params['x1'], params['px1'], params['py1'], params['x2'], params['px2'], params['py2'], params['I'], params['xunit'], params['xn']
        yunit = xunit
        y1 = (I - px1*x1)/py1
        y2 = (I - px2*x2)/py2
        bc1 = BudgetConstraint(px1,py1,I)
        bc2 = BudgetConstraint(px2,py2,I)
        setup_axis = Axis(xn=xn, xunit=xunit, yn=xn, yunit=yunit)
        solution_axis = Axis(xn=xn, xunit=xunit, yn=xn, yunit=yunit)
        contours = CESContours(x1,y1,x2,y2,bc1,bc2,setup_axis)
        self.contours = contours
        setup_axis.add(contours)
        solution_axis.add(contours)
        point1 = Point(x1,y1,text='A',position='sw')
        point2 = Point(x2,y2,text='B',position='sw')
        solution_axis.add(bc1, bc2, point1, point2)
        self.setup_axis = setup_axis
        self.solution_axis = solution_axis
        if px2!=px1:
            price_that_changed = 'x'
            myp2 = px2
            if (y2-y1)/(px2-px1) > 0:
                comp_or_sub = 'substitutes'
            elif (y2-y1)/(px2-px1) < 0:
                comp_or_sub = 'complements'
            else:
                comp_or_sub = 'neither'
        if py2!=py1:
            myp2 = py2
            price_that_changed = 'y'
            if (x2-x1)/(py2-py1) > 0:
                comp_or_sub = 'substitutes'
            elif (x2-x1)/(py2-py1) < 0:
                comp_or_sub = 'complements'
            else:
                comp_or_sub = 'neither'
        self.sol = {'y1':y1, 'y2':y2, 'price_that_changed':price_that_changed, 'comp_or_sub':comp_or_sub}
        setup_list = []
        setup = fr"""
A consumer with income \(I={I:g}\) has utility over two goods, \(x\) and \(y\), shown by the indifference curves below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
The prices of the goods are initially \(p_x = {px1:g}\) and \(p_y = {py1:g}\). One day, the price of good \({price_that_changed}\) changes to \(p_{price_that_changed}^\prime = {myp2:g}\).
"""
        online_setup = fr"""
A consumer with income \(I={I:g}\) has utility over two goods, \(x\) and \(y\), shown by the indifference curves below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
The prices of the goods are initially \(p_x = {px1:g}\) and \(p_y = {py1:g}\). One day, the price of good \({price_that_changed}\) increases to \(p_{price_that_changed}^\prime = {myp2:g}\).
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Draw the consumer's budget constraint in both periods. Label the optimal point in the initial period A. Label the optimal point after the prices change B."
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
        question = fr"Did consumption of \(x\) increase or decrease as a result of the price change?"
        online_question = question
        if x2>x1: answer = "increase"
        elif x2<x1: answer = "decrease"
        else: answer = "neither"
        online_answer = answer
        answers = ["increase", "decrease", "neither"]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,shuffle=False,rng=rng)
        })
        question = fr"Did consumption of \(y\) increase or decrease as a result of the price change?"
        online_question = question
        if y2>y1: answer = "increase"
        elif y2<y1: answer = "decrease"
        else: answer = "neither"
        online_answer = answer
        answers = ["increase", "decrease", "neither"]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,shuffle=False,rng=rng)
        })
        question = fr"Are \(x\) and \(y\) complements or substitutes at current price levels?"
        online_question = question
        answer = comp_or_sub
        online_answer = answer
        answers = ["complements", "substitutes","neither"]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,shuffle=False,rng=rng)
        })
        question = fr"What is the optimal choice of \(x\) at the initial prices?"
        online_question = question
        answer = x1
        online_answer = fr"\({answer:g}\)"
        answers = generate_distractors(answer,delta=xunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the optimal choice of \(y\) at the initial prices?"
        online_question = question
        answer = y1
        online_answer = fr"\({answer:g}\)"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the optimal choice of \(x\) at the later prices?"
        online_question = question
        answer = x2
        online_answer = fr"\({answer:g}\)"
        answers = generate_distractors(answer,delta=xunit,type='add',rng=rng,must_positive=True)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the optimal choice of \(y\) at the later prices?"
        online_question = question
        answer = y2
        online_answer = fr"\({answer:g}\)"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng,must_positive=True)
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
        if self.contours.ces.bad: return False
        if self.contours.ces.a<=0: return False
        if self.contours.ces.a>=1: return False
        if self.params['x1']==self.params['x2']: return False
        if (self.params['px1']==self.params['px2']) and (self.params['py1']==self.params['py2']): return False
        if (self.params['px1']!=self.params['px2']) and (self.params['py1']!=self.params['py2']): return False
        if not is_divisible(self.sol['y1'], self.setup_axis.yunit): return False
        if not is_divisible(self.sol['y2'], self.setup_axis.yunit): return False
        if not is_divisible(self.params['x1'], self.setup_axis.xunit): return False
        if not is_divisible(self.params['x2'], self.setup_axis.xunit): return False
        if not is_divisible(self.params['I']/self.params['py1'], self.setup_axis.yunit): return False
        if not is_divisible(self.params['I']/self.params['py2'], self.setup_axis.yunit): return False
        if not is_divisible(self.params['I']/self.params['px1'], self.setup_axis.xunit): return False
        if not is_divisible(self.params['I']/self.params['px2'], self.setup_axis.xunit): return False
        return True


@register
class PublicSchoolProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='public_school_problem'):
        rng = default_rng(rng)
        default_params = {'x_private':6,'x_public':4,'px':1,'py':1,'I':12,'xunit':1,'xn':13}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        x_private, x_public, px, py, I, xunit, xn = params['x_private'], params['x_public'], params['px'], params['py'], params['I'], params['xunit'], params['xn']
        y_private = (I-px*x_private)/py
        bc = BudgetConstraint(px,py,I)
        cb = get_cb_from_point(x_private, y_private, bc)
        U_private = cb.eval_at(x_private, y_private)
        U_public = cb.eval_at(x_public, I/py)
        if equals(U_private, U_public):
            choice = 'unknown'
            xchoice = None
            ychoice = None
        elif U_private>U_public: 
            choice = 'private'
            xchoice = x_private
            ychoice = y_private
        else:
            choice = 'public'
            xchoice = x_public
            ychoice = I/py
        self.sol = {'y_private':y_private,'U_private':U_private, 'U_public':U_public, 'choice':choice, 'xchoice':xchoice, 'ychoice':ychoice}
        setup_axis = Axis(xn=xn,yn=xn,xunit=xunit,yunit=xunit, xlab="Education", ylab="Other Goods")
        solution_axis = Axis(xn=xn,yn=xn,xunit=xunit,yunit=xunit, xlab="Education", ylab="Other Goods")
        levels = get_cb_levels(cb, setup_axis, U_private)
        contours = CobbDouglasContours(cb, levels)
        setup_axis.add(contours)
        solution_axis.add(contours)
        private_point = Point(x=x_private, y=y_private, text='A', position='sw')
        public_point = Point(x=x_public, y=I/py, text='B', position='ne')
        solution_axis.add(private_point)
        setup_axis.add(bc)
        solution_axis.add(bc)
        solution_axis.add(public_point)
        self.setup_axis = setup_axis
        self.solution_axis = solution_axis
        setup_list = []
        setup = fr"""
A family can spend their income on either education or other goods. The diagram below shows the family's budget constraint when only private school options are available, as well as their indifference curves over education and other consumption.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
A public school option is also available which provides {x_public:g} units of education for free.
"""
        online_setup = fr"""
A family can spend their income on either education or other goods. The diagram below shows the family's budget constraint when only private school options are available, as well as their indifference curves over education and other consumption.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
A public school option is also available which provides {x_public:g} units of education for free.
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"On the diagram, label the optimal private school option A and label the public school option B."
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
        question = fr"Which option offers higher utility, public or private?"
        online_question = question
        if choice=="private": answer="private"
        elif choice=="public": answer="public"
        else: answer="they offer the same utility"
        online_answer = answer
        answers = ["public", "private", "they offer the same utility"]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,shuffle=False,rng=rng)
        })
        question = fr"Which option will the family choose?"
        online_question = question
        if choice=="private": answer="private"
        elif choice=="public": answer="public"
        else: answer="not enough information"
        online_answer = answer
        answers = ["public", "private", "not enough information"]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=True,shuffle=False,rng=rng)
        })
        question = fr"Does the availability of the public school option increase or decrease the amount of education consumed, relative to only private options?"
        online_question = question
        if choice=='private': answer="neither increase nor decrease"
        elif choice=='unknown': answer="not enough information"
        elif x_private>xchoice: answer="decrease"
        elif x_private<xchoice: answer="increase"
        else: answer="neither increase nor decrease"
        online_answer = answer
        answers = ["increase", "decrease", "neither increase nor decrease", "not enough information"]
        sol = answers.index(answer)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,horz=False,shuffle=False,rng=rng)
        })
        question = fr"If the family were to choose private school, how many units of education would it consume?"
        online_question = question
        answer = x_private
        online_answer = fr"\({answer:g}\)"
        answers = generate_distractors(answer,delta=xunit,type='add',rng=rng,must_positive=True)
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
        if not is_divisible(self.sol['y_private'], self.params['xunit']): return False
        if np.abs(self.sol['U_public'] - self.sol['U_private'])/self.sol['U_private']<0.1: return False
        return True
