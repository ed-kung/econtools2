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
class ExponentialRewriteProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='exponential_rewrite_problem'):
        rng = default_rng(rng)
        default_params = {'a':2,'k':-0.5,'y':4}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, k, y = params['a'], params['k'], params['y']
        x = (1/a)**(1/k) * y**(1/k)
        self.sol = {'x':x}
        func = fr"y = {PolyEq([a],'x',[k])}"
        c = asfrac(1/a, inline=False)
        invk = asfrac(1/k, inline=True)
        inv =             fr"\(x = \left({c}\right)^{{{invk}}} y^{{{invk}}}\)"
        inv_distractor1 = fr"\(x = {c} y^{{{invk}}}\)"
        c = asfrac(1/a, inline=False)
        invk = asfrac(-1/k, inline=True)
        inv_distractor2 = fr"\(x = \left({c}\right)^{{{invk}}} y^{{{invk}}}\)"
        c = asfrac(a, inline=False)
        invk = asfrac(1/k, inline=True)
        inv_distractor3 = fr"\(x = \left({c}\right)^{{{invk}}} y^{{{invk}}}\)"
        setup_list = []
        setup = fr"""
$$ {func} $$
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Write \(x\) as a function of \(y\)."
        online_question = question
        answer = inv
        online_answer = answer
        answers = [inv, inv_distractor1, inv_distractor2, inv_distractor3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Calculate \(x\) when \(y={params['y']:g}\)."
        online_question = question
        answer = x
        online_answer = fr"\(x = {answer:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,horz=True,sort=True,numerical=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        if self.sol['x']<0.1: return False
        return True


@register
class CobbDouglasSimplifyProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='cobbdouglas_simplify_problem'):
        rng = default_rng(rng)
        default_params = {'x':'x','y':'y','A':4,'B':12,'a':1/3,'b':-2/3}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        x, y, A, B, a, b = self.params['x'], self.params['y'], self.params['A'], self.params['B'], self.params['a'], self.params['b']
        assert A!=B
        assert a!=0
        assert b!=0
        assert sign(a)!=sign(b)
        cbtop = CobbDouglas(A,x,a,y,b)
        if (a>0):
            cbbot = CobbDouglas(B,x,a-1,y,b+1)
        else:
            cbbot = CobbDouglas(B,x,a+1,y,b-1)
        answer = simplifyCB(cbtop,cbbot)
        distractor1 = simplifyCB(CobbDouglas(B,x,cbtop.a,y,cbtop.b),CobbDouglas(A,x,cbbot.a,y,cbbot.b))
        distractor2 = simplifyCB(CobbDouglas(A,x,cbtop.b,y,cbtop.a),CobbDouglas(B,x,cbbot.b,y,cbbot.a))
        distractor3 = simplifyCB(CobbDouglas(B,x,cbtop.b,y,cbtop.a),CobbDouglas(A,x,cbbot.b,y,cbbot.a))
        setup_list = []
        setup_list.append({
            'setup':'',
            'online_setup':''
        })
        question_list = []
        question = fr"""
Simplify:
$$ \frac{{{cbtop.print()}}}{{{cbbot.print()}}} $$
"""
        online_question = question
        answer = answer
        online_answer = answer
        answers = [
            answer,
            distractor1,
            distractor2,
            distractor3
        ]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,horz=True,rng=rng)
        })
        self.cbtop = cbtop
        self.cbbot = cbbot
        self.setup_list = setup_list
        self.question_list = question_list


@register
class LogDifferencesProblem(GenericProblem):
    def __init__(self, params=None, rng=None, name='log_differences_problem'):
        rng = default_rng(rng)
        default_params = {'delta':0.05}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        delta = params['delta']
        assert delta!=0
        setup_list = []
        setup = fr"$$\ln y - \ln x = {delta:g}$$"
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Is \(y\) larger or smaller than \(x\)?"
        online_question = question
        answers = ["larger", "smaller"]
        if delta>0:
            answer = "larger"
            online_answer = "larger"
            sol = 0
        else:
            answer = "smaller"
            online_answer = "smaller"
            sol = 1
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,sol,shuffle=False,horz=True,rng=rng)
        })
        question = fr"By how much?"
        online_question = question
        answer = fr"\({np.abs(delta)*100:g}\%\)"
        online_answer = answer
        answers = [
            answer,
            fr"\({np.abs(delta):g}\%\)",
            fr"\({np.abs(delta)*10:g}\%\)",
            fr"\({np.abs(delta)*1000:g}\%\)"
        ]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,horz=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
        self.delta = delta


@register
class QuadraticOptimizationProblem(GenericProblem):
    # f(x) = ax - 0.5*bx^2 + c
    def __init__(self, params=None, rng=None, name='quadratic_optimization_problem'):
        rng = default_rng(rng)
        default_params = {'a':12,'b':1,'c':7}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b, c = params['a'], params['b'], params['c']
        x = a/b
        f = a*x - 0.5*b*x**2 + c
        self.sol = {'x':x, 'f':f}
        
        foc =             fr"\(f^\prime(x) = {PolyEq([a,    -b],'x',[0,1])} \)"
        foc_distractor1 = fr"\(f^\prime(x) = {PolyEq([a,     b],'x',[0,1])} \)"
        foc_distractor2 = fr"\(f^\prime(x) = {PolyEq([a+c,  -b],'x',[0,1])} \)"
        foc_distractor3 = fr"\(f^\prime(x) = {PolyEq([a,-0.5*b],'x',[0,1])} \)"

        setup_list = []
        setup = fr"""
$$f(x) = {PolyEq([a,-0.5*b,c],'x',[1,2,0])}$$
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"What is the first derivative of \(f(x)\)?"
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
        question = fr"What is the value of \(x\) that maximizes \(f(x)\)?"
        online_question = question
        answer = x
        online_answer = fr"\(x = {x:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"What is the maximum value of \(f(x)\)?"
        online_question = question
        answer = f
        online_answer = fr"\(f(x) = {f:g}\)"
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


@register
class ExponentialOptimizationProblem(GenericProblem):
    # f(x) = ax^p - bx + c
    def __init__(self, params=None, rng=None, name='exponential_optimization_problem'):
        rng = default_rng(rng)
        default_params = {'a':1,'p':0.5,'b':1,'c':0}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, p, b, c = params['a'], params['p'], params['b'], params['c']
        assert a>0 and b>0 and p>0 and p<1
        x = (b/(a*p))**(1/(p-1))
        f = a*x**p - b*x + c
        self.sol = {'x':x, 'f':f}
        
        foc =             fr"\(f^\prime(x) = {PolyEq([a*p,  -b],'x',[p-1,0])} \)"
        foc_distractor1 = fr"\(f^\prime(x) = {PolyEq([a/p,  -b],'x',[p-1,0])} \)"
        foc_distractor2 = fr"\(f^\prime(x) = {PolyEq([a,    -b],'x',[p-1,0])} \)"
        foc_distractor3 = fr"\(f^\prime(x) = {PolyEq([a,     b],'x',[p,0])} \)"

        setup_list = []
        setup = fr"""
$$f(x) = {PolyEq([a,-b,c],'x',[p,1,0])}$$
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"What is the first derivative of \(f(x)\)?"
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
        question = fr"What is the value of \(x\) that maximizes \(f(x)\)?"
        online_question = question
        answer = x
        online_answer = fr"\(x = {x:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"What is the maximum value of \(f(x)\)?"
        online_question = question
        answer = f
        online_answer = fr"\(f(x) = {f:g}\)"
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


@register
class LogOptimizationProblem(GenericProblem):
    # f(x) = aln(x) - bx + c
    def __init__(self, params=None, rng=None, name='log_optimization_problem'):
        rng = default_rng(rng)
        default_params = {'a':1,'b':1,'c':0}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b, c = params['a'], params['b'], params['c']
        assert a>0 and b>0
        x = a/b
        f = a*np.log(x) - b*x + c
        self.sol = {'x':x, 'f':f}
        
        foc =             fr"\(f^\prime(x) = \frac{{{a:.0f}}}{{x}} - {b:.0f} \)"
        foc_distractor1 = fr"\(f^\prime(x) = {PTerm(a,'x',1)} - {b:.0f} \)"
        foc_distractor2 = fr"\(f^\prime(x) = \frac{{{a:.0f}}}{{x}} + {b:.0f} \)"
        foc_distractor3 = fr"\(f^\prime(x) = \frac{{{2*a:.0f}}}{{x}} - {b:.0f} \)"

        setup_list = []
        temparg = '\\ln(x)'
        setup = fr"""
$$f(x) = {PTerm(a, temparg, 1)} - {PolyEq([b, c], 'x', [1, 0])}$$
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"What is the first derivative of \(f(x)\)?"
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
        question = fr"What is the value of \(x\) that maximizes \(f(x)\)?"
        online_question = question
        answer = x
        online_answer = fr"\(x = {x:g}\)"
        answers = generate_distractors(answer,rng=rng)
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=False,sort=True,horz=True,numerical=True,rng=rng)
        })
        question = fr"What is the maximum value of \(f(x)\)?"
        online_question = question
        answer = f
        online_answer = fr"\(f(x) = {f:g}\)"
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


@register
class LinearContourProblem(GenericProblem):
    # ax + by = z
    # y = z/b - (a/b)x
    def __init__(self, params=None, rng=None, name='linear_contour_line_problem'):
        rng = default_rng(rng)
        default_params = {'a':1,'b':1,'z':5,'x':'x','y':'y','xunit':1,'yunit':1,'xn':13,'yn':13}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b, z, x, y, xunit, yunit, xn, yn = params['a'], params['b'], params['z'], params['x'], params['y'], params['xunit'], params['yunit'], params['xn'], params['yn']
        yint = z/b
        xint = z/a
        self.sol = {'yint':yint, 'xint':xint}
        axis = Axis(xn=xn, yn=yn, xunit=xunit, yunit=yunit)
        passing = [(i*xunit, i*yunit) for i in range(1,xn)]
        contours = LineContours(a,b,passing)
        axis.add(contours)
        setup_list = []
        setup = fr"$$ f({x},{y}) = {PolyEq([a,b],[x,y],[1,1])} $$"
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Write an equation describing the contour line for \(f(x,y)=z\)."
        online_question = question
        answer = fr"\(y = {PTerm(1/b, 'z', 1)} - {PTerm(a/b, 'x', 1)}\)"
        online_answer = answer
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": None
        })
        question = fr"What is the y-intercept of the contour line for \(f(x,y)={z:g}\)?"
        online_question = question
        answer = yint
        online_answer = fr"{answer:g}"
        answers = generate_distractors(answer,delta=yunit,type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"What is the x-intercept of the contour line for \(f(x,y)={z:g}\)?"
        online_question = question
        answer = xint
        online_answer = fr"{answer:g}"
        answers = generate_distractors(answer,delta=xunit,type='add',rng=rng)
        question_list.append({
            "question": question,
            "online_question": question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,horz=True,shuffle=False,sort=True,numerical=True,rng=rng)
        })
        question = fr"""
Using the grid below, draw contour lines for \(f(x,z)=z\) for multiple values of \(z\).
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
"""
        online_question = fr"""
Using the grid below, draw contour lines for \(f(x,z)=z\) for multiple values of \(z\).
<p><img src="{get_config().image_url_prefix}/{name}_setup.png"></p>
"""
        answer = fr"""
\begin{{center}}
\includegraphics[width=3in]{{{name}_sol.png}}
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
        self.setup_list = setup_list
        self.question_list = question_list
        self.axis = axis
    def check_solution(self):
        if not is_divisible(self.sol['yint'], self.params['yunit']): return False
        if not is_divisible(self.sol['xint'], self.params['xunit']): return False
        if self.sol['yint'] >= self.params['yunit']*self.params['yn']: return False
        if self.sol['xint'] >= self.params['xunit']*self.params['xn']: return False
        return True


@register
class CBDerivativeProblem(GenericProblem):
    # f(x,y) = Ax^a y^b
    # f_x = a Ax^(a-1) y^b
    # f_y = b Ax^a y^(b-1)
    def __init__(self, params=None, rng=None, name='cb_derivative_problem'):
        rng = default_rng(rng)
        default_params = {'A':1,'a':1/2,'b':1/2}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        A, a, b = params['A'], params['a'], params['b']

        A_, a_, b_ = a*A, a-1, b
        fx    = fr"\(f_x(x,y) = {CobbDouglas(A_,'x',a_,'y',b_)} \)"
        A_, a_, b_ = A, a-1, b
        fx_d1 = fr"\(f_x(x,y) = {CobbDouglas(A_,'x',a_,'y',b_)} \)"
        A_, a_, b_ = a*b*A, a-1, b-1
        fx_d2 = fr"\(f_x(x,y) = {CobbDouglas(A_,'x',a_,'y',b_)} \)"
        A_, a_, b_ = b*A, a, b-1
        fx_d3 = fr"\(f_x(x,y) = {CobbDouglas(A_,'x',a_,'y',b_)} \)"

        A_, a_, b_ = b*A, a, b-1
        fy    = fr"\(f_y(x,y) = {CobbDouglas(A_,'x',a_,'y',b_)} \)"
        A_, a_, b_ = A, a, b-1
        fy_d1 = fr"\(f_y(x,y) = {CobbDouglas(A_,'x',a_,'y',b_)} \)"
        A_, a_, b_ = a*b*A, a-1, b-1
        fy_d2 = fr"\(f_y(x,y) = {CobbDouglas(A_,'x',a_,'y',b_)} \)"
        A_, a_, b_ = a*A, a-1, b
        fy_d3 = fr"\(f_y(x,y) = {CobbDouglas(A_,'x',a_,'y',b_)} \)"

        setup_list = []
        setup = fr"""
$$ f(x,y) = {CobbDouglas(A,'x',a,'y',b)} $$
"""
        online_setup = setup
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Write the partial derivative of \(f(x,y)\) with respect to \(x\)."
        online_question = question
        answer = fx
        online_answer = answer
        answers = [answer, fx_d1, fx_d2, fx_d3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        question = fr"Write the partial derivative of \(f(x,y)\) with respect to \(y\)."
        online_question = question
        answer = fy
        online_answer = answer
        answers = [answer, fy_d1, fy_d2, fy_d3]
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,0,shuffle=True,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list


@register
class CobbDouglasPointsProblem(GenericProblem):
    # u(x,y) = x^a y^b 
    def __init__(self, params=None, rng=None, name='cobb_douglas_points_problem'):
        rng = default_rng(rng)
        default_params = {'a':1/2, 'b':1/2}
        GenericProblem.__init__(self, params=params, default_params=default_params, rng=rng, name=name)
        params = self.params
        a, b = params['a'], params['b']
        cobb_douglas = CobbDouglas(A=1, a=a, b=b)
        setup_axis = Axis(noticklabels=True)
        levels = get_cb_levels(cobb_douglas, setup_axis)
        margin = 1 # keep markers oen grid point inside the graph edges
        xlo, xhi = margin*setup_axis.xunit, setup_axis.xmax - margin*setup_axis.xunit
        ylo, yhi = margin*setup_axis.yunit, setup_axis.ymax - margin*setup_axis.yunit
        def visible_x_range(u):
            c = u**(1/b)
            return max(xlo, (c/yhi)**(b/a)), min(xhi, (c/ylo)**(b/a))
        mids = [(levels[k] + levels[k+1])/2 for k in range(len(levels)-1)]
        candidates = [u for u in mids if visible_x_range(u)[0] < visible_x_range(u)[1]]
        chosen = rng.choice(candidates, size=4, replace=False)
        points = []
        best_u = -99
        worst_u = 100000
        best_point = 'A'
        worst_point = 'A'
        for u, label in zip(chosen, ['A', 'B', 'C', 'D']):
            x_min, x_max = visible_x_range(u)
            x = rng.uniform(x_min, x_max)
            y = u**(1/b) * x**(-a/b)
            points.append(Point(x,y,text=label,position='ne'))
            if u > best_u:
                best_u = u
                best_point = label
            if u < worst_u:
                worst_u = u
                worst_point = label
        setup_axis.add(CobbDouglasContours(cobb_douglas, levels), points[0], points[1], points[2], points[3])
        self.setup_axis = setup_axis
        self.sol = {'best_point': best_point, 'worst_point': worst_point}
        setup_list = []
        setup = fr"""
A consumer has a utility function over two goods, \(x\) and \(y\), represented by the indifference curves shown below.
\begin{{center}}
\includegraphics[width=3in]{{{name}_setup.png}}
\end{{center}}
"""
        online_setup = fr"""
A consumer has a utility function over two goods, \(x\) and \(y\), represented by the indifference curves shown below.
<p>
<img src="{get_config().image_url_prefix}/{name}_setup.png">
</p>
"""
        setup_list.append({
            "setup": setup,
            "online_setup": online_setup
        })
        question_list = []
        question = fr"Which point gives the highest utility?"
        online_question = question
        answer = best_point
        online_answer = answer
        answers = ['A', 'B', 'C', 'D']
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,answers.index(answer),horz=True,shuffle=False,sort=False,numerical=False,rng=rng)
        })
        question = fr"Which point gives the lowest utility?"
        online_question = question
        answer = worst_point
        online_answer = answer
        answers = ['A', 'B', 'C', 'D']
        question_list.append({
            "question": question,
            "online_question": online_question,
            "answer": answer,
            "online_answer": online_answer,
            "MCQ": MCQ(question,answers,answers.index(answer),horz=True,shuffle=False,sort=False,numerical=False,rng=rng)
        })
        self.setup_list = setup_list
        self.question_list = question_list
    def check_solution(self):
        return True
