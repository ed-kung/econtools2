import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils.numbers import *
from econtools2.mathutils.formatting import *


class CobbDouglas:
    def __init__(self, A=1, x='x', a=1/2, y='y', b=1/2):
        self.A, self.x, self.a, self.y, self.b = A, x, a, y, b
        self.bad=False
    def print(self, maxdenom=8, rmplus=True, rmneg=False):
        out = PTerm(self.A, self.x, self.a).print(maxdenom=maxdenom,rmplus=rmplus,rmneg=rmneg)
        out+= PTerm(1, self.y, self.b).print(maxdenom=maxdenom,rmplus=True,rmneg=True)
        return out
    def eval_at(self, x, y):
        A, a, b = self.A, self.a, self.b
        return A*(x**a)*(y**b)
    def get_IC(self, U, xg):
        A, a, b = self.A, self.a, self.b
        return (U/A)**(1/b)*xg**(-a/b)
    def get_IC_from_point(self, x, y, xg):
        U = self.eval_at(x, y)
        return self.get_IC(U, xg)
    def __repr__(self):
        return self.print()


class CES:
    def __init__(self, a=0.5, rho=0.5):
        self.a = a
        self.rho = rho
        self.bad = False
        if self.a<=0.1 or self.a>=0.9:
            self.bad = True
        if self.rho>=0.8:
            self.bad = True
        if self.rho<-2:
            self.bad = True
        if np.abs(self.rho)<=0.1:
            self.bad = True
    def eval_at(self, x, y):
        a, rho = self.a, self.rho
        return (a*x**rho + (1-a)*y**rho)**(1/rho)
    def get_IC(self, U, xg):
        a, rho = self.a, self.rho
        numerator = U**rho - a*xg**rho
        ids = (numerator>0)
        return xg[ids], (numerator[ids]/(1-a))**(1/rho)
    def get_IC_from_point(self, x, y, xg):
        U = self.eval_at(x,y)
        return self.get_IC(U, xg)


def get_cb_from_point(x, y, budget_constraint, x_='x', y_='y'):
    px, py, I = budget_constraint.px, budget_constraint.py, budget_constraint.I
    assert equals(x*px + y*py, I)
    a = px*x/I
    b = py*y/I
    return CobbDouglas(A=1,a=a,b=b,x=x_,y=y_)


def get_ces_from_points(x1,y1,x2,y2,bc1,bc2):
    px1, py1, I1 = bc1.px, bc1.py, bc1.I
    px2, py2, I2 = bc2.px, bc2.py, bc2.I
    assert equals(px1*x1 + py1*y1, I1)
    assert equals(px2*x2 + py2*y2, I2)
    A1 = np.log((I1-px1*x1)/(py1*x1))
    A2 = np.log((I2-px2*x2)/(py2*x2))
    B1 = np.log(py1/px1)
    B2 = np.log(py2/px2)
    try:
        M = np.array([
            [A1, 1],
            [A2, 1]
        ])
        Y = np.array([B1, B2])
        X = np.linalg.solve(M,Y)
    except:
        return CES(rho=2,a=0)
    rho = X[0]+1
    a = 1/(1+np.exp(X[1]))
    if equals(rho, 0):
        return CobbDouglas(A=1, a=a, b=1-a)
    else:
        return CES(rho=rho, a=a)


def simplifyCB(cbtop, cbbot):
    assert cbtop.a!=0
    assert cbtop.b!=0
    assert equals( np.abs(cbtop.a)+np.abs(cbbot.a), 1)
    assert equals( np.abs(cbtop.b)+np.abs(cbbot.b), 1)
    assert sign(cbtop.a)!=sign(cbtop.b)
    assert cbtop.x==cbbot.x
    assert cbtop.y==cbbot.y
    assert is_rational(cbtop.A/cbbot.A, maxdenom=cbbot.A)
    x = cbtop.x
    y = cbtop.y
    a = cbtop.a
    n = nsimplify(cbtop.A/cbbot.A).p
    d = nsimplify(cbtop.A/cbbot.A).q
    if a>0:
        return fr"\(\frac{{{PTerm(n,x,1).print()}}}{{{PTerm(d,y,1).print()}}}\)"
    else:
        return fr"\(\frac{{{PTerm(n,y,1).print()}}}{{{PTerm(d,x,1).print()}}}\)"
