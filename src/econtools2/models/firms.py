import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils import *
from econtools2.graphs import *
from econtools2.models.markets import *


class QuadraticCostFirm:
    # c(q) = aq + 0.5*bq^2
    # inv.supply: p = a + bq
    def __init__(self, a=0, b=1):
        assert a>0 or b>0
        supply = LinearSupply(a=a,b=b)
        self.a, self.b = a, b
        self.supply = supply
    def print_cost_function(self, q='q'):
        a, b = self.a, self.b
        return fr"{PolyEq([a,0.5*b],q,[1,2])}"
    def setup(self):
        return fr"""
A representative, price-taking firm decides how many units, \(q\), of a commodity to produce and sell at unit price \(p\). The
firm's total cost function for producing \(q\) units is:
$$ c(q) = {self.print_cost_function()} $$
"""
    def profit_at(self,p,q):
        a, b = self.a, self.b
        return p*q - a*q - 0.5*b*q**2


class ExponentialProductionFirm:
    # f(L) = AL^k 
    # Pi = pAL^k - wL
    # c(q) = (1/A)^(1/k) wq^(1/k)
    # w = p*k*A*L^(k-1)
    def __init__(self, A=1, k=1/2):
        self.A, self.k = A, k
    def print_production_func(self, L='L'):
        A, k = self.A, self.k
        return fr"{PolyEq([A],L,[k])}"
    def setup(self):
        return fr"""
A representative, price-taking firm uses labor to produce and sell a commodity at unit price \(p\). The firm hires labor at a constant wage rate \(w\). If the firm employs \(L\) units of labor, it can produce \(f(L)\) units of commodity output, where:

$$ f(L) = {self.print_production_func()} $$
"""
    def profit_at(self, p, w, L):
        A, k = self.A, self.k
        return p*A*L**k - w*L
    def get_labor_demand(self, p):
        A, k = self.A, self.k
        return ExponentialDemand(p*k*A, k-1)
    def get_commodity_supply(self, p):
        A, k = self.A, self.k
        assert equals(k,1/2)
        b = 2*(1/A)**(1/k)
        return QuadraticCostFirm(0,b).supply


class CobbDouglasFirm:
    # f(L,K) = A*L^a K^(1-a)
    def __init__(self, A, a, color='black', linewidth=2, label='_nolegend_', linestyle='solid', alpha=1.0):
        self.cb = CobbDouglas(A=A, a=a, b=1-a, x='L', y='K')
        self.color = color
        self.linewidth = linewidth
        self.label = label
        self.linestyle = linestyle
        self.alpha = alpha
    def print(self, maxdenom=8, rmplus=True, rmneg=False):
        return self.cb.print(maxdenom=maxdenom,rmplus=rmplus,rmneg=rmneg)
    def eval_at(self, L, K):
        return cb.eval_at(L, K)
    def get_isoquant(self, Q, xg):
        return self.cb.get_IC(Q, xg)
    def get_isoquant_from_point(self, L, K, xg):
        return self.cb.get_IC_from_point(L, K, xg)
    def __repr__(self):
        return self.print()
    def unit_cost_K(self, w, r):
        A, a = self.cb.A, self.cb.a
        return (1/A)*(((1-a)/a)**a)*(w/r)**a
    def unit_cost_L(self, w, r):
        A, a = self.cb.A, self.cb.a
        return (1/A)*((a/(1-a)))**(1-a)*(r/w)**(1-a)
    def unit_cost(self, w, r):
        K = self.unit_cost_K(w,r)
        L = self.unit_cost_L(w,r)
        return w*L + r*K
    def get_unit_isoquant(self, xg):
        return self.get_isoquant(1, xg)
    def plot(self, ax, xg):  # draw unit isoquant
        ic = self.get_unit_isoquant(xg[1:])
        ax.plot(xg[1:], ic, color=self.color, linewidth=self.linewidth, label=self.label, linestyle=self.linestyle, alpha=self.alpha)
        return ax


def get_cb_firm_from_point(L, K, w, r):
    a = w*L/(w*L + r*K)
    b = 1-a
    A = 1/(L**a*K**(1-a))
    return CobbDouglasFirm(A=A, a=a)
