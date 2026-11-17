import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils import *
from econtools2.graphs import *
from econtools2.models.markets import *


class LinearConsumer:
    # u(q) = aq - 0.5bq^2 - pq
    # inv.demand: p = a - bq
    def __init__(self, a=12, b=1):
        demand = LinearDemand(a=a,b=b)
        self.a, self.b = a, b
        self.demand = demand
    def print_utility(self, q='q'):
        a, b = self.a, self.b
        return fr"{PolyEq([a,-0.5*b],q,[1,2])} - p{q}"
    def setup(self):
        return fr"""
A representative, price-taking consumer decides how many units, \(q\), of a commodity to purchase at unit price \(p\). The utility
they receive for purchasing \(q\) units at price \(p\) is:
$$ u(q) = {self.print_utility()} $$
"""
    def utility_at(self, p,q):
        a, b = self.a, self.b
        return a*q - 0.5*b*q**2 - p*q


class LogConsumer:
    # u(q) = a ln(q) - pq
    # inv.demand: p = a - bq
    def __init__(self, a=1):
        demand = ExponentialDemand(a=a,k=-1)
        self.a = a
        self.demand = demand
    def print_utility(self, q='q'):
        a = self.a
        lnq = PTerm(a,fr'\ln {q}',1)
        return fr"{lnq} - p{q}"
    def setup(self):
        return fr"""
A representative, price-taking consumer decides how many units, \(q\), of a commodity to purchase at unit price \(p\). The utility
they receive for purchasing \(q\) units at price \(p\) is:
$$ u(q) = {self.print_utility()} $$
"""
    def utility_at(self, p,q):
        a = self.a
        return a*np.log(q) - p*q


class Worker:
    # u(L) = wL - d*L^k
    def __init__(self, d=0.5, k=2):
        self.d, self.k = d, k
        supply = ExponentialSupply(a=k*d, k=k-1)
        self.supply = supply
    def print_objective(self, w='w', L='L'):
        d, k = self.d, self.k
        wL = f'{w}{L}'
        return fr"{PolyEq([1,-d],[wL,L],[1,k])}"
    def setup(self):
        return fr"""
A representative, price-taking worker decides how many units, \(L\), of labor to supply (e.g. how many hours to work), at a unit wage \(w\).
The worker's utility function over working \(L\) labor-units at wage \(w\) is:

$$ u(L) = {self.print_objective()} $$
"""
    def utility_at(self, w, L):
        d, k = self.d, self.k
        return w*L - d*L**k


class CobbDouglasConsumer:
    # u(x,y) = x^a y^b
    # px*x + py*y = I
    def __init__(self, cobb_douglas, budget_constraint):
        assert type(cobb_douglas)==CobbDouglas
        assert type(budget_constraint)==BudgetConstraint
        A, a, b = cobb_douglas.A, cobb_douglas.a, cobb_douglas.b 
        px, py, I = budget_constraint.px, budget_constraint.py, budget_constraint.I
        x = I/(px*(1+b/a))
        y = I/(py*(1+a/b))
        U = x**a * y**b
        self.x = x
        self.y = y
        self.U = U


class IncomeSupportBudget:
    def __init__(self, bc, min_y, color='black'):
        self.bc = bc
        self.min_y = min_y
        self.color = color
    def plot(self, ax, xg):
        m = - self.bc.px / self.bc.py
        b = self.bc.I / self.bc.py
        bc = m*xg + b
        ax.plot(xg, np.maximum(bc, self.min_y), color=self.color, linewidth=self.bc.linewidth, alpha=self.bc.alpha, label=self.bc.label)
        return ax
