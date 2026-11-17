import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils import *
from econtools2.graphs import *
from econtools2.models.markets import *


class Monopoly:
    # c(q) = f + aq + b*q^2
    # demand is of class LinearDemand
    def __init__(self, demand, f=0, a=0, b=0.5):
        q = (demand.a - a)/(2*(demand.b + b))
        p = demand.a - demand.b*q
        profit = p*q - f - a*q - b*q**2
        CS = 0.5*(demand.a - p)*q
        q_eff = (demand.a - a)/(demand.b + 2*b)
        p_eff = demand.a - demand.b*q_eff
        profit_eff = p_eff*q_eff - f - a*q_eff - b*q_eff**2
        CS_eff = 0.5*(demand.a - p_eff)*q_eff
        DWL = (CS_eff + profit_eff) - (CS + profit)
        self.demand = demand
        self.f = f
        self.a = a
        self.b = b
        self.sol = {'q':q, 'p':p, 'CS':CS, 'profit':profit, 
                    'q_eff': q_eff, 'p_eff': p_eff, 'CS_eff':CS_eff, 'profit_eff':profit_eff,
                    'DWL':DWL}
    def print_cost_function(self, q='q'):
        f, a, b = self.f, self.a, self.b
        return fr"{PolyEq([f,a,b],q,[0,1,2])}"
    def setup(self):
        return fr"""
The market is supplied by a single monopolist, who can produce \(q\) units of the commodity at a total cost of:
$$ c(q) = {self.print_cost_function()} $$
"""


class PriceDiscrimination:
    # c(q) = cq
    # demand1 and demand2 are of class LinearDemand
    def __init__(self, demand1, demand2, c=1):
        monopoly1 = Monopoly(demand1, f=0, a=c, b=0)
        monopoly2 = Monopoly(demand2, f=0, a=c, b=0)
        a1 = demand1.a
        b1 = demand1.b
        a2 = demand2.a
        b2 = demand2.b
        gamma = (a1/b1 + a2/b2)/(1/b1+1/b2)
        delta = 1/(1/b1 + 1/b2)
        demand_both = LinearDemand(a=gamma, b=delta)
        monopoly_both = Monopoly(demand_both, f=0, a=c, b=0)
        self.demand1 = demand1
        self.demand2 = demand2
        self.c = c
        self.monopoly1 = monopoly1
        self.monopoly2 = monopoly2
        self.demand_both = demand_both
        self.monopoly_both = monopoly_both
    def setup(self):
        return fr"""
The market is supplied by a single monopolist, who can practice third degree price discrimination. The monopoly can produce \(q\) units of the commodity at a total cost of:
$$ c(q) = {self.monopoly1.print_cost_function()} $$
"""


class Cournot2:
    def __init__(self, firm1, firm2, demand):
        a1 = firm1.a
        b1 = firm1.b
        a2 = firm2.a
        b2 = firm2.b
        alpha = demand.a
        beta = demand.b
        M = np.array([
            [2*beta+b1,      beta],
            [     beta, 2*beta+b2]
        ])
        y = np.array([alpha-a1, alpha-a2])
        x = np.linalg.solve(M, y)
        q1 = x[0]
        q2 = x[1]
        Q = q1 + q2
        p = alpha - beta*Q
        profit1 = firm1.profit_at(p, q1)
        profit2 = firm2.profit_at(p, q2)
        self.sol = {'p':p, 'q1':q1, 'q2':q2, 'Q':Q, 'profit1':profit1, 'profit2':profit2}
        self.firm1 = firm1
        self.firm2 = firm2
        self.demand = demand
    def setup(self):
        return fr"""
Price-taking consumers in the market for a commodity have a demand curve given by:
$$ Q = {self.demand.print()} $$
The market is supplied by two firms who produce identical products. Firm 1 has a cost function given by:
$$ c_1(q_1) = {self.firm1.print_cost_function(q='q_1')} $$
Firm 2 has a cost function given by:
$$ c_2(q_2) = {self.firm2.print_cost_function(q='q_2')} $$
The firms engage in Cournot competition, i.e. they choose the quantity they wish to produce and let the market determine the price.
"""


class CournotN:
    def __init__(self, mc, N, demand):
        alpha = demand.a
        beta = demand.b
        q = (alpha - mc)/((N+1)*beta)
        Q = N*q
        p = alpha - beta*Q
        profit = p*q - mc*q
        total_profit = N*profit
        self.N = N
        self.mc = mc
        self.demand = demand
        self.sol = {'q':q, 'Q':Q, 'p':p, 'profit':profit, 'total_profit':total_profit}
    def setup(self):
        return fr"""
Price-taking consumers in the market for a commodity have a demand curve given by:
$$ Q = {self.demand.print()} $$
The market is supplied by \(N={self.N:g}\) identical firms. Each firm has a constant average and marginal cost of production equal to \(c={self.mc:g}\). The firms engage in Cournot competition, i.e. they choose the quantity they wish to produce and let the market determine the price.
"""
