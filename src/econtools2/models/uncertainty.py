import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils import *
from econtools2.graphs import *


class Insurance:
    def __init__(self, W0=1000, D=200, p=0.2, fun='ln'):
        self.W0 = W0
        self.D = D
        self.p = p
        self.fun = fun
        if fun=='ln':
            ufun = np.log
            uinv = np.exp
        elif fun=='sqrt':
            ufun = np.sqrt
            uinv = lambda x: x**2
        EX = (1-p)*W0 + p*(W0 - D)
        EU = (1-p)*ufun(W0) + p*ufun(W0 - D)
        CE = uinv(EU)
        WTP = W0 - CE
        FairCost = p*D
        self.sol = {'EX':EX, 'EU':EU, 'CE':CE, 'WTP':WTP, 'FairCost':FairCost}
    def setup(self):
        W0, D, p, fun = self.W0, self.D, self.p, self.fun
        if fun=='ln':
            funstr = fr"\ln X"
        elif fun=='sqrt':
            funstr = fr"\sqrt{{X}}"
        return fr"""
An individual has an initial wealth of \({W0:,g}\). There is a \({p*100:g}\%\) chance that the individual has an accident which would force them to pay \({D:,g}\) to fix the problem. Let \(X\) be a random variable representing the person's wealth at the end of the day. The individual's utility function over wealth is:
$$ u(X) = {funstr} $$
"""


class Savings:
    def __init__(self, Y, beta, p):
        self.Y = Y
        self.beta = beta
        self.p = p
        c2 = Y/(p*(1+1/beta))
        c1 = Y - p*c2
        r = (1-p)/p
        self.sol = {'c1':c1, 'c2':c2, 'r':r}
    def setup(self):
        Y, beta, p = self.Y, self.beta, self.p
        return fr"""
An individual lives for two periods. In period 1, they earn an income of \(Y={Y:,g}\). In period 2, they earn no income. In order to consume in period 2, they must buy bonds in period 1. One bond pays \(\$1\) in period 2, and can be purchased for price \(p={p:g}\) in period 1. The individual's objective is to maximize the present value of their utility:
$$\ln c_1 + \beta \ln c_2$$
where \(c_1\) is consumption period 1, \(c_2\) is consumption in period 2, and \(\beta={beta:g}\) is the individual's subjective time discount factor.
"""
