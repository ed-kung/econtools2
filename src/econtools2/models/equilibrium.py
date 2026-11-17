import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils import *
from econtools2.graphs import *
from econtools2.models.markets import *
from econtools2.models.consumers import *
from econtools2.models.firms import *


class GeneralEquilibrium:
    # f(L) = A*L^kf
    # u(L) = w*L - d*L^kw
    # u(q) = a*ln(q) - p*q
    # Commodity Demand: p = a*q^-1
    # Labor Demand: w = p*A*kf*L^(kf-1)
    # Labor Supply: w = d*kw*L^(kw-1)
    # Equilibrium Labor: L = (a*kf/d*kw)^(1/kw)
    # Equilibrium Wage: a*kf*L^-1
    def __init__(self, consumer, firm, worker):
        assert type(consumer)==LogConsumer
        assert type(firm)==ExponentialProductionFirm
        assert type(worker)==Worker
        self.consumer = consumer
        self.firm = firm
        self.worker = worker
        A = firm.A
        kf = firm.k
        d = worker.d
        kw = worker.k
        a = consumer.a
        L = (a*kf/(d*kw))**(1/kw)
        w = a*kf/L
        q = A*L**kf
        p = a/q
        U_consumer = consumer.utility_at(p,q)
        U_worker = worker.utility_at(w,L)
        profit = firm.profit_at(p, w, L)
        assert equals(consumer.demand.eval_at_p(p), q)
        assert equals(worker.supply.eval_at_p(w), L)
        assert equals(q, A*L**kf)
        self.eq = {'L':L, 'w':w, 'p':p, 'q':q, 'U_consumer':U_consumer, 'U_worker':U_worker, 'profit':profit}
