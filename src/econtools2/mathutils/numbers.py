import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt


def sign(x):
    if x>0: return '+'
    else: return '-'


def equals(x,y,tol=1e-4):
    return np.abs(x-y)<tol


def is_rational(x, maxdenom=8):
    try:
        d = nsimplify(x).q
        return (d<=maxdenom)
    except:
        return False


def is_divisible(p,q):
    if not is_rational(p/q,30):
        return False
    d = nsimplify(p/q).q
    return (d==1)
