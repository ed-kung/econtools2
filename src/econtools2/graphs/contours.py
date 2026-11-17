import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils import *
from econtools2.graphs.axis import *
from econtools2.graphs.primitives import *


class LineContours:
    # ax + by = z
    # y = -(a/b)*x + (z/b)
    def __init__(self, a, b, passing, linewidth=1, color='green', alpha=0.5):
        self.linewidth = linewidth
        self.color = color
        self.alpha = alpha
        contours = []
        for i in range(len(passing)):
            x = passing[i][0]
            y = passing[i][1]
            z = a*x + b*y
            contours.append(Line(-a/b, z/b))
        self.contours = contours
    def plot(self, ax, xg):
        for contour in self.contours:
            m = contour.m
            b = contour.b
            ax.plot(xg, m*xg+b, color=self.color, linewidth=self.linewidth, alpha=self.alpha, label='_nolegend_')
        return ax


class BudgetConstraint:
    def __init__(self, px, py, I, linewidth=2, color='black', alpha=1.0, label='_nolegend_'):
        self.px, self.py, self.I = px, py, I
        self.xint = I/px
        self.yint = I/py
        self.linewidth = linewidth
        self.color = color
        self.alpha = alpha
        self.label = label
    def plot(self, ax, xg):
        m = -self.px / self.py
        b = self.I / self.py
        ax.plot(xg, m*xg+b, color=self.color, linewidth=self.linewidth, alpha=self.alpha, label=self.label)
        return ax


class CobbDouglasContours:
    # A x^a y^b
    # y = (z/A)^(1/b) * x^(-a/b)
    def __init__(self, cobb_douglas, levels, linewidth=1, color='green', alpha=0.5):
        assert type(cobb_douglas)==CobbDouglas
        self.linewidth = linewidth
        self.color = color
        self.alpha = alpha
        self.levels = levels
        self.cobb_douglas = cobb_douglas
    def plot(self, ax, xg):
        A, a, b = self.cobb_douglas.A, self.cobb_douglas.a, self.cobb_douglas.b
        for z in self.levels:
            ax.plot(xg[1:], (z/A)**(1/b)*xg[1:]**(-a/b), color=self.color, linewidth=self.linewidth, alpha=self.alpha, label='_nolegend_')
        return ax


class CESContours:
    def __init__(self, x1, y1, x2, y2, bc1, bc2, axis, color='green', alpha=0.5, linewidth=1):
        self.color = color
        self.alpha = alpha
        self.linewidth = linewidth
        xn, yn, xunit, yunit, xmax, ymax, xg = axis.xn, axis.yn, axis.xunit, axis.yunit, axis.xmax, axis.ymax, axis.xg
        ces = get_ces_from_points(x1,y1,x2,y2,bc1,bc2)
        self.ces = ces
        if not self.ces.bad:
            U1 = ces.eval_at(x1, y1)
            U2 = ces.eval_at(x2, y2)
            IC1 = ces.get_IC_from_point(x1,y1,xg[1:])
            IC2 = ces.get_IC_from_point(x2,y2,xg[1:])
            Umax = ces.eval_at((xn+1)*xunit, (yn+1)*yunit)
            Umin = ces.eval_at(xunit, yunit)
            du_ideal = (Umax - Umin)/(xn-2)
            du = np.abs(U2-U1)/np.maximum(np.round(np.abs(U2-U1)/du_ideal),1)
            levels = [U1+i*du for i in np.arange(-xn,xn) if U1+i*du>=Umin and U1+i*du<=Umax]
            self.levels = levels
    def plot(self, ax, xg):
        if not self.ces.bad:
            for u in self.levels:
                if type(self.ces)==CES:
                    myxg, ic = self.ces.get_IC(u, xg[1:])
                else: 
                    myxg = xg[1:]
                    ic = self.ces.get_IC(u, xg[1:])
                ax.plot(myxg, ic, color=self.color, alpha=self.alpha, linewidth=self.linewidth, label='_nolegend_')
        return ax


def get_cb_levels(cobb_douglas, axis, U1=None, U2=None):
    assert type(cobb_douglas)==CobbDouglas
    assert type(axis)==Axis
    xn = axis.xn
    yn = axis.yn
    xunit = axis.xunit
    yunit = axis.yunit
    umax = cobb_douglas.eval_at((xn)*xunit, (yn)*yunit)
    umin = cobb_douglas.eval_at(xunit, yunit)
    du_ideal = (umax - umin)/(xn-2)
    if U1 is None:
        return [umin+i*du_ideal for i in range(0,xn-1)]
    if U2 is None:
        U2 = umin
    if equals(U2,U1):
        U2 = umin
    du = np.abs(U2-U1)/np.maximum(np.round(np.abs(U2-U1)/du_ideal),1)
    levels = [U1+i*du for i in np.arange(-xn,xn) if U1+i*du>=umin and U1+i*du<=umax]
    return levels


class LeontieffContours:
    def __init__(self, xunit=1, yunit=1, xn=13, yn=13, linewidth=1, color='green', alpha=0.5):
        self.xunit = xunit
        self.xn = xn
        self.yunit = yunit
        self.yn = yn
        self.linewidth = linewidth
        self.color = color
        self.alpha = alpha
    def plot(self, ax, xg):
        for i in np.arange(1, self.xn):
            xmax = (self.xn+1)*self.xunit
            ymax = (self.yn+1)*self.yunit
            interval = xg[1]-xg[0]
            x = i*self.xunit
            y = i*self.yunit
            myxg = np.arange(x,xmax,interval)
            myyg = np.arange(y,ymax,interval)
            ax.plot(myxg, [y]*len(myxg), linewidth=self.linewidth, color=self.color, alpha=self.alpha, label='_nolegend_')
            ax.plot([x]*len(myyg), myyg, linewidth=self.linewidth, color=self.color, alpha=self.alpha, label='_nolegend_')
        return ax
