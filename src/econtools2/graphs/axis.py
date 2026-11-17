import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils import *


class Axis:
    def __init__(self, xn=13, yn=13, xunit=1, yunit=1, xlab=r'$x$', ylab=r'$y$', noticklabels=False, title=None):
        self.xn = xn
        self.yn = yn
        self.xunit = xunit
        self.yunit = yunit
        self.xmax = xn*xunit
        self.ymax = yn*yunit
        self.objects = []
        self.xg = np.arange(0, self.xmax, self.xunit/10)
        self.xlab = xlab
        self.ylab = ylab
        self.noticklabels = noticklabels
        self.title = title
    def add(self, *args):
        for obj in args:
            self.objects.append(obj)
    def get_figax(self, alpha=0.6, saveas=None):
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(0, self.xmax, self.xunit))
        ax.set_yticks(np.arange(0, self.ymax, self.yunit))
        if self.noticklabels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.set_xlim([0, self.xmax])
        ax.set_ylim([0, self.ymax])
        ax.set_xlabel(self.xlab)
        ax.set_ylabel(self.ylab)
        ax.set_axisbelow(True)
        ax.grid(alpha=alpha)
        if self.title is not None:
            ax.set_title(self.title)
        if saveas is not None:
            plt.savefig(saveas, bbox_inches='tight')
        return fig, ax
    def draw(self, alpha=0.6, legend=None, saveas=None):
        fig, ax = self.get_figax(alpha=alpha)
        for obj in self.objects:
            xg = np.arange(0, self.xmax, self.xunit/10)
            obj.plot(ax, xg)
        if legend:
            ax.legend()
        if saveas is not None:
            plt.savefig(saveas, bbox_inches='tight')
        return fig, ax
    def on_grid(self,x,y):
        x_on_grid = is_divisible(x,self.xunit) and (x<self.xmax)
        y_on_grid = is_divisible(y,self.yunit) and (y<self.ymax)
        return x_on_grid and y_on_grid
