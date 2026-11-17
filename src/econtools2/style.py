"""Opt-in matplotlib style used for course graphs. Nothing here runs on import."""
from matplotlib import pyplot as plt

SMALL_SIZE = 14
MEDIUM_SIZE = 16
LARGE_SIZE = 18
HUGE_SIZE = 20


def use_econ_style(usetex=True):
    plt.rcdefaults()
    plt.rcParams['text.usetex'] = usetex
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = MEDIUM_SIZE
    plt.rcParams['axes.titlesize'] = LARGE_SIZE
    plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
    plt.rcParams['xtick.labelsize'] = MEDIUM_SIZE
    plt.rcParams['ytick.labelsize'] = MEDIUM_SIZE
    plt.rcParams['legend.fontsize'] = MEDIUM_SIZE
    plt.rcParams['figure.titlesize'] = LARGE_SIZE
    plt.rcParams['figure.figsize'] = [7.2, 7.2]
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['figure.facecolor'] = (1.0, 1.0, 1.0, 0.0)
