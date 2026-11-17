import numpy as np
import pandas as pd
import sympy
from sympy import nsimplify
from matplotlib import pyplot as plt
from econtools2.mathutils import *
from econtools2.graphs import *


class NormalForm:
    def __init__(self, players, strategies, payoffs, gametype):
        assert len(players)==2
        assert len(strategies)==2
        assert len(strategies[0])==len(payoffs)
        assert len(payoffs[0])==len(strategies[1])
        _payoffs_ = payoffs
        # Initialize payoffs and best responses
        payoffs = {}
        payoffs[players[0]] = {}
        payoffs[players[1]] = {}
        br = {}
        br[players[0]] = {}
        br[players[1]] = {}
        for s in strategies[1]:
            payoffs[players[0]][s] = {}
            br[players[0]][s] = {}
        for s in strategies[0]:
            payoffs[players[1]][s] = {}
            br[players[1]][s] = {}

        # Populate payoffs
        for i in range(len(_payoffs_)):
            for j in range(len(_payoffs_[0])):
                strategy_1 = strategies[0][i]
                strategy_2 = strategies[1][j]
                payoffs[players[0]][strategy_2][strategy_1] = _payoffs_[i][j][0]
                payoffs[players[1]][strategy_1][strategy_2] = _payoffs_[i][j][1]

        # Find best responses
        for player, opp_strategies in payoffs.items():
            for opp_strategy, my_strategies in opp_strategies.items():
                my_payoffs = np.array([v for (k,v) in my_strategies.items()])
                #print(my_payoffs)
                #print(my_payoffs.shape)
                my_br = []
                my_br_ids = np.argwhere(my_payoffs==np.amax(my_payoffs)).flatten()
                #print(type(my_br_ids))
                #print(my_br_ids)
                #print(type(list(my_strategies.keys())))
                #print(list(my_strategies.keys()))
                for br_id in my_br_ids:
                    my_br.append(list(my_strategies.keys())[br_id] )
                br[player][opp_strategy] = my_br

        # Find Nash equilibria
        ne = []
        ne_payoffs = []
        ne_distractors = []
        for s1 in strategies[0]:
            for s2 in strategies[1]:
                if (s1 in br[players[0]][s2]) and (s2 in br[players[1]][s1]):
                    ne.append(fr"({s1},{s2})")
                    ne_payoffs.append((payoffs[players[0]][s1][s2], payoffs[players[1]][s1][s2]))
                else:
                    ne_distractors.append(fr"({s1},{s2})")
        
        self.players = players
        self.strategies = strategies
        self._payoffs_ = _payoffs_
        self.payoffs = payoffs
        self.br = br
        self.ne = ne
        self.ne_payoffs = ne_payoffs
        self.ne_distractors = ne_distractors
        self.gametype = gametype
    
    def table_as_html(self, circle_br=False):
        t = '<table border=1px align="center">'
        N = len(self.strategies[0])
        K = len(self.strategies[1])
        players = self.players
        strategies = self.strategies
        payoffs = self._payoffs_
        br = self.br
        t+=  '<tr>'
        t+=  '<td></td>'
        t+=  '<td></td>'
        t+= f'<td colspan={K} align="center">{players[1]}</td>'
        t+=  '</tr>'
        t+=  '<tr>'
        t+=  '<td></td>'
        t+=  '<td></td>'
        for k in range(K):
            t+= f'<td align="center">{strategies[1][k]}</td>'
        t+= '</tr>'
        for i in range(N):
            t+= '<tr>'
            if i==0:
                t+= f'<td rowspan={N}>{players[0]}</td>'
            t+= f'<td>{strategies[0][i]}</td>'
            for k in range(K):
                if circle_br and (strategies[0][i] in br[players[0]][strategies[1][k]]):
                    u1 = f'<span style="border-width:2px; border-style:solid; border-color:#FF0000;">{payoffs[i][k][0]}</span>'
                else:
                    u1 = f'{payoffs[i][k][0]}'
                if circle_br and (strategies[1][k] in br[players[1]][strategies[0][i]]):
                    u2 = f'<span style="border-width:2px; border-style:solid; border-color:#FF0000;">{payoffs[i][k][1]}</span>'
                else:
                    u2 = f'{payoffs[i][k][1]}'
                t+= f'<td align="center">{u1}, {u2}</td>'
            t+=  '</tr>'
        t+=  '</table>'
        return t

    def table_as_latex(self):
        players = self.players
        strategies = self.strategies
        payoffs = self._payoffs_
        N = len(strategies[0])
        K = len(strategies[1])
        t = fr"""
\begin{{center}}
\begin{{tabular}}{{|c|c|{'c|'*K}}} \hline
 & & \multicolumn{{{K}}}{{c|}}{{ {players[1]} }} \\ \hline
"""
        t+= " & "
        for k in range(K):
            t+= fr" & {strategies[1][k]}"
        t+= fr"\\ \hline" + '\n'
        for i in range(N):
            if i==0:
                t+= fr"\multirow{{{N}}}{{*}}{{{players[0]}}} "
            t+= fr" & {strategies[0][i]} "
            for k in range(K):
                t+= fr" & {payoffs[i][k][0]}, {payoffs[i][k][1]} "
            if i==N-1:
                t+= fr"\\ \hline" + '\n'
            else:
                t+= fr"\\ \cline{{2-{K+2}}}" + '\n'
        t+=fr"""
\end{{tabular}}
\end{{center}}
"""
        return t
