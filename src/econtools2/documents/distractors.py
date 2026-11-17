import numpy as np

from econtools2._rng import default_rng


def generate_distractors(x, K=4, delta=None, type='add', rng=None, must_positive=False):
    rng = default_rng(rng)
    if not delta:
        if type=='add':
            if np.abs(x)>=10: delta = np.round(0.1*np.abs(x))
            elif np.abs(x)>=4: delta = 1
            elif np.abs(x)>=2: delta = 0.5
            elif np.abs(x)>=1: delta = 0.1
            elif np.abs(x)>0: delta = 0.1*np.abs(x)
            else: delta = 1
        else:
            delta = 2
    
    max_tries = 100
    for i in range(max_tries):
        nless = rng.integers(K)
        nmore = K - nless - 1
        answers = [x]
        for i in range(nless):
            if type=='mul':
                answers.append(x*delta**(-i-1))
            if type=='add':
                answers.append(x - delta*(i+1))
        for i in range(nmore):
            if type=='mul':
                answers.append(x*delta**(i+1))
            if type=='add':
                answers.append(x + delta*(i+1))
        
        any_negative = False
        for ans in answers:
            if ans<0:
                any_negative = True
        if must_positive and any_negative:
            continue
        else:
            return answers
    raise()
