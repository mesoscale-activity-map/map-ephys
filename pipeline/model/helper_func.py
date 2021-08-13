# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:00:16 2020

@author: Han
"""
import numpy as np
import seaborn as sns
import matplotlib

def softmax(x, softmax_temperature, bias = 0):
    
    # Put the bias outside /sigma to make it comparable across different softmax_temperatures.
    if len(x.shape) == 1:
        X = x/softmax_temperature + bias   # Backward compatibility
    else:
        X = np.sum(x/softmax_temperature, axis=0) + bias  # Allow more than one kernels (e.g., choice kernel)
    
    max_temp = np.max(X)
    
    if max_temp > 700: # To prevent explosion of EXP
        greedy = np.zeros(len(x))
        greedy[np.random.choice(np.where(X == np.max(X))[0])] = 1
        return greedy
    else:   # Normal softmax
        return np.exp(X)/np.sum(np.exp(X))  # Accept np.
    
def choose_ps(ps):
    '''
    "Poisson"-choice process
    '''
    ps = ps/np.sum(ps)
    return np.max(np.argwhere(np.hstack([-1e-16, np.cumsum(ps)]) < np.random.rand()))

def seaborn_style():
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper", font_scale=1.4)
    # sns.set(style="ticks", context="talk", font_scale=2)
    sns.despine(trim=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
def moving_average(a, n=3) :
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n