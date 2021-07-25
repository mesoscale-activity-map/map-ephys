# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:47:05 2020

@author: Han
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from statannot import add_stat_annotation
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from pipeline import lab, foraging_model

plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})
sns.set(context='talk')

def plot_model_comparison_result(sess_key, model_comparison_idx=0):
    sns.set()

    # -- Fetch data --
    # Get all relevant models
    q_model_comparison = (foraging_model.FittedSessionModelComparison.RelativeStat
                          & sess_key & {'model_comparison_idx': model_comparison_idx})
    q_result = (q_model_comparison
                * foraging_model.Model
                * foraging_model.FittedSessionModel)
    best_aic_id, best_bic_id = (foraging_model.FittedSessionModelComparison.BestModel & q_model_comparison).fetch1(
        'best_aic', 'best_bic')

    # Add fitted params
    q_result *= q_result.aggr(foraging_model.FittedSessionModel.FittedParam * foraging_model.ModelParam,
                             fitted_param='GROUP_CONCAT(ROUND(fitted_value,2))')
    results = pd.DataFrame(q_result.fetch())
    results['para_notation_with_best_fit'] = [f'{name}\n{value}' for name, value in results[['model_notation','fitted_param']].values]

    # -- Plotting --
    fig = plt.figure(figsize=(15, 8), dpi = 150)
    gs = GridSpec(1, 5, wspace = 0.1, bottom = 0.1, top = 0.85, left = 0.23, right = 0.95)
    fig.text(0.05, 0.9, f'{(lab.WaterRestriction & sess_key).fetch1("water_restriction_number")}, '
                        f'session {sess_key["session"]}, {results.n_trials[0]} trials\n'
                        f'Model comparison: {(foraging_model.ModelComparison & q_model_comparison).fetch1("desc")}'
                        f' (n = {len(results)})')

    # -- 1. LPT -- 
    ax = fig.add_subplot(gs[0, 0])
    s = sns.barplot(x = 'lpt', y = 'para_notation_with_best_fit', data = results, color = 'grey')
    s.set_xlim(min(0.5,np.min(np.min(results[['lpt_aic', 'lpt_bic']]))) - 0.005)
    plt.axvline(0.5, color='k', linestyle='--')
    s.set_ylabel('')
    s.set_xlabel('Likelihood per trial')

    # -- 2. aic, bic raw --
    ax = fig.add_subplot(gs[0, 1])
    df = pd.melt(results[['para_notation_with_best_fit', 'aic', 'bic']],
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'ic')
    s = sns.barplot(x = 'ic', y = 'para_notation_with_best_fit', hue = '', data = df)

    # annotation
    x_max = max(plt.xlim())
    ylim = plt.ylim()
    plt.plot(x_max, results.index[results.model_id == best_aic_id] - 0.2, '*', markersize = 15)
    plt.plot(x_max, results.index[results.model_id == best_bic_id] + 0.2, '*', markersize = 15)
    plt.ylim(ylim)
    
    s.set_yticklabels('')
    s.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol = 1)
    s.set_ylabel('')
    s.set_xlabel('AIC or BIC')

    # -- 3. log10_bayesfactor --
    ax = fig.add_subplot(gs[0, 2])
    df = pd.melt(results[['para_notation_with_best_fit','log10_bf_aic','log10_bf_bic']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'log10 (bayes factor)')
    s = sns.barplot(x = 'log10 (bayes factor)', y = 'para_notation_with_best_fit', hue = '', data = df)
    h_d = plt.axvline(-2, color='r', linestyle='--', label = 'decisive')
    s.legend(handles = [h_d,], bbox_to_anchor=(0,1.02,1,0.2), loc='lower left')
    # s.invert_xaxis()
    s.set_xlabel(r'log$_{10}\frac{p(model)}{p(best\,model)}$')
    s.set_ylabel('')
    s.set_yticklabels('')
    
    # -- 4. model weight --
    ax = fig.add_subplot(gs[0, 3])
    df = pd.melt(results[['para_notation_with_best_fit','model_weight_aic','model_weight_bic']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'model weight')
    s = sns.barplot(x = 'model weight', y = 'para_notation_with_best_fit', hue = '', data = df)
    ax.legend_.remove()
    plt.xlim([0,1.05])
    plt.axvline(1, color='k', linestyle='--')
    s.set_ylabel('')
    s.set_yticklabels('')
    
    # -- 5. Prediction accuracy --
    results.cross_valid_accuracy_test *= 100
    ax = fig.add_subplot(gs[0, 4])
    s = sns.barplot(x = 'cross_valid_accuracy_test', y = 'para_notation_with_best_fit', data = results, color = 'grey')
    plt.axvline(50, color='k', linestyle='--')
    ax.set_xlim(min(50, np.min(results.cross_valid_accuracy_test)) - 5)
    ax.set_ylabel('')
    ax.set_xlabel('Prediction_accuracy_cross_valid')
    s.set_yticklabels('')

    return


def set_label(h,ii,jj, model_notations):
    
    if jj == 0:
        h.set_yticklabels(model_notations, rotation = 0)
    else:
        h.set_yticklabels('')
        
    if ii == 0:
        h.set_xticklabels(model_notations, rotation = 45, ha = 'left')
        h.xaxis.tick_top()
    else:
        h.set_xticklabels('')
             











