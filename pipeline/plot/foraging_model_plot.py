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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from utils.plot_fitting import set_label

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm

from utils.helper_func import moving_average

# import models.bandit_model_comparison
# import matplotlib as mpl 
# mpl.rcParams['figure.dpi'] = 300

# matplotlib.use('qt5agg')
plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})

# from utils.helper_func import seaborn_style
# seaborn_style()

plt.rcParams.update({'figure.max_open_warning': 0})
sns.set(context='talk')

def corrfunc(x, y, **kws):
    (r, p) = stats.pearsonr(x, y)
    ax = plt.gca()
    title_obj = ax.set_title("r = %.3f, p = %.4f " % (r, p), fontsize=8)
    if p < 0.05:
        plt.setp(title_obj, color='r')


def plot_each_mice(group_result, if_hattori_Fig1I):
    
    # Unpack data
    file = group_result['file']
    data = group_result['raw_data']

    grand_result = data['model_comparison_grand'].results
    sessionwise_result = data['model_comparison_session_wise']
    overall_best = np.where(grand_result['best_model_AIC'])[0][0] + 1
    session_best_matched = group_result['session_best'] == overall_best

    if_CVed = 'k_fold' in group_result

    # Update notations
    if 'para_notation_with_best_fit' in grand_result:
        model_notations = grand_result['para_notation_with_best_fit']
    else:
        model_notations = []
        for i, row in grand_result.iterrows():
            model_notations.append('(' + str(i) + ') ' + row.para_notation + '\n' + str(np.round(row.para_fitted, 2)))

    fitted_para_names = np.array(grand_result['para_notation'].iloc[overall_best - 1].split(','))

    # plt.close('all')
    sns.set(context='talk')

    plt.rcParams.update({'font.size': 8, 'figure.dpi': 150})

    fig = plt.figure(figsize=(15, 9 / 15 * len(grand_result)), dpi=150)
    gs = GridSpec(1, 20, wspace=0.1, bottom=0.15, top=0.9, left=0.15, right=0.95)
    fig.text(0.01, 0.95, '%s' % (file), fontsize=20)

    # === 1.1 Overall result ===
    ax = fig.add_subplot(gs[0, 0:1])
    sns.heatmap(grand_result[['model_weight_AIC']], annot=True, fmt=".2f", square=False, cbar=False, cbar_ax=[0, 1])

    patch = Rectangle((0, np.where(grand_result['best_model_AIC'])[0]), 1, 1, color='dodgerblue', linewidth=4,
                      fill=False)
    ax.add_artist(patch)

    set_label(ax, 1, 0, model_notations)
    ax.set_xticklabels(['Overall\n(' + str(data['model_comparison_grand'].n_trials) + ')'])

    # --- 1.2 Session-wise model weight ---
    ax = fig.add_subplot(gs[0, 1: round(20)])
    sns.heatmap(group_result['model_weight_AIC'], annot=True, fmt=".2f", square=False, cbar=False, cbar_ax=[0, 1])
    ax.set_yticklabels('')
    ax.set_xticklabels(group_result['xlabel'])

    for ss, this_mc in enumerate(sessionwise_result):
        patch = Rectangle((ss, group_result['session_best'][ss] - 1), 1, 1, color='dodgerblue', linewidth=4, fill=False)
        ax.add_artist(patch)

    # -- Fitting results --
    data['model_comparison_grand'].plot_predictive_choice()

    # -- 2.1 deltaAIC relative to RW1972-softmax-noBias (Fig.1I of Hattori 2019) --        
    x = group_result['session_number']
    marker_sizes = (group_result["n_trials"][0] / 100 * 2) ** 2

    fig = plt.figure(figsize=(15, 5), dpi=150)
    gs = GridSpec(1, 20, wspace=0.2, bottom=0.15, top=0.9, left=0.07, right=0.95)

    if if_hattori_Fig1I:
        plot_colors = ['orange', 'b', 'g', 'r']  # Orange, blue, green, red

        ax = fig.add_subplot(gs[0, 0:9])
        plt.axhline(0, c='k', ls='--')
        for mm, cc in enumerate(plot_colors):
            # plt.plot(group_result['session_number'].T, group_result['delta_AIC'].T)
            y = group_result['delta_AIC'][mm, :]
            ax.plot(x, y, color=cc, label=group_result['delta_AIC_para_notation'].iloc[mm], linewidth=0.7)

            ax.scatter(x[session_best_matched], y[session_best_matched], color=cc, s=marker_sizes[session_best_matched],
                       alpha=0.9)
            ax.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)],
                       facecolors='none', edgecolors=cc,
                       s=marker_sizes[np.logical_not(session_best_matched)], alpha=0.7)

        ax.text(min(plt.xlim()), 10, '(12) RW1972-noBias')
        ax.set_xlabel('Session number')
        ax.set_ylabel('$\\Delta$ AIC')
        ax.legend()

        # -- 2.2 Likelihood per trial, prediction accuracy (nonCVed), and foraging efficiency --
    ax_grand = fig.add_subplot(gs[0, 10])
    plt.axhline(0.5, c='k', ls='--')

    # > Overall
    # # LPT_AIC
    # sns.pointplot(data = group_result['LPT_AIC'][overall_best-1,:], ax = ax_grand, color = 'b', ci = 68)
    # ax_grand.plot(-1, group_result['LPT_AIC_grand'][overall_best-1], marker = 's', markersize = 13, color = 'b')

    if not if_CVed:
        hattori_col_name = 'prediction_accuracy_NONCV'
        hattori_label = 'Pred. acc. best model noCV'
        bias_col_name = 'prediction_accuracy_bias_only'
        bias_label = 'Pred. acc. bias only noCV'
    else:
        hattori_col_name = 'prediction_accuracy_CV_test'
        hattori_label = 'Pred. acc. best model %g-CV' % group_result['k_fold']
        bias_col_name = 'prediction_accuracy_CV_test_bias_only'
        bias_label = 'Pred. acc. bias only %g-CV' % group_result['k_fold']

    # Prediction accuracy NONCV
    sns.pointplot(data=group_result[hattori_col_name], ax=ax_grand, color='black', ci=68)
    ax_grand.plot(-1, group_result['prediction_accuracy_NONCV_grand'], marker='s', markersize=13, color='black')

    # Prediction accuracy bias only
    sns.pointplot(data=group_result[bias_col_name], ax=ax_grand, color='gray', ci=68)
    ax_grand.plot(-1, group_result['prediction_accuracy_bias_only_grand'], marker='s', markersize=13, color='gray')

    # Foraging efficiency
    sns.pointplot(data=group_result['foraging_efficiency'], ax=ax_grand, color='g', ci=68, marker='^')
    ax_grand.plot(-1, group_result['foraging_efficiency_grand'], marker='^', markersize=13, color='g')

    # > Session-wise
    ax = fig.add_subplot(gs[0, 11:20], sharey=ax_grand)
    plt.axhline(0.5, c='k', ls='--')

    # # LPT_AIC
    # x = group_result['session_number']
    # y = group_result['LPT_AIC'][overall_best-1,:]
    # plt.plot(x, y, 'b',label = 'likelihood per trial (AIC)', linewidth = 0.7)
    # plt.scatter(x[session_best_matched], y[session_best_matched], color = 'b', s = marker_sizes, alpha = 0.9)
    # plt.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)], 
    #             facecolors='none', edgecolors = 'b', s = marker_sizes, alpha = 0.7)

    # Prediction accuracy NONCV
    y = group_result[hattori_col_name]
    plt.plot(x, y, 'k', label=hattori_label, linewidth=0.7)
    plt.scatter(x[session_best_matched], y[session_best_matched], color='k', s=marker_sizes[session_best_matched],
                alpha=0.9, label='session = Overall best')
    plt.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)],
                facecolors='none', edgecolors='k', s=marker_sizes[np.logical_not(session_best_matched)], alpha=0.7,
                label='session $\\neq$ Overall best')

    # Prediction accuracy bias only
    y = group_result[bias_col_name]
    plt.plot(x, y, 'gray', ls='--', label=bias_label, linewidth=0.7)
    plt.scatter(x[session_best_matched], y[session_best_matched], color='gray', s=marker_sizes[session_best_matched],
                alpha=0.9)
    plt.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)],
                facecolors='none', edgecolors='gray', s=marker_sizes[np.logical_not(session_best_matched)], alpha=0.7)

    # Foraging efficiency
    y = group_result['foraging_efficiency']
    plt.plot(x, y, 'g', ls='-', label='foraging efficiency', linewidth=0.7)
    plt.scatter(x[session_best_matched], y[session_best_matched], color='g', s=marker_sizes[session_best_matched],
                alpha=0.9, marker='^')
    plt.scatter(x[np.logical_not(session_best_matched)], y[np.logical_not(session_best_matched)],
                facecolors='none', edgecolors='g', s=marker_sizes[np.logical_not(session_best_matched)], alpha=0.7,
                marker='^')

    ax_grand.set_xticks([-1, 0])
    ax_grand.set_xlim([-1.5, .5])
    ax_grand.set_xticklabels(['Overall', 'mean'], rotation=45)
    # ax_grand.set_ylabel('Likelihood per trial (AIC)')
    plt.setp(ax.get_yticklabels(), visible=False)

    plt.title('Overall best: %s {%s}' % (
    grand_result['model'].iloc[overall_best - 1], grand_result['para_notation'].iloc[overall_best - 1]))
    plt.xlabel('Session number')
    plt.legend()

    # -- 3 Fitted paras of the overall best model --
    para_plot_group = [[0, 1, 2], [3], [4]]
    para_plot_color = [('g', 'r', 'k'), ('k'), ('k')]

    fig = plt.figure(figsize=(15, 5), dpi=150)
    gs = GridSpec(1, len(para_plot_group) * 10, wspace=0.2, bottom=0.15, top=0.85, left=0.05, right=0.95)

    for pp, (ppg, ppc) in enumerate(zip(para_plot_group, para_plot_color)):

        x = group_result['session_number']
        y = group_result['fitted_paras'][ppg, :]

        ax_grand = fig.add_subplot(gs[0, (pp * 10) + 1])
        ax_grand.set_prop_cycle(color=ppc)
        plt.axhline(0, c='k', ls='--')
        ax_grand.plot(np.array([[-1] * len(ppg)]), np.array([group_result['fitted_paras_grand'][ppg]]), 's',
                      markersize=13, alpha=0.8)

        ax = fig.add_subplot(gs[0, (pp * 10 + 2): (pp * 10 + 10)], sharey=ax_grand)
        ax.set_prop_cycle(color=ppc)

        plt.plot(x.T, y.T, linewidth=0.7)

        for ny, yy in enumerate(y):
            ax.scatter(x[session_best_matched], yy[session_best_matched], s=marker_sizes[session_best_matched],
                       alpha=0.9)
            ax.scatter(x[np.logical_not(session_best_matched)], yy[np.logical_not(session_best_matched)],
                       edgecolor=ppc[ny], facecolors='none', s=marker_sizes[np.logical_not(session_best_matched)],
                       alpha=0.9)
            sns.pointplot(data=yy.T, ax=ax_grand, color=ppc[ny], ci=68, alpha=0.8)

        plt.xlabel('Session number')
        if pp == 0: ax_grand.set_ylabel('Fitted parameters')
        plt.legend(fitted_para_names[ppg], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower center', ncol=4)
        plt.axhline(0, c='k', ls='--')

        ax_grand.set_xticks([-1, 0])
        ax_grand.set_xlim([-1.5, .5])
        ax_grand.set_xticklabels(['Overall', 'mean'], rotation=45)
        plt.setp(ax.get_yticklabels(), visible=False)

    # plt.pause(10)      
    return


def plot_group_results(result_path="..\\results\\model_comparison\\w_bias_8\\",
                       group_results_name='group_results_all_with_bias.npz',
                       average_session_number_range=None):
    sns.set()

    # results_all_mice = pd.read_pickle(result_path + group_results_name)
    data = np.load(result_path + group_results_name, allow_pickle=True)
    group_results = data.f.group_results.item()
    results_all_mice = group_results['results_all_mice']
    raw_LPT_AICs = group_results['raw_LPT_AICs']

    if_hattori_Fig1I = group_results['if_hattori_Fig1I']

    # to %
    # results_all_mice[['foraging_efficiency', 'prediction_accuracy_NONCV', 'prediction_accuracy_bias_only', 'prediction_accuracy_Sugrue_NONCV']] *= 100
    results_all_mice.update(results_all_mice.filter(regex='prediction_accuracy|efficiency') * 100)

    if_CVed = 'k_fold' in group_results
    if if_CVed:
        CV_k_fold = group_results['k_fold']

    if average_session_number_range is None:
        average_session_number_range = [0, np.inf]

    select_average_session = (average_session_number_range[0] <= results_all_mice['session_number']) & \
                             (results_all_mice['session_number'] <= average_session_number_range[1])

    results_all_mice_session_filtered = results_all_mice[select_average_session]

    fitted_para_names = group_results['fitted_para_names']

    # == Plotting ==
    # -- 1. deltaAIC, aligned to session_number (Hattori Figure 1I) --
    if if_hattori_Fig1I:
        delta_AIC_para_notation = group_results['delta_AIC_para_notation']
        fig = plt.figure(figsize=(10, 5), dpi=150)
        ax = fig.subplots()
        plt.axhline(0, c='k', ls='--')
        palette = sns.color_palette(['orange', 'b', 'g', 'r'])
        # hh = sns.pointplot(data = results_all_mice.melt(id_vars = ['session_idx', 'session_number'], var_name='parameters',
        #                                            value_vars = delta_AIC_para_notation.tolist()),
        #               x = 'session_number', y = 'value' , hue = 'parameters', ci = 68, palette=palette)

        for pp, var in zip(palette, delta_AIC_para_notation.tolist()):
            means = results_all_mice.groupby('session_number')[var].mean()
            errs = results_all_mice.groupby('session_number')[var].sem()
            ax.errorbar(means.index, means, marker='o', yerr=errs, color=pp, label=var)

        ax.text(min(plt.xlim()), 10, '(12) RW1972-noBias')
        ax.set_xlabel('Session number (actual)')
        ax.set_ylabel('$\\Delta$ AIC')
        # ax.set_title('%g mice' % group_results['n_mice'])
        plt.xticks(rotation=60, horizontalalignment='right')

        n_mice_per_session = results_all_mice.session_number.value_counts().sort_index()
        ax.plot(means.index, n_mice_per_session.values * max(plt.ylim()) / max(n_mice_per_session.values) * 0.9,
                'ks-', label='max = %g mice' % group_results['n_mice'])

        if average_session_number_range is not None:
            patch = Rectangle((average_session_number_range[0], min(plt.ylim())),
                              np.diff(np.array(average_session_number_range)), np.diff(np.array(plt.ylim())),
                              color='gray', linewidth=1, fill=True, alpha=0.2)
            ax.add_artist(patch)

        ax.legend()
    else:
        print('No Fig.1I of Hattori2019')

    # -- 1.2 rawAIC, all models --
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.subplots()

    plt.axhline(0.5, c='k', ls='--')

    for n_m, var in enumerate(group_results['para_notation']):
        means = raw_LPT_AICs.groupby('session_number')[var].mean()
        errs = raw_LPT_AICs.groupby('session_number')[var].sem()
        ax.errorbar(means.index + 0.07 * n_m - 0.04, means, yerr=errs, label=var, linewidth=0.7, marker='o', alpha=0.7)

    ax.set_xlabel('Session number (actual)')
    ax.set_ylabel('LPT_AIC')
    plt.xticks(rotation=60, horizontalalignment='right')

    # n_mice_per_session = results_all_mice.session_number.value_counts().sort_index()
    # ax.plot(means.index, n_mice_per_session.values * max(plt.ylim()) / max(n_mice_per_session.values) * 0.9, 
    #         'ks-', label = 'max = %g mice' % group_results['n_mice'])

    if average_session_number_range is not None:
        patch = Rectangle((average_session_number_range[0], min(plt.ylim())),
                          np.diff(np.array(average_session_number_range)), np.diff(np.array(plt.ylim())),
                          color='gray', linewidth=1, fill=True, alpha=0.2)
        ax.add_artist(patch)

    ax.legend()

    # -- 1.5 Foraging efficiency, aligned to session_number --
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.subplots()
    plt.axhline(100, c='k', ls='--')

    means = results_all_mice.groupby('session_number')['foraging_efficiency'].mean()
    errs = results_all_mice.groupby('session_number')['foraging_efficiency'].sem()
    ax.set_ylim([50, 120])
    ax.errorbar(means.index, means, marker='^', yerr=errs, color='g')

    ax.text(min(plt.xlim()), 100, 'Ideal-$\hat{p}$-optimal')
    ax.set_xlabel('Session number (actual)')
    ax.set_ylabel('Foraging efficiency')

    if average_session_number_range is not None:
        patch = Rectangle((average_session_number_range[0], min(plt.ylim())),
                          np.diff(np.array(average_session_number_range)), np.diff(np.array(plt.ylim())),
                          color='gray', linewidth=1, fill=True, alpha=0.2)
        ax.add_artist(patch)

    # -- 2. deltaAIC, aligned to session_idx (Hattori Figure 1I) --
    if if_hattori_Fig1I:
        fig = plt.figure(figsize=(10, 5), dpi=150)
        ax = fig.subplots()
        plt.axhline(0, c='k', ls='--')
        palette = sns.color_palette(['orange', 'b', 'g', 'r'])
        # sns.pointplot(data = results_all_mice_session_filtered.melt(id_vars = ['session_idx', 'session_number'], var_name='parameters',
        #                                            value_vars = delta_AIC_para_notation.tolist()),
        #               x = 'session_idx', y = 'value' , hue = 'parameters', ci = 68, palette=palette)

        for pp, var in zip(palette, delta_AIC_para_notation.tolist()):
            means = results_all_mice_session_filtered.groupby('session_idx')[var].mean()
            errs = results_all_mice_session_filtered.groupby('session_idx')[var].sem()
            ax.errorbar(means.index, means, marker='o', yerr=errs, color=pp, label=var)

        ax.text(min(plt.xlim()), 10, '(12) RW1972-noBias')
        ax.set_xlabel('Session Index')
        ax.set_ylabel('$\\Delta$ AIC')
        # ax.set_title('%g mice' % group_results['n_mice'])
        plt.xticks(rotation=60, horizontalalignment='right')

        n_mice_per_session = results_all_mice_session_filtered.session_idx.value_counts().sort_index()
        ax.plot(n_mice_per_session.values * max(plt.ylim()) / max(n_mice_per_session.values) * 0.9, 'ks-',
                label='max = %g mice' % group_results['n_mice'])

        ax.legend()

        # -- 2.5 Foraging efficiency, aligned to session_idx --
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.subplots()
    plt.axhline(100, c='k', ls='--')

    means = results_all_mice_session_filtered.groupby('session_idx')['foraging_efficiency'].mean()
    errs = results_all_mice_session_filtered.groupby('session_idx')['foraging_efficiency'].sem()
    ax.set_ylim([50, 120])
    ax.errorbar(means.index, means, marker='^', yerr=errs, color='g')

    ax.text(min(plt.xlim()), 100, 'Ideal-$\hat{p}$-optimal')
    ax.set_xlabel('Session Index')
    ax.set_ylabel('Foraging efficiency')

    # -- 3. Prediction accuracy (Hattori Figure 1J)
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.subplots()
    plt.axhline(50, c='k', ls='--')

    if not if_CVed:  # No CVed data
        hattori_col_name = 'prediction_accuracy_NONCV'
        hattori_label = "Hattori (noCV)"
        bias_label = "Bias only (noCV)"
        prediction_accuracies = results_all_mice_session_filtered[
            ['mice', hattori_col_name, 'prediction_accuracy_bias_only']]
        prediction_accuracies = prediction_accuracies.rename(
            columns={"prediction_accuracy_NONCV": hattori_label, "prediction_accuracy_bias_only": bias_label})
        prediction_accuracies = pd.DataFrame.melt(prediction_accuracies, id_vars='mice', var_name='models',
                                                  value_name='value')
    else:
        hattori_col_name = 'prediction_accuracy_CV_test'
        hattori_label = "Hattori acc (%g-CV)" % CV_k_fold
        bias_label = "Bias only (%g-CV)" % CV_k_fold
        prediction_accuracies = results_all_mice_session_filtered[
            ['mice', hattori_col_name, 'prediction_accuracy_CV_test_bias_only']]
        prediction_accuracies = prediction_accuracies.rename(
            columns={"prediction_accuracy_CV_test": hattori_label, "prediction_accuracy_CV_test_bias_only": bias_label})
        prediction_accuracies = pd.DataFrame.melt(prediction_accuracies, id_vars='mice', var_name='models',
                                                  value_name='value')

    x = "mice"
    y = "value"
    hue = 'models'
    hh = sns.violinplot(x=x, y=y, hue=hue, data=prediction_accuracies, inner="box",
                        hue_order=[bias_label, hattori_label],
                        palette=sns.color_palette(['lightgray', 'r']))

    ax.set_ylabel(hattori_label)
    plt.xticks(rotation=45, horizontalalignment='right')
    ax.set_xlabel('')

    # Add Wilcoxon test
    box_pairs = [((s, hattori_label), (s, bias_label)) for s in prediction_accuracies.mice.unique()]

    add_stat_annotation(hh, data=prediction_accuracies, x=x, y=y, hue=hue, box_pairs=box_pairs,
                        test='Wilcoxon', loc='inside', verbose=0, line_offset_to_box=0.2)

    # -- 4. fitted_para_names --
    # -- Separately
    for ff, fitted_para_this in enumerate(fitted_para_names):
        fig = plt.figure(figsize=(10, 5), dpi=150)
        ax = fig.subplots()
        plt.axhline(0, c='k', ls='--')
        sns.violinplot(x="mice", y=fitted_para_this, data=results_all_mice_session_filtered, inner="box")
        ax.set_title(fitted_para_this)
        plt.xticks(rotation=45, horizontalalignment='right')
        ax.set_xlabel('')
        ax.set_ylabel('')

    # -- Step sizes comparison
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = fig.subplots()
    plt.axhline(0, c='k', ls='--')

    step_sizes = results_all_mice_session_filtered[['mice', '$\\alpha_{rew}$', ' $\\alpha_{unr}$', ' $\\delta$']]
    step_sizes = pd.DataFrame.melt(step_sizes, id_vars='mice', var_name='paras', value_name='value')
    x = "mice"
    y = "value"
    hue = 'paras'
    # sns.violinplot(x=x, y =y, hue=hue, data = step_sizes, inner="box")    
    hh = sns.boxplot(x=x, y=y, hue=hue, data=step_sizes, palette=sns.color_palette(['g', 'r', 'gray']))
    hh.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower center', ncol=4)

    plt.xticks(rotation=45, horizontalalignment='right')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add Wilcoxon test
    box_pairs = []
    for s in step_sizes.mice.unique():
        box_pairs.extend([((s, '$\\alpha_{rew}$'), (s, ' $\\alpha_{unr}$')),
                          ((s, '$\\alpha_{rew}$'), (s, ' $\\delta$')),
                          ((s, ' $\\alpha_{unr}$'), (s, ' $\\delta$'))])

    add_stat_annotation(hh, data=step_sizes, x=x, y=y, hue=hue, box_pairs=box_pairs,
                        test='Wilcoxon', loc='inside', verbose=0, line_offset_to_box=0.2, line_offset=0.03,
                        line_height=0.01, text_offset=0.2)

    # % -- 5. All mice statistics --
    fig = plt.figure(figsize=(10, 5), dpi=150)
    gs = GridSpec(1, 6, wspace=1, bottom=0.15, top=0.85, left=0.1, right=0.95)

    # 1> Prediction accuracy
    ax = fig.add_subplot(gs[0, 0:2])
    hh = sns.barplot(x='models', y='value', data=prediction_accuracies, order=[bias_label, hattori_label],
                     capsize=.1, palette=sns.color_palette(['lightgray', 'r']))
    plt.axhline(50, c='k', ls='--')

    add_stat_annotation(hh, data=prediction_accuracies, x='models', y='value', box_pairs=[[bias_label, hattori_label]],
                        test='Wilcoxon', loc='inside', verbose=0, line_offset_to_box=0)
    ax.set_xlabel('')
    ax.set_ylabel(hattori_label)
    ax.set_title('n = %g sessions' % len(prediction_accuracies))

    # 2> Step sizes 
    # Turn to mice-wise like Hattori??
    step_sizes = results_all_mice_session_filtered[['mice', '$\\alpha_{rew}$', ' $\\alpha_{unr}$', ' $\\delta$']]
    step_sizes_mice_average = step_sizes.groupby(['mice']).mean()
    step_sizes_mice_average = pd.DataFrame.melt(step_sizes_mice_average)

    ax = fig.add_subplot(gs[0, 2:4])
    x = 'variable'
    y = 'value'
    hh = sns.boxplot(x=x, y=y, data=step_sizes_mice_average,
                     palette=sns.color_palette(['g', 'r', 'gray']), width=0.5)
    sns.swarmplot(x=x, y=y, data=step_sizes_mice_average, color='k')

    add_stat_annotation(hh, data=step_sizes_mice_average, x=x, y=y,
                        box_pairs=[('$\\alpha_{rew}$', ' $\\alpha_{unr}$'), ('$\\alpha_{rew}$', ' $\\delta$'),
                                   (' $\\alpha_{unr}$', ' $\\delta$')],
                        test='Mann-Whitney', loc='inside', verbose=0, line_offset_to_box=0.1, line_offset=0.03,
                        line_height=0.01, text_offset=0.2)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('n = %g mice' % group_results['n_mice'])
    plt.axhline(0, c='k', ls='--')

    # 3> beta(1/softmax_sigma)
    # Turn to mice-wise like Hattori??
    betas = results_all_mice_session_filtered[['mice', ' $\\sigma$', ' $b_L$']]
    betas_mice_average = betas.groupby(['mice']).mean()
    data = 1 / betas_mice_average[' $\\sigma$']

    ax = fig.add_subplot(gs[0, 4])
    hh = sns.boxplot(data=data, width=0.5, color='gray')
    sns.swarmplot(data=data, color='k')
    ax.set_ylim([0, 16])
    ax.set_xticklabels(['$\\beta_{\\Delta Q}$'])
    ax.set_ylabel('')

    # 4> beta_0 (Hattori's bias)
    ax = fig.add_subplot(gs[0, 5])
    data = betas_mice_average[' $b_L$']
    hh = sns.boxplot(data=data, width=0.5, color='gray')
    sns.swarmplot(data=data, color='k')

    plt.axhline(0, c='k', ls='--')

    ax.set_xticklabels(['$\\beta_0$'])
    ax.set_ylabel('')
    ax.set_ylim([-2, 2])

    # % 5> Correlations
    # - 1. All paras, session filtered -
    data = results_all_mice_session_filtered[
        ['session_number', hattori_col_name, 'prediction_accuracy_Sugrue_NONCV', 'foraging_efficiency',
         '$\\alpha_{rew}$', ' $\\alpha_{unr}$', ' $\\delta$', ' $\\sigma$', ' $b_L$', ]]
    data = data.rename(columns={'session_number': 'session #', hattori_col_name: hattori_label,
                                'prediction_accuracy_Sugrue_NONCV': 'pred. accu. Sugrue',
                                'foraging_efficiency': 'foraging eff.'})

    hh = sns.pairplot(data=data,
                      kind="reg", height=1.5, corner=True,
                      plot_kws={'scatter_kws': dict(s=5, color='gray'), 'line_kws': dict(color='k')})

    hh.map_lower(corrfunc)
    plt.tight_layout()

    # - 2. Compare cross-validation versus non-cross-validation -
    if if_CVed:
        def plot_diag(x, y, **kargs):
            plt.plot([50, 100], [50, 100], 'k--')

        data = results_all_mice_session_filtered[['prediction_accuracy_NONCV', 'prediction_accuracy_CV_test',
                                                  'prediction_accuracy_bias_only',
                                                  'prediction_accuracy_CV_test_bias_only']]
        data = data.rename(columns={'prediction_accuracy_NONCV': 'Hattori non-CV',
                                    'prediction_accuracy_CV_test': 'Hattori %g-CV' % CV_k_fold,
                                    'prediction_accuracy_bias_only': 'Bias only non-CV',
                                    'prediction_accuracy_CV_test_bias_only': 'Bias only %g-CV' % CV_k_fold})

        hh = sns.pairplot(data=data,
                          kind="reg", height=1.5, corner=True,
                          plot_kws={'scatter_kws': dict(s=5, color='gray'), 'line_kws': dict(color='k')})

        hh.map_lower(plot_diag)
        plt.tight_layout()

    # % - Foraging Efficiency VS Prediction Accuracy (all sessions included) -
    # > All mice
    data = results_all_mice[['session_number', 'mice',
                             hattori_col_name, 'prediction_accuracy_Sugrue_NONCV', 'foraging_efficiency',
                             'n_trials']].copy()

    # palette = sns.color_palette("coolwarm",len(results_all_mice['session_number'].unique()))
    palette_all = sns.color_palette("RdYlGn", max(results_all_mice['session_number'].unique()))

    palette = []
    for s in np.sort(results_all_mice['session_number'].unique()):
        palette.append(palette_all[s - 1])

    # Hattori prediction accuracy
    x, y = [hattori_col_name, "foraging_efficiency"]
    hh = sns.relplot(x=x, y=y, hue="session_number", size="n_trials",
                     sizes=(40, 400), alpha=.5, palette=palette, height=6, data=data, legend=False)

    plt.axhline(100, c='k', ls='--', lw=1)
    plt.axvline(50, c='k', ls='--', lw=1)

    (r, p) = stats.pearsonr(data[x], data[y])

    ## OLS
    sns.regplot(x=x, y=y, ax=hh.ax, data=data,
                scatter=False, label='r = %.3g\np = %.3g' % (r, p), color='k')

    plt.xlabel(hattori_label)
    plt.ylabel('Foraging efficiency %')

    plt.legend()
    plt.title('All sessions')
    plt.tight_layout()

    # Sugrue prediction acc (Matron's question)
    x, y = ["prediction_accuracy_Sugrue_NONCV", "foraging_efficiency"]
    hh = sns.relplot(x=x, y=y, hue="session_number", size="n_trials",
                     sizes=(40, 400), alpha=.5, palette=palette, height=6, data=data, legend=False)

    plt.axhline(100, c='k', ls='--', lw=1)
    plt.axvline(50, c='k', ls='--', lw=1)

    (r, p) = stats.pearsonr(data[x], data[y])

    ## OLS
    sns.regplot(x=x, y=y, ax=hh.ax, data=data,
                scatter=False, label='r = %.3g\np = %.3g' % (r, p), color='k')

    plt.xlabel('Sugrue pred. acc. (noCVed)%')
    plt.ylabel('Foraging efficiency %')

    plt.legend()
    plt.title('All sessions')
    plt.tight_layout()

    # % > Each mice
    fig = plt.figure(figsize=(9, 8), dpi=150)
    n_mice = len(data.mice.unique())
    gs = GridSpec(int(np.ceil(n_mice / 4)), 4, hspace=.6, wspace=0.5,
                  left=0.1, right=0.95, bottom=0.05, top=0.95)

    for mm, mouse in enumerate(data.mice.unique()):
        ax = fig.add_subplot(gs[mm])

        x = data[data.mice == mouse][hattori_col_name]
        y = data[data.mice == mouse].foraging_efficiency
        (r, p) = stats.pearsonr(x, y)

        # palette = sns.color_palette("coolwarm", sum(data.mice == mouse))  # Relative colormap
        # Use absolute colormap instead
        palette = []
        for s in np.sort(results_all_mice[data.mice == mouse]['session_number'].unique()):
            palette.append(palette_all[s - 1])

        sns.scatterplot(x=hattori_col_name, y='foraging_efficiency', data=data[data.mice == mouse],
                        hue='session_number', size='n_trials',
                        sizes=(20, 100), alpha=0.8, palette=palette, ax=ax, legend=False)

        # OLS
        sns.regplot(x, y, ax=ax, scatter=False, label='r$^2$ = %.3f, p = %.3f' % (r ** 2, p), color='k',
                    line_kws={'linestyle': ('--', '-')[p < 0.05], 'lw': 2})

        # Trial-number-weighted WLS
        # wls_model = sm.WLS(y,sm.add_constant(x), weights = data[data.mice == mouse].session_number)
        # result_wls = wls_model.fit()
        # b_wls, k_wls = result_wls.params
        # r2_wls, p_wls = result_wls.rsquared, result_wls.pvalues[1]
        # plt.plot(np.sort(x), result_wls.fittedvalues[np.argsort(x)], 'k', label='r$^2$ = %.3f, p = %.3f'%(r2_wls, p_wls), linestyle = ('--','-')[p_wls<0.05])

        plt.legend(fontsize=7, handlelength=0)

        if mm > 0:
            plt.xlabel('')
            plt.ylabel('')
        else:
            plt.xlabel(hattori_label)
            plt.ylabel('Foraging efficiency %')

        plt.xlim([48, 100])
        plt.axhline(100, c='k', ls='--', lw=1)
        plt.axvline(50, c='k', ls='--', lw=1)
        plt.title(mouse)

    # hh = sns.relplot(x=hattori_col_name, y="foraging_efficiency", hue = "session_number", col="mice", size="n_trials",
    #         sizes=(40, 400), alpha=.5, palette = palette, data=data, legend = False, col_wrap = 4, height=3, aspect=1)
    # plt.tight_layout()
    plt.show()

    # %
    # plt.pause(10)
    return


def plot_example_sessions(result_path="..\\results\\model_comparison\\w_wo_bias_15\\",
                          combine_prefix='model_comparison_15_CV_patched_',
                          group_results_name='group_results_15_CV.npz', session_of_interest=[['FOR05', 33]],
                          block_partitions=[70, 70], smooth_factor=1):

    from utils.plot_fitting import plot_model_comparison_predictive_choice_prob

    sns.set()

    data = np.load(result_path + group_results_name, allow_pickle=True)
    group_results = data.f.group_results.item()
    results_all_mice = group_results['results_all_mice']

    for soi in session_of_interest:
        mouse, session = soi
        data = np.load(result_path + combine_prefix + mouse + '.npz', allow_pickle=True)
        data = data.f.results_each_mice.item()
        this_entry = results_all_mice[(results_all_mice.mice == mouse) & (results_all_mice.session_number == session)]

        session_idx = this_entry.session_idx.values[0] - 1
        this_class = data['model_comparison_session_wise'][session_idx]

        # -- Recovery some essentials 
        this_class.plot_predictive = [1]
        this_class.p_reward = data['model_comparison_grand'].p_reward[:,
                              data['model_comparison_grand'].session_num == session]

        # -- Show and plot fitting results
        this_class.show()
        this_class.plot()

        # -- Plot session
        plot_model_comparison_predictive_choice_prob(this_class, smooth_factor=smooth_factor)
        fig = plt.gcf()
        fig.text(0.05, 0.94,
                 'Mouse = %s, Session_number = %g (idx = %g), Foraging eff. = %g%%' % (mouse, session, session_idx,
                                                                                       this_entry.foraging_efficiency.iloc[
                                                                                           0] * 100), fontsize=13)
        # -- Conventional Runlength --
        choice_history = this_class.fit_choice_history
        p_reward = this_class.p_reward

        fig = plt.figure(figsize=(8, 3), dpi=150)
        gs = GridSpec(1, 2, hspace=.6, wspace=0.5,
                      left=0.15, right=0.95, bottom=0.05, top=0.95)

        ax = fig.add_subplot(gs[0, 0])
        temp = np.array([[-999]])  # -999 is to capture the first and the last stay
        changeover_position = np.where(np.diff(np.hstack((temp, choice_history, temp))))[1]
        stay_durations = np.diff(changeover_position)
        bins = np.arange(1, np.max(stay_durations) + 1) - 0.5
        sns.distplot(stay_durations, bins=bins, norm_hist=False, ax=ax)
        plt.xlabel('Runlength (grand)')
        plt.xlim(0.5, max(plt.xlim()))
        plt.xticks(np.r_[1, 5:np.max(stay_durations):5])

        # == Lau2005 style (try to find any clue of optimality) == 
        # Locate block switch
        p_reward_ratio = p_reward[1] / p_reward[0]  # R/L
        block_starts = np.where(np.diff(np.hstack((-999, p_reward_ratio, 999))))[0]

        # Block length distribution
        ax = fig.add_subplot(gs[0, 1])
        sns.distplot(np.diff(block_starts), ax=ax, bins=20, kde=False)
        plt.xlabel('Block length')

        # -- Lau2005 Runlength -- 
        df_run_length_Lau = analyze_runlength_Lau2005(choice_history, p_reward)
        plot_runlength_Lau2005(df_run_length_Lau, block_partitions)


def analyze_runlength_Lau2005(choice_history, p_reward, min_trial=50, block_partitions=[70, 70]):
    '''
    Runlength analysis in Fig.5, Lau2005
    '''
    # %
    df_run_length_Lau = [pd.DataFrame(), pd.DataFrame()]  # First half, Second half

    p_reward_ratio = p_reward[1] / p_reward[0]  # R/L
    block_starts = np.where(np.diff(np.hstack((-999, p_reward_ratio, 999))))[0]

    # For each block
    for bb in range(len(block_starts) - 1):
        select_this_block = range(block_starts[bb], block_starts[bb + 1])
        this_block_choice = choice_history[0, select_this_block]
        # print(len(this_session_choice))
        this_block_p_base_ratio = p_reward_ratio[select_this_block][0]

        # Flip choices such that 1 = rich arm, 0 = lean arm
        if this_block_p_base_ratio < 1:  # if rich arm = Left
            this_block_choice = 1 - this_block_choice

        # Define block partitions
        block_len = len(this_block_choice)

        if block_len < 30: continue  # Exclude too short blocks

        select_trials = [this_block_choice[:int(block_partitions[0] / 100 * block_len)],
                         this_block_choice[int(((1 - block_partitions[0] / 100) * block_len)):]]

        for pp, this_half_choice in enumerate(select_trials):

            # Get runlength for the best (rich) and the worst (lean) arm
            # These two magic lines below are correct, believe it or not :)
            this_runlength_rich = \
            np.diff(np.where(np.hstack((999, np.diff(np.where(this_half_choice == 1)[0]), 999)) > 1))[0]
            this_runlength_lean = \
            np.diff(np.where(np.hstack((999, np.diff(np.where(this_half_choice == 0)[0]), 999)) > 1))[0]
            # assert(np.sum(this_runlength_rich) == np.sum(this_half_choice == 1))
            # assert(np.sum(this_runlength_lean) == np.sum(this_half_choice == 0))

            # Remove the first and the last run (due to slicing of the blocks/halves)
            if this_half_choice[0] == 1:
                this_runlength_rich = this_runlength_rich[1:]
            else:
                this_runlength_lean = this_runlength_lean[1:]

            if this_half_choice[-1] == 1:
                this_runlength_rich = this_runlength_rich[:-1]
            else:
                this_runlength_lean = this_runlength_lean[:-1]

            # Some facts
            n_choice_rich = np.sum(this_half_choice == 1)  # In the sense of ground truth
            n_choice_lean = np.sum(this_half_choice == 0)

            if n_choice_rich * n_choice_lean == 0:
                this_choice_ratio = np.inf
            else:
                # In terms of ground-truth rich (could be smaller than 1, meaning that the animal chose the wrong arm)
                this_choice_ratio = n_choice_rich / n_choice_lean

                # Align everything to subjective rich (always larger than 1. I believe Hattori should have used this)
                if this_choice_ratio < 1:
                    this_choice_ratio = 1 / this_choice_ratio
                    this_runlength_rich, this_runlength_lean = this_runlength_lean, this_runlength_rich

            if this_block_p_base_ratio == 1:
                this_m_star = 1
            else:
                p_rich = max(p_reward[:, select_this_block[0]])
                p_lean = min(p_reward[:, select_this_block[0]])
                this_m_star = np.floor(np.log(1 - p_rich) / np.log(1 - p_lean))  # Ideal-p-hat-greed

            if len(this_runlength_rich) * len(this_runlength_lean) == 0: continue  # Exclude extreme bias block

            df_this_half = pd.Series(dict(m_star=this_m_star,
                                          p_base_ratio=this_block_p_base_ratio,
                                          choice_ratio=this_choice_ratio,
                                          mean_runlength_rich=np.mean(this_runlength_rich),
                                          mean_runlength_lean=np.mean(this_runlength_lean),
                                          trial_num=len(this_half_choice)))

            df_run_length_Lau[pp] = df_run_length_Lau[pp].append(df_this_half, ignore_index=True)

    return df_run_length_Lau


#  Compute mean_runlength_Bernoulli
X = np.linspace(1, 16, 50)
mean_runlength_Bernoulli = np.zeros([3, len(X)])
n = np.r_[1:100]

mean_runlength_Bernoulli[2, :] = X

for i, x in enumerate(X):
    p = x / (x + 1)
    mean_runlength_Bernoulli[0, i] = np.sum(n * ((1 - p) ** (n - 1)) * p)  # Lean
    mean_runlength_Bernoulli[1, i] = np.sum(n * (p ** (n - 1)) * (1 - p))  # Rich


def plot_runlength_Lau2005(df_run_length_Lau, block_partitions=['unknown', 'unknown']):

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8), dpi=150)
    gs = GridSpec(2, 3, hspace=0.6, wspace=0.4,
                  left=0.1, right=0.95, bottom=0.14, top=0.85)

    annotations = ['First %g%% trials' % block_partitions[0], 'Last %g%% trials' % block_partitions[1]]

    for pp, df_this_half in enumerate(df_run_length_Lau):

        df_this_half = df_this_half[~ df_this_half.isin([0, np.inf, -np.inf]).choice_ratio]
        df_this_half = df_this_half[~ df_this_half.isin([0, np.inf, -np.inf]).p_base_ratio]

        # == Fig.5 Lau 2005 ==
        ax = fig.add_subplot(gs[pp, 0])

        plt.plot(df_this_half.choice_ratio, df_this_half.mean_runlength_rich, 'go', label='Rich', alpha=0.7,
                 markersize=5)
        plt.plot(df_this_half.choice_ratio, df_this_half.mean_runlength_lean, 'rx', label='Lean', alpha=0.7,
                 markersize=8)

        ax.axhline(1, c='r', ls='--', lw=1)
        plt.plot([1, 16], [1, 16], c='g', ls='--', lw=1)
        # ax.axvline(1, c = 'k', ls = '--', lw = 1)

        plt.plot(mean_runlength_Bernoulli[2, :], mean_runlength_Bernoulli[0, :], 'k--', lw=1)
        plt.plot(mean_runlength_Bernoulli[2, :], mean_runlength_Bernoulli[1, :], 'k-', lw=1)

        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels([1, 2, 4, 8, 16])
        ax.set_xlim([0.9, 16])
        ax.set_yscale('log')
        ax.set_yticks([1, 2, 4, 8, 16])
        ax.set_yticklabels([1, 2, 4, 8, 16])
        ax.set_ylim([0.9, 16])

        # ax.axis('equal')

        plt.xlabel('Choice ratio (#rich / #lean)')
        plt.ylabel('Mean runlength')
        plt.legend()
        plt.title(annotations[pp])

        # == Mean rich runlength VS optimal rich runlength (m*) ==
        ax = fig.add_subplot(gs[pp, 1])

        x = df_this_half.m_star
        y = df_this_half.mean_runlength_rich

        sns.regplot(x=x, y=y, ax=ax)

        try:
            (r, p) = stats.pearsonr(x, y)
            plt.annotate('r = %.3g\np = %.3g' % (r, p), xy=(0, 0.8), xycoords=ax.transAxes, fontsize=9)
        except:
            pass

        plt.plot([0, 15], [0, 15], 'b--', lw=1)
        plt.xlabel('Optimal rich runlength')
        plt.ylabel('Mean rich runlength')
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 15])

        # == Choice ratio VS optimal rich runlength (m*) ==
        ax = fig.add_subplot(gs[pp, 2])
        x = df_this_half.m_star
        y = df_this_half.choice_ratio

        try:
            (r, p) = stats.pearsonr(x, y)
            plt.annotate('r = %.3g\np = %.3g' % (r, p), xy=(0, 0.8), xycoords=ax.transAxes, fontsize=9)
        except:
            pass

        sns.regplot(x=x, y=y, ax=ax)

        plt.plot([0, 15], [0, 15], 'b--', lw=1)
        plt.xlabel('Optimal rich runlength')
        plt.ylabel('Choice ratio (#rich / #lean)')
        ax.set_xlim([0, 15])
        ax.set_ylim([0, 15])

    return fig


def plot_block_switch(result_path="..\\results\\model_comparison\\w_bias_8\\",
                      group_results_name='group_results_all_with_bias.npz',
                      min_p_change=0, session_partitions=[1 / 3, 1 / 3, 1 / 3],
                      efficiency_partitions=[1 / 3, 1 / 3, 1 / 3]):
    sns.set(context='notebook')

    # == Load dataframe ==
    data = np.load(result_path + group_results_name, allow_pickle=True)
    group_results = data.f.group_results.item()
    results_all_mice = group_results['results_all_mice']
    df_block_switch = group_results['df_block_switch']
    df_block_switch = df_block_switch[df_block_switch.change_p_R >= min_p_change]
    prev_align = group_results['block_switch_para']['prev_align']

    palette_all = sns.color_palette("RdYlGn", max(results_all_mice['session_number'].unique()))

    # == Reproduce Marton's figure ==
    fig = plt.figure(figsize=(8, 3))
    fig.suptitle('All mice, all blocks, min_p_change = %g' % (min_p_change), fontsize=10)

    # 0. 
    ax1 = fig.add_subplot(121)
    choice_matrix = np.vstack(df_block_switch[(df_block_switch.mice == 'FOR06') & \
                                              (df_block_switch.session_num == 29)].choice_matrix)
    plot_choice_matrix([choice_matrix], prev_align, ax=ax1)
    plt.show()

    # 1. All mice, all blocks
    ax1 = fig.add_subplot(121)
    choice_matrix = np.vstack(df_block_switch.choice_matrix)
    plot_choice_matrix([choice_matrix], prev_align, ax=ax1)
    ax1.set_title('all')

    ax2 = fig.add_subplot(122)
    choice_matrix = np.vstack(df_block_switch.choice_norm_matrix)
    plot_choice_matrix([choice_matrix], prev_align, ax=ax2)
    ax2.set_title('all_norm')
    ax2.set_ylim([-0.5, 1.5])

    plt.subplots_adjust(top=0.8)
    plt.show()

    # 2. Each mice, all blocks
    fig = plt.figure(figsize=(9, 8), dpi=150)
    fig.suptitle('Each mouse, all blocks, min_p_change = %g' % (min_p_change), fontsize=16)
    n_mice = len(df_block_switch.mice.unique())
    gs = GridSpec(int(np.ceil(n_mice / 4)), 4, hspace=.6, wspace=0.5,
                  left=0.1, right=0.95, bottom=0.05, top=0.9)

    # Raw
    for mm, mouse in enumerate(df_block_switch.mice.unique()):
        ax = fig.add_subplot(gs[mm])

        choice_matrix = np.vstack(df_block_switch[df_block_switch.mice == mouse].choice_matrix)
        plot_choice_matrix([choice_matrix], prev_align, ax=ax)
        plt.title(mouse)

    fig = plt.figure(figsize=(9, 8), dpi=150)
    fig.suptitle('Each mouse, all blocks, min_p_change = %g' % (min_p_change), fontsize=16)
    n_mice = len(df_block_switch.mice.unique())
    gs = GridSpec(int(np.ceil(n_mice / 4)), 4, hspace=.6, wspace=0.5,
                  left=0.1, right=0.95, bottom=0.05, top=0.9)

    # Raw
    for mm, mouse in enumerate(df_block_switch.mice.unique()):
        ax = fig.add_subplot(gs[mm])

        choice_matrix = np.vstack(df_block_switch[df_block_switch.mice == mouse].choice_matrix)
        plot_choice_matrix([choice_matrix], prev_align, ax=ax)
        plt.title(mouse)
        ax.set_xlim([-10, 20])

    # Norm
    # fig = plt.figure(figsize=(9, 8), dpi = 150)
    # n_mice = len(df_block_switch.mice.unique())
    # gs = GridSpec(int(np.ceil(n_mice/4)), 4, hspace = .6, wspace = 0.5, 
    #               left = 0.1, right = 0.95, bottom = 0.05, top = 0.95)

    # for mm, mouse in enumerate(df_block_switch.mice.unique()):
    #     ax = fig.add_subplot(gs[mm]) 

    #     choice_matrix = np.vstack(df_block_switch[df_block_switch.mice == mouse].choice_norm_matrix)
    #     plot_choice_matrix(choice_matrix, prev_align, ax = ax, color = 'b', style = '-')
    #     plt.title(mouse)
    #     ax.set_ylim([-0.5, 1.5])

    #  3. Each mice, training progress
    fig = plt.figure(figsize=(9, 8), dpi=200)
    fig.suptitle(
        'Each mouse, training sessions %s, min_p_change = %g' % (np.round(session_partitions, 3), min_p_change),
        fontsize=16)
    n_mice = len(df_block_switch.mice.unique())
    gs = GridSpec(int(np.ceil(n_mice / 4)), 4, hspace=.6, wspace=0.5,
                  left=0.1, right=0.95, bottom=0.05, top=0.9)

    for mm, mouse in enumerate(df_block_switch.mice.unique()):
        ax = fig.add_subplot(gs[mm])
        unique_session_this = np.unique(df_block_switch[df_block_switch.mice == mouse].session_num)

        # Partition data into different choice_matrices
        session_end = np.quantile(unique_session_this, np.cumsum(session_partitions))
        session_start = np.hstack([0, session_end[:-1]])
        choice_matrix_list = []
        for s_start, s_end in zip(session_start, session_end):
            valid_block = (df_block_switch.mice == mouse) & (s_start < df_block_switch.session_num) & (
                        df_block_switch.session_num <= s_end)
            choice_matrix_list.append(np.vstack(df_block_switch[valid_block].choice_matrix))

        # Plotting
        plot_choice_matrix(choice_matrix_list, prev_align, ax=ax, color_list=['r', 'y', 'g'],
                           style_list=['-'] * len(session_partitions), error_bar=False)
        plt.title(mouse)

    plt.show()

    # 4. Each mice, foraging efficiency
    fig = plt.figure(figsize=(9, 8), dpi=200)
    fig.suptitle(
        'Each mouse, foraging efficiency %s, min_p_change = %g' % (np.round(efficiency_partitions, 3), min_p_change),
        fontsize=16)
    n_mice = len(df_block_switch.mice.unique())
    gs = GridSpec(int(np.ceil(n_mice / 4)), 4, hspace=.6, wspace=0.5,
                  left=0.1, right=0.95, bottom=0.05, top=0.9)

    for mm, mouse in enumerate(df_block_switch.mice.unique()):
        ax = fig.add_subplot(gs[mm])

        # Get efficiency from the other table
        all_efficiency_this = results_all_mice[results_all_mice.mice == mouse].foraging_efficiency

        # Partition data into different choice_matrices
        eff_end = np.quantile(all_efficiency_this, np.cumsum(efficiency_partitions))
        eff_start = np.hstack([0, eff_end[:-1]])
        choice_matrix_list = []

        for e_start, e_end in zip(eff_start, eff_end):
            valid_session = results_all_mice[(results_all_mice.mice == mouse) &
                                             (e_start < results_all_mice.foraging_efficiency) &
                                             (results_all_mice.foraging_efficiency <= e_end)].session_number
            valid_block = (df_block_switch.mice == mouse) & (df_block_switch.session_num.isin(valid_session))
            if np.sum(valid_block) == 0:
                choice_matrix_list.append([])
            else:
                choice_matrix_list.append(np.vstack(df_block_switch[valid_block].choice_matrix))

        # Plotting
        plot_choice_matrix(choice_matrix_list, prev_align, ax=ax, color_list=['r', 'y', 'g'],
                           style_list=['-'] * len(efficiency_partitions), error_bar=False)
        plt.title(mouse)
    plt.show()

    return


from scipy.optimize import curve_fit


def exp_func(x, b1, b2, tau):
    return b1 + (b2 - b1) * np.exp(-x / tau)


def plot_choice_matrix(choice_matrix_list, prev_align, ax=None, color_list=['b'], style_list=['-'], error_bar=True,
                       if_curve_fit='exp'):
    ''' For block switch '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    x = np.arange(choice_matrix_list[0].shape[1]) - prev_align

    for choice_matrix, color, style in zip(choice_matrix_list, color_list, style_list):
        if choice_matrix == []: continue

        mean = np.nanmean(choice_matrix, axis=0)
        ns = np.sum(~np.isnan(choice_matrix), axis=0)

        ax.plot(x, mean, color=color, linestyle=style, lw=1)

        if error_bar:
            sem = np.nanstd(choice_matrix, axis=0) / np.sqrt(ns)
            ax.fill_between(x, mean + sem, mean - sem, color=color, alpha=0.5)

        # Curve fitting
        if if_curve_fit == 'exp':
            xx = np.arange(0, 50)
            yy = mean[prev_align:prev_align + 50]
            popt, pcov = curve_fit(exp_func, xx, yy)
            plt.plot(xx, exp_func(xx, *popt), color + '--', lw=2, label='n = %g, $\\tau$ = %5.3f' % (max(ns), popt[-1]))

    ax.legend(fontsize=5)

    ax.set_xlim([-30, 50])
    ax.axvline(0, c='k', ls='--', lw=1)
    ax.axhline(0, c='k', ls='--', lw=1)
    ax.axhline(1, c='k', ls='--', lw=1)

    return


def plot_LL_surface(forager, LLsurfaces, CI_cutoff_LPTs, para_names, para_2ds, para_grids, para_scales, true_para, fitted_para, fit_history, fit_method, n_trials):
    
    sns.reset_orig()
            
    n_para_2ds = len(para_2ds)
    
    # ==== Figure setting ===
    nn_ax = min(3, n_para_2ds) # Column number
    mm_ax = np.ceil(n_para_2ds/nn_ax).astype(int)
    fig = plt.figure(figsize=(2.5+nn_ax*5, 1.5+mm_ax*5), dpi = 100)
    gs = GridSpec(mm_ax, nn_ax, wspace=0.2, hspace=0.35, bottom=0.1, top=0.84, left=0.07, right=0.97) 
    fig.text(0.05,0.88,'Likelihood Per Trial = p(data|paras, model)^(1/T): %s,\n Method: %s, N_trials = %g\n  True values: %s\nFitted values: %s' % (forager, fit_method, n_trials, 
                                                                                                                            np.round(true_para,3), np.round(fitted_para,3)),fontsize = 13)

    # ==== Plot each LL surface ===
    for ppp,(LLs, CI_cutoff_LPT, ps, para_2d) in enumerate(zip(LLsurfaces, CI_cutoff_LPTs, para_grids, para_2ds)):
    
        ax = fig.add_subplot(gs[np.floor(ppp/nn_ax).astype(int), np.mod(ppp,nn_ax).astype(int)]) 
        
        fitted_para_this = [fitted_para[para_2d[0]], fitted_para[para_2d[1]]]
        true_para_this = [true_para[para_2d[0]], true_para[para_2d[1]]]     
        para_names_this = [para_names[para_2d[0]], para_names[para_2d[1]]]
        para_scale = [para_scales[para_2d[0]], para_scales[para_2d[1]]]
                            
        for ii in range(2):
            if para_scale[ii] == 'log':
                ps[ii] = np.log10(ps[ii])
                fitted_para_this[ii] = np.log10(fitted_para_this[ii])
                true_para_this[ii] = np.log10(true_para_this[ii])
        
        dx = ps[0][1]-ps[0][0]
        dy = ps[1][1]-ps[1][0]
        extent=[ps[0].min()-dx/2, ps[0].max()+dx/2, ps[1].min()-dy/2, ps[1].max()+dy/2]

        # -- Gaussian filtering ---

        if dx > 0 and dy > 0:
            plt.imshow(LLs, cmap='plasma', extent=extent, interpolation='none', origin='lower')
            plt.colorbar()
        # plt.pcolor(pp1, pp2, LLs, cmap='RdBu', vmin=z_min, vmax=z_max)
        
        plt.contour(LLs, colors='grey', levels = 20, extent=extent, linewidths=0.7)
        # plt.contour(-np.log(-LLs), colors='grey', levels = 20, extent=extent, linewidths=0.7)
        
        # -- Cutoff LPT --
        plt.contour(LLs, levels = [CI_cutoff_LPT], colors = 'r', extent=extent)
        
        # ==== True value ==== 
        plt.plot(true_para_this[0], true_para_this[1],'ob', markersize = 20, markeredgewidth=3, fillstyle='none')
        
        # ==== Fitting history (may have many) ==== 
        if fit_history != []:
            
            # Compatible with one history (global optimizers) or multiple histories (local optimizers)
            for nn, hh in reversed(list(enumerate(fit_history))):  
                hh = np.array(hh)
                hh = hh[:,(para_2d[0], para_2d[1])]  # The user-defined 2-d subspace
                
                for ii in range(2):
                    if para_scale[ii] == 'log': hh[:,ii] = np.log10(hh[:,ii])
                
                sizes = 100 * np.linspace(0.1,1,np.shape(hh)[0])
                plt.scatter(hh[:,0], hh[:,1], s = sizes, c = 'k' if nn == 0 else None)
                plt.plot(hh[:,0], hh[:,1], '-' if nn == 0 else ':', color = 'k' if nn == 0 else None)
                
        # ==== Final fitted result ====
        plt.plot(fitted_para_this[0], fitted_para_this[1],'Xb', markersize=17)
    
        ax.set_aspect(1.0/ax.get_data_ratio()) 
        
        plt.xlabel(('log10 ' if para_scale[0] == 'log' else '') + para_names_this[0])
        plt.ylabel(('log10 ' if para_scale[1] == 'log' else '') + para_names_this[1])
        
    
    plt.show()
    
def plot_session_lightweight(fake_data, fitted_data = None, smooth_factor = 5, base_color = 'y'):
    # sns.reset_orig()
    sns.set(style="ticks", context="paper", font_scale=1.4)

    choice_history, reward_history, p_reward = fake_data
    
    # == Fetch data ==
    n_trials = np.shape(choice_history)[1]
    
    p_reward_fraction = p_reward[1,:] / (np.sum(p_reward, axis = 0))
                                      
    rewarded_trials = np.any(reward_history, axis = 0)
    unrewarded_trials = np.logical_not(rewarded_trials)
    
    # == Choice trace ==
    fig = plt.figure(figsize=(9, 4), dpi = 150)
        
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left = 0.1, right=0.8)

    # Rewarded trials
    ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0,rewarded_trials]-0.5) * 1.4, 
            'k|',color='black',markersize=20, markeredgewidth=2)

    # Unrewarded trials
    ax.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[0,unrewarded_trials] - 0.5) * 1.4, 
            '|',color='gray', markersize=10, markeredgewidth=1)
    
    # Base probability
    ax.plot(np.arange(0, n_trials), p_reward_fraction, color=base_color, label = 'base rew. prob.', lw = 1.5)
    
    # Smoothed choice history
    y = moving_average(choice_history, smooth_factor)
    x = np.arange(0, len(y)) + int(smooth_factor/2)
    ax.plot(x, y, linewidth = 1.5, color='black', label = 'choice (smooth = %g)' % smooth_factor)
    
    # For each session, if any
    if fitted_data is not None:
        ax.plot(np.arange(0, n_trials), fitted_data[1,:], linewidth = 1.5, label = 'model') 

    ax.legend(fontsize = 10, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)
     
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Left','Right'])
    # ax.set_xlim(0,300)
    
    # fig.tight_layout() 
    sns.despine(trim=True)

    return ax
    

def plot_model_comparison_predictive_choice_prob(model_comparison, smooth_factor = 5):
    # sns.reset_orig()
    
    choice_history, reward_history, p_reward, trial_numbers = model_comparison.fit_choice_history, model_comparison.fit_reward_history, model_comparison.p_reward, model_comparison.trial_numbers
    if not hasattr(model_comparison,'plot_predictive'):
        model_comparison.plot_predictive = [1,2,3]
        
    
        
    n_trials = np.shape(choice_history)[1]

    ax = plot_session_lightweight([choice_history, reward_history, p_reward], smooth_factor = smooth_factor)
    # Predictive choice prob
    for bb in model_comparison.plot_predictive:
        bb = bb - 1
        if bb < len(model_comparison.results):
            this_id = model_comparison.results_sort.index[bb] - 1
            this_choice_prob = model_comparison.results_raw[this_id].predictive_choice_prob
            this_result = model_comparison.results_sort.iloc[bb]
           
            ax.plot(np.arange(0, n_trials), this_choice_prob[1,:] , linewidth = max(1.5-0.3*bb,0.2), 
                    label = 'Model %g: %s, Km = %g\n%s\n%s' % (bb+1, this_result.model, this_result.Km, 
                                                                                        this_result.para_notation, this_result.para_fitted))
    
    # Plot session starts
    if len(trial_numbers) > 1:  # More than one sessions
        for session_start in np.cumsum([0, *trial_numbers[:-1]]):
            plt.axvline(session_start, color='b', linestyle='--', linewidth = 2)
            try:
                plt.text(session_start + 1, 1, '%g' % model_comparison.session_num[session_start], fontsize = 10, color='b')
            except:
                pass

    ax.legend(fontsize = 7, loc=1, bbox_to_anchor=(0.985, 0.89), bbox_transform=plt.gcf().transFigure)
     
    # ax.set_xlim(0,300)
    
    # fig.tight_layout() 
    sns.set()
    return


def plot_model_comparison_result(model_comparison):
    sns.set()
    
    results = model_comparison.results
    
    # Update notations
    para_notation_with_best_fit = []
    for i, row in results.iterrows():
        para_notation_with_best_fit.append('('+str(i)+') '+row.para_notation + '\n' + str(np.round(row.para_fitted,2)))
        
    results['para_notation_with_best_fit'] = para_notation_with_best_fit
        
    fig = plt.figure(figsize=(12, 8), dpi = 150)
    gs = GridSpec(1, 5, wspace = 0.1, bottom = 0.1, top = 0.85, left = 0.23, right = 0.95)
    
    
    # -- 1. LPT -- 
    ax = fig.add_subplot(gs[0, 0])
    s = sns.barplot(x = 'LPT', y = 'para_notation_with_best_fit', data = results, color = 'grey')
    s.set_xlim(min(0.5,np.min(np.min(model_comparison.results[['LPT_AIC','LPT_BIC']]))) - 0.005)
    plt.axvline(0.5, color='k', linestyle='--')
    s.set_ylabel('')
    s.set_xlabel('Likelihood per trial')

    # -- 2. AIC, BIC raw --
    ax = fig.add_subplot(gs[0, 1])
    df = pd.melt(results[['para_notation_with_best_fit','AIC','BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'IC')
    s = sns.barplot(x = 'IC', y = 'para_notation_with_best_fit', hue = '', data = df)

    # Annotation
    x_max = max(plt.xlim())
    ylim = plt.ylim()
    best_AIC = np.where(results.best_model_AIC)[0][0]
    plt.plot(x_max, best_AIC - 0.2, '*', markersize = 15)
    best_BIC = np.where(results.best_model_BIC)[0][0]
    plt.plot(x_max, best_BIC + 0.2, '*', markersize = 15)
    plt.ylim(ylim)
    
    s.set_yticklabels('')
    s.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol = 1)
    s.set_ylabel('')
    s.set_xlabel('AIC or BIC')

    # -- 3. log10_BayesFactor --
    ax = fig.add_subplot(gs[0, 2])
    df = pd.melt(results[['para_notation_with_best_fit','log10_BF_AIC','log10_BF_BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'log10 (Bayes factor)')
    s = sns.barplot(x = 'log10 (Bayes factor)', y = 'para_notation_with_best_fit', hue = '', data = df)
    h_d = plt.axvline(-2, color='r', linestyle='--', label = 'decisive')
    s.legend(handles = [h_d,], bbox_to_anchor=(0,1.02,1,0.2), loc='lower left')
    # s.invert_xaxis()
    s.set_xlabel('log$_{10}\\frac{p(model)}{p(best\,model)}$')
    s.set_ylabel('')
    s.set_yticklabels('')
    
    # -- 4. Model weight --
    ax = fig.add_subplot(gs[0, 3])
    df = pd.melt(results[['para_notation_with_best_fit','model_weight_AIC','model_weight_BIC']], 
                 id_vars = 'para_notation_with_best_fit', var_name = '', value_name= 'Model weight')
    s = sns.barplot(x = 'Model weight', y = 'para_notation_with_best_fit', hue = '', data = df)
    ax.legend_.remove()
    plt.xlim([0,1.05])
    plt.axvline(1, color='k', linestyle='--')
    s.set_ylabel('')
    s.set_yticklabels('')
    
    # -- 5. Prediction accuracy --
    
    prediction_accuracy_NONCV = np.zeros(len(results))
    
    for rr, raw in enumerate(model_comparison.results_raw):
        this_predictive_choice_prob = raw.predictive_choice_prob
        this_predictive_choice = np.argmax(this_predictive_choice_prob, axis = 0)
        prediction_accuracy_NONCV[rr] = np.sum(this_predictive_choice == model_comparison.fit_choice_history) / model_comparison.n_trials
        
    results['prediction_accuracy_NONCV'] = prediction_accuracy_NONCV * 100

    ax = fig.add_subplot(gs[0, 4])
    s = sns.barplot(x = 'prediction_accuracy_NONCV', y = 'para_notation_with_best_fit', data = results, color = 'grey')
    plt.axvline(50, color='k', linestyle='--')
    ax.set_xlim(min(50,np.min(np.min(model_comparison.results[['prediction_accuracy_NONCV']]))) - 5)
    ax.set_ylabel('')
    ax.set_xlabel('Prediction_accuracy_NONCV')
    s.set_yticklabels('')

    return

def plot_confusion_matrix(confusion_results, order = None):
    sns.set()
    
    n_runs = np.sum(~np.isnan(confusion_results['raw_AIC'][0,0,:]))
    n_trials = confusion_results['n_trials']
    
    # === Get notations ==
    model_notations = confusion_results['models_notations']
    
    # Reorder if needed
    if order is not None:
        model_notations = ['('+str(ii+1)+') '+model_notations[imodel] for ii,imodel in enumerate(order)]
    else:
        model_notations = ['('+str(ii+1)+') '+ mm for ii,mm in enumerate(model_notations)]
        
    # === Plotting ==
    contents = [
               [['confusion_best_model_AIC','inversion_best_model_AIC'],
               ['confusion_best_model_BIC','inversion_best_model_BIC']],
               [['confusion_log10_BF_AIC','confusion_AIC'],
               ['confusion_log10_BF_BIC','confusion_BIC']],
               ]

    for cc, content in enumerate(contents):    
        fig = plt.figure(figsize=(10, 8.5))
        fig.text(0.05,0.97,'Model Recovery: n_trials = %g, n_runs = %g, True →, Fitted ↓' % (n_trials, n_runs))

        gs = GridSpec(2, 2, wspace=0.15, hspace=0.1, bottom=0.02, top=0.8, left=0.15, right=0.97)
    
        for ii in range(2):
            for jj in range(2):
                
                # Get data
                data = confusion_results[content[ii][jj]]
                
                # Reorder
                if order is not None:
                    data = data[:, np.array(order) - 1]
                    data = data[np.array(order) - 1, :]
                    
                # -- Plot --
                ax = fig.add_subplot(gs[ii, jj])
                
                # I transpose the data here so that the columns are ground truth and the rows are fitted results,
                # which is better aligned to the model comparison plot and the model-comparison-as-a-function-of-session in the real data.
                if cc == 0:
                    sns.heatmap(data.T, annot = True, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10})
                else:
                    if jj == 0:
                        sns.heatmap(data.T, annot = True, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10}, vmin=-2, vmax=0)
                    else:
                        sns.heatmap(data.T, annot = False, fmt=".2g", ax = ax, square = True, annot_kws={"size": 10})
                        
                set_label(ax, ii,jj, model_notations)
                plt.title(content[ii][jj])
        
        fig.show()
        

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
             











