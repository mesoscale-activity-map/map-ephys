# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:47:05 2020

@author: Han
"""
import pdb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from statannot import add_stat_annotation
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from pipeline import lab, foraging_model, util
from ..model.util import moving_average


# plt.rcParams.update({'font.size': 14, 'figure.dpi': 150})
# sns.set(context='talk')


def plot_session_model_comparison(sess_key={'subject_id': 473361, 'session': 47}, model_comparison_idx=0, sort=None):
    """
    Plot model comparison results of a specified session
    :param sess_key:
    :param model_comparison_idx: --> pipeline.foraging_model.ModelComparison
    :param sort: (None) 'aic', 'bic', 'cross_valid_accuracy_test' (descending)
    :return:
    """

    # -- Fetch data --
    results, q_model_comparison = _get_model_comparison_results(sess_key, model_comparison_idx, sort)
    best_aic_id, best_bic_id, best_cross_valid_id = (foraging_model.FittedSessionModelComparison.BestModel & q_model_comparison).fetch1(
        'best_aic', 'best_bic', 'best_cross_validation_test')

    # -- Plotting --
    with sns.plotting_context("notebook", font_scale=1), sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(15, 8 + len(results) / 7), dpi=150)
        gs = GridSpec(1, 5, wspace=0.1, bottom=0.11, top=0.85, left=0.33, right=0.98)
        fig.text(0.05, 0.9, f'{(lab.WaterRestriction & sess_key).fetch1("water_restriction_number")}, '
                            f'session {sess_key["session"]}, {results.n_trials[0]} trials\n'
                            f'Model comparison: {(foraging_model.ModelComparison & q_model_comparison).fetch1("desc")}'
                            f' (n = {len(results)})')

        # -- 1. LPT --
        ax = fig.add_subplot(gs[0, 0])
        s = sns.barplot(x='lpt', y='para_notation_with_best_fit', data=results, color='grey')
        s.set_xlim(min(0.5, np.min(np.min(results[['lpt_aic', 'lpt_bic']]))) - 0.005)
        plt.axvline(0.5, color='k', linestyle='--')
        s.set_ylabel('')
        s.set_xlabel('Likelihood per trial')

        # -- 2. aic, bic raw --
        ax = fig.add_subplot(gs[0, 1])
        df = pd.melt(results[['para_notation_with_best_fit', 'aic', 'bic']],
                     id_vars='para_notation_with_best_fit', var_name='', value_name='ic')
        s = sns.barplot(x='ic', y='para_notation_with_best_fit', hue='', data=df)

        # annotation
        x_max = max(plt.xlim())
        ylim = plt.ylim()
        plt.plot(x_max, results.index[results.model_id == best_aic_id] - 0.2, '*', markersize=15)
        plt.plot(x_max, results.index[results.model_id == best_bic_id] + 0.2, '*', markersize=15)
        plt.ylim(ylim)
        s.set_yticklabels('')
        s.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=1)
        s.set_ylabel('')
        s.set_xlabel('AIC or BIC')

        # -- 3. log10_bayesfactor --
        ax = fig.add_subplot(gs[0, 2])
        df = pd.melt(results[['para_notation_with_best_fit', 'log10_bf_aic', 'log10_bf_bic']],
                     id_vars='para_notation_with_best_fit', var_name='', value_name='log10 (bayes factor)')
        s = sns.barplot(x='log10 (bayes factor)', y='para_notation_with_best_fit', hue='', data=df)
        h_d = plt.axvline(-2, color='r', linestyle='--', label='decisive')
        s.legend(handles=[h_d, ], bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left')
        plt.ylim(ylim)
        # s.invert_xaxis()
        s.set_xlabel(r'log$_{10}\frac{p(model)}{p(best\,model)}$')
        s.set_ylabel('')
        s.set_yticklabels('')

        # -- 4. model weight --
        ax = fig.add_subplot(gs[0, 3])
        df = pd.melt(results[['para_notation_with_best_fit', 'model_weight_aic', 'model_weight_bic']],
                     id_vars='para_notation_with_best_fit', var_name='', value_name='model weight')
        s = sns.barplot(x='model weight', y='para_notation_with_best_fit', hue='', data=df)
        ax.legend_.remove()
        plt.xlim([-0.05, 1.05])
        plt.axvline(1, color='k', linestyle='--')
        s.set_xlabel('Model weight')
        s.set_ylabel('')
        s.set_yticklabels('')

        # -- 5. Prediction accuracy --
        results.cross_valid_accuracy_test *= 100
        ax = fig.add_subplot(gs[0, 4])
        s = sns.barplot(x='cross_valid_accuracy_test', y='para_notation_with_best_fit', data=results, color='grey')
        plt.axvline(50, color='k', linestyle='--')
        x_max = max(plt.xlim())
        plt.plot(x_max, results.index[results.model_id == best_cross_valid_id], '*', markersize=15, color='grey')
        ax.set_xlim(min(50, np.min(results.cross_valid_accuracy_test)) - 5)
        plt.ylim(ylim)
        ax.set_ylabel('')
        ax.set_xlabel('Prediction accuracy %\n(2-fold cross valid.)')
        s.set_yticklabels('')

    return


def plot_session_fitted_choice(sess_key={'subject_id': 473361, 'session': 47},
                               specified_model_ids=None,
                               model_comparison_idx=0, sort='aic',
                               first_n=1, last_n=0,
                               remove_ignored=True, smooth_factor=5,
                               ax=None):

    """
    Plot actual and fitted choice trace of a specified session
    :param sess_key: could across several sessions
    :param model_comparison_idx: model comparison group
    :param sort: {'aic', 'bic', 'cross_validation_test'}
    :param first_n: top best n competitors to plot
    :param last_n: last n competitors to plot
    :param specified_model_ids: if not None, override first_n and last_n
    :param smooth_factor: for actual data
    :return: axis
    """

    # Fetch actual data
    choice_history, reward_history, iti, p_reward, _ = foraging_model.get_session_history(sess_key, remove_ignored=remove_ignored)
    n_trials = np.shape(choice_history)[1]

    # Fetch fitted data
    if specified_model_ids is None:  # No specified model, plot model comparison
        results, q_model_comparison = _get_model_comparison_results(sess_key, model_comparison_idx, sort)
        results_to_plot = pd.concat([results.iloc[:first_n], results.iloc[len(results)-last_n:]])
    else:  # only plot specified model_id
        results_to_plot = results = _get_specified_model_fitting_results(sess_key, specified_model_ids)
        
    # -- Plot actual choice and reward history --
    with sns.plotting_context("notebook", font_scale=1, rc={'style': 'ticks'}):
        ax = plot_session_lightweight([choice_history, reward_history, p_reward], smooth_factor=smooth_factor, ax=ax)

        # -- Plot fitted choice probability etc. --
        model_str =  (f'Model comparison: {(foraging_model.ModelComparison & q_model_comparison).fetch1("desc")}'
                      f'(n = {len(results)})') if specified_model_ids is None else ''
        plt.gcf().text(0.05, 0.95, f'{(lab.WaterRestriction & sess_key).fetch1("water_restriction_number")}, '
                                    f'session {sess_key["session"]}, {results.n_trials[0]} trials\n' + model_str)
        
        for idx, result in results_to_plot.iterrows():
            trial, right_choice_prob = (foraging_model.FittedSessionModel.TrialLatentVariable
                                & dict(result) & 'water_port="right"').fetch('trial', 'choice_prob')
            ax.plot(np.arange(0, n_trials) if remove_ignored else trial, right_choice_prob, linewidth=max(1.5 - 0.3 * idx, 0.2),
                    label=f'{idx + 1}: <{result.model_id}>'
                            f'{result.model_notation}\n'
                            f'({result.fitted_param})')

        #TODO Plot session starts
        # if len(trial_numbers) > 1:  # More than one sessions
        #     for session_start in np.cumsum([0, *trial_numbers[:-1]]):
        #         plt.axvline(session_start, color='b', linestyle='--', linewidth=2)
        #         try:
        #             plt.text(session_start + 1, 1, '%g' % model_comparison.session_num[session_start], fontsize=10,
        #                      color='b')
        #         except:
        #             pass

    ax.legend(fontsize=5, loc='upper left', bbox_to_anchor=(1, 1.3))
    ax.text(0, 1.1, util._get_sess_info(sess_key), fontsize=10, transform=ax.transAxes)

    # fig.tight_layout()
    # sns.set()

    return ax


def plot_session_lightweight(data, fitted_data=None, smooth_factor=5, base_color='y', ax=None):
    # sns.reset_orig()

    with sns.plotting_context("notebook", font_scale=1):

        choice_history, reward_history, p_reward = data

        # == Fetch data ==
        n_trials = np.shape(choice_history)[1]

        p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))

        ignored_trials = np.isnan(choice_history[0])
        rewarded_trials = np.any(reward_history, axis=0)
        unrewarded_trials = np.logical_not(np.logical_or(rewarded_trials, ignored_trials))

        # == Choice trace ==
        fig = plt.figure(figsize=(8, 3), dpi=200)

        if ax is None:
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.1, right=0.8, bottom=0.05, top=0.7)

        # Rewarded trials
        ax.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0, rewarded_trials] - 0.5) * 1.4,
                '|', color='black', markersize=10, markeredgewidth=2)

        # Unrewarded trials
        ax.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[0, unrewarded_trials] - 0.5) * 1.4,
                '|', color='gray', markersize=6, markeredgewidth=1)

        # Ignored trials
        ax.plot(np.nonzero(ignored_trials)[0], [1.1] * sum(ignored_trials),
                'x', color='red', markersize=2, markeredgewidth=0.5)

        # Base probability
        ax.plot(np.arange(0, n_trials), p_reward_fraction, color=base_color, label='base rew. prob.', lw=1.5)

        # Smoothed choice history
        y = moving_average(choice_history, smooth_factor)
        x = np.arange(0, len(y)) + int(smooth_factor / 2)
        ax.plot(x, y, linewidth=1.5, color='black', label='choice (smooth = %g)' % smooth_factor)

        # For each session, if any
        if fitted_data is not None:
            ax.plot(np.arange(0, n_trials), fitted_data[1, :], linewidth=1.5, label='model')

        ax.legend(fontsize=5, loc='upper left', bbox_to_anchor=(1, 1.3))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left', 'Right'])
        # ax.set_xlim(0,300)

        # fig.tight_layout()
        sns.despine(trim=True)

    return ax


# ---- Helper funcs -----

def _get_model_comparison_results(sess_key, model_comparison_idx=0, sort=None):
    """
    Fetch relevent model comparison results of a specified session
    :param sess_key:
    :param model_comparison_idx:
    :param sort: 'aic', 'bic', 'cross_valid_accuracy_test', etc.
    :return: results in DataFrame, q_model_comparison
    """
    # Get all relevant models
    q_model_comparison = (foraging_model.FittedSessionModelComparison.RelativeStat
                          & sess_key & {'model_comparison_idx': model_comparison_idx})
    q_result = (q_model_comparison
                * foraging_model.Model.proj(..., '-n_params')
                * foraging_model.FittedSessionModel)

    # Add fitted params
    q_result *= q_result.aggr(foraging_model.FittedSessionModel.Param * foraging_model.Model.Param,
                              fitted_param='GROUP_CONCAT(ROUND(fitted_value,2) ORDER BY param_idx SEPARATOR ", ")')
    results = pd.DataFrame(q_result.fetch())
    results['para_notation_with_best_fit'] = [f'<{id}> {name}\n({value})' for id, name, value in
                                              results[['model_id', 'model_notation', 'fitted_param']].values]

    # Sort if necessary
    if sort:
        results.sort_values(by=[sort], ascending=sort != 'cross_valid_accuracy_test', ignore_index=True, inplace=True)

    return results, q_model_comparison


def  _get_specified_model_fitting_results(sess_key, specified_model_ids):
    model_ids_str = ', '.join(str(id) for id in np.array([specified_model_ids]).ravel())
    q_result = (foraging_model.Model.proj(..., '-n_params') * foraging_model.FittedSessionModel
                ) & sess_key & f'model_id in ({model_ids_str})'
    q_result *= q_result.aggr(foraging_model.FittedSessionModel.Param * foraging_model.ModelParam * foraging_model.Model.Param.proj('param_idx'),
                                    fitted_param='GROUP_CONCAT(ROUND(fitted_value,2) ORDER BY param_idx SEPARATOR ", ")')
    return pd.DataFrame(q_result.fetch())
