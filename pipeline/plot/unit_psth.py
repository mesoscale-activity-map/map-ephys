
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from pipeline import psth, psth_foraging, ephys, lab, ccf, histology, experiment, foraging_model
from pipeline.util import _get_trial_event_times, _get_ephys_trial_event_times, _get_units_hemisphere


_plt_xlim = [-3, 3]

def _plot_spike_raster(ipsi, contra, vlines=[], shade_bar=None, ax=None, title='', xlim=_plt_xlim):
    if not ax:
        fig, ax = plt.subplots(1, 1)

    ipsi_tr = ipsi['raster'][1]
    for i, tr in enumerate(set(ipsi['raster'][1])):
        ipsi_tr = np.where(ipsi['raster'][1] == tr, i, ipsi_tr)

    contra_tr = contra['raster'][1]
    for i, tr in enumerate(set(contra['raster'][1])):
        contra_tr = np.where(contra['raster'][1] == tr, i, contra_tr)

    ipsi_tr_max = ipsi_tr.max() if ipsi_tr.size > 0 else 0

    ax.plot(ipsi['raster'][0], ipsi_tr, 'r.', markersize=1)
    ax.plot(contra['raster'][0], contra_tr + ipsi_tr_max + 1, 'b.', markersize=1)

    for x in vlines:
        ax.axvline(x=x, linestyle='--', color='k')
    if shade_bar is not None:
        ax.axvspan(shade_bar[0], shade_bar[0] + shade_bar[1], alpha=0.3, color='royalblue')

    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_title(title)


def _plot_psth(ipsi, contra, vlines=[], shade_bar=None, ax=None, title='', xlim=_plt_xlim):
    if not ax:
        fig, ax = plt.subplots(1, 1)

    ax.plot(contra['psth'][1], contra['psth'][0], 'b')
    ax.plot(ipsi['psth'][1], ipsi['psth'][0], 'r')

    for x in vlines:
        ax.axvline(x=x, linestyle='--', color='k')
    if shade_bar is not None:
        ax.axvspan(shade_bar[0], shade_bar[0] + shade_bar[1], alpha=0.3, color='royalblue')

    ax.set_ylabel('spikes/s')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(xlim)
    ax.set_xlabel('Time (s)')
    ax.set_title(title)


def plot_unit_psth(unit_key, axs=None, title='', xlim=_plt_xlim):
    """
    Default raster and PSTH plot for a specified unit - only {good, no early lick, correct trials} selected
    condition_name_kw: list of keywords to match for the TrialCondition name
    """

    hemi = _get_units_hemisphere(unit_key)

    ipsi_hit_unit_psth = psth.UnitPsth.get_plotting_data(
        unit_key, {'trial_condition_name': f'good_noearlylick_{"left" if hemi == "left" else "right"}_hit'})

    contra_hit_unit_psth = psth.UnitPsth.get_plotting_data(
        unit_key, {'trial_condition_name':  f'good_noearlylick_{"right" if hemi == "left" else "left"}_hit'})

    ipsi_miss_unit_psth = psth.UnitPsth.get_plotting_data(
        unit_key, {'trial_condition_name': f'good_noearlylick_{"left" if hemi == "left" else "right"}_miss'})

    contra_miss_unit_psth = psth.UnitPsth.get_plotting_data(
        unit_key, {'trial_condition_name':  f'good_noearlylick_{"right" if hemi == "left" else "left"}_miss'})

    # get event start times: sample, delay, response
    periods, period_starts = _get_trial_event_times(['sample', 'delay', 'go'], unit_key, 'good_noearlylick_hit')

    fig = None
    if axs is None:
        fig, axs = plt.subplots(2, 2)

    # correct response
    _plot_spike_raster(ipsi_hit_unit_psth, contra_hit_unit_psth, ax=axs[0, 0],
                       vlines=period_starts,
                       title=title if title else f'Unit #: {unit_key["unit"]}\nCorrect Response', xlim=xlim)
    _plot_psth(ipsi_hit_unit_psth, contra_hit_unit_psth,
               vlines=period_starts, ax=axs[1, 0], xlim=xlim)

    # incorrect response
    _plot_spike_raster(ipsi_miss_unit_psth, contra_miss_unit_psth, ax=axs[0, 1],
                       vlines=period_starts,
                       title=title if title else f'Unit #: {unit_key["unit"]}\nIncorrect Response', xlim=xlim)
    _plot_psth(ipsi_miss_unit_psth, contra_miss_unit_psth,
               vlines=period_starts, ax=axs[1, 1], xlim=xlim)

    return fig

    
def _plot_spike_raster_foraging(ipsi, contra, offset=0, vlines=[], shade_bar=None, ax=None, title='', xlim=_plt_xlim):
    if not ax:
        fig, ax = plt.subplots(1, 1)
        
    contra_tr = contra['raster'][1]
    for i, tr in enumerate(contra['trials']):
        contra_tr = np.where(contra['raster'][1] == tr, i, contra_tr)

    ipsi_tr = ipsi['raster'][1]
    for i, tr in enumerate(ipsi['trials']):
        ipsi_tr = np.where(ipsi['raster'][1] == tr, i, ipsi_tr)
    
    contra_tr_max = contra_tr.max() if contra_tr.size > 0 else 0

    start_at = offset
    ax.plot(contra['raster'][0], contra_tr + start_at + 1, 'b.', markersize=1)
    ax.axhline(y=start_at, linestyle='-', color='k')
    start_at += contra_tr_max

    ax.plot(ipsi['raster'][0], ipsi_tr + start_at, 'r.', markersize=1)
    ax.axhline(y=start_at, linestyle='-', color='k')

    for x in vlines:
        ax.axvline(x=x, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='-', color='k', lw=1.5)

    if shade_bar is not None:
        ax.axvspan(shade_bar[0], shade_bar[0] + shade_bar[1], alpha=0.3, color='royalblue')

    # ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_title(title)


def _plot_psth_foraging(ipsi, contra, vlines=[], shade_bar=None, ax=None, title='', label='', xlim=_plt_xlim, **karg):
    if not ax:
        fig, ax = plt.subplots(1, 1)

    ax.plot(contra['bins'], contra['psth'], 'b', label='contra ' + label, **karg)
    ax.plot(ipsi['bins'], ipsi['psth'], 'r', label='ipsi ' + label, **karg)

    for x in vlines:
        ax.axvline(x=x, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='-', color='k', lw=1.5)
        
    if shade_bar is not None:
        ax.axvspan(shade_bar[0], shade_bar[0] + shade_bar[1], alpha=0.3, color='royalblue')

    ax.set_ylabel('spikes/s')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(xlim)
    ax.set_xlabel('Time (s)')
    ax.set_title(title)


def _plot_psths(psths, kargs=[], vlines=[], shade_bar=None, ax=None, title='', label='', xlim=_plt_xlim):
    """
    Plot arbitrary number of psths
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)

    if not kargs:
        kargs = [{'color': 'b'}] * len(psths)

    for psth, karg in zip(psths, kargs):
        ax.plot(psth['bins'], psth['psth'], **karg)

    for x in vlines:
        ax.axvline(x=x, linestyle='--', color='k')
    ax.axvline(x=0, linestyle='-', color='k', lw=1.5)

    if shade_bar is not None:
        ax.axvspan(shade_bar[0], shade_bar[0] + shade_bar[1], alpha=0.3, color='royalblue')

    ax.set_ylabel('spikes/s')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(xlim)
    ax.set_xlabel('Time (s)')
    ax.set_title(title)


def _set_same_horizonal_aspect_ratio(axs, xlims, gap=0.02):
    """
    Scale axis widths to keep the same horizonal aspect ratio across axs
    assuming axs are already from left to right
    """
    n = len(axs)
    leftmost, b, _, h = axs[0].get_position().bounds
    rightmost = np.array(axs[-1].get_position().bounds)[[0, 2]].sum()  # left + width
    spans = np.array([t_max - t_min for (t_min, t_max) in xlims])
    scaled_widths = (rightmost - leftmost - gap * (n-1)) / sum(spans) * spans
    scaled_lefts = leftmost + np.cumsum([0] + list(scaled_widths[:-1])) + gap * np.arange(n)

    for ax, l, w in zip(axs, scaled_lefts, scaled_widths):
        ax.set_position([l, b, w, h])


def plot_unit_psth_choice_outcome(unit_key,
                                  align_types=['trial_start', 'go_cue', 'first_lick_after_go_cue', 'iti_start', 'next_trial_start'],
                                  if_raster=True, if_exclude_early_lick=False,
                                  axs=None, title=''):
    """
    (for foraging task) Plot psth grouped by (choice x outcome)
    """
    
    # for (the very few) sessions without zaber feedback signal, use 'bitcodestart' with manual correction (see compute_unit_psth_and_raster)
    if not ephys.TrialEvent & unit_key & 'trial_event_type = "zaberready"':
        align_types = [a + '_bitcode' if 'trial_start' in a else a for a in align_types]

    hemi = _get_units_hemisphere(unit_key)
    ipsi = "L" if hemi == "left" else "R"
    contra = "R" if hemi == "left" else "L"
    no_early_lick = '_noearlylick' if if_exclude_early_lick else ''

    fig = None
    if axs is None:
        fig = plt.figure(figsize=(len(align_types)/5 * 25, (1+if_raster)/2 * 9))
        axs = fig.subplots(1 + if_raster, len(align_types), sharey='row', sharex='col')
        axs = np.atleast_2d(axs).reshape((1+if_raster, -1))
        plt.subplots_adjust(top=0.8)

    xlims = []

    for ax_i, align_type in enumerate(align_types):
        
        offset, xlim = (psth_foraging.AlignType & {'align_type_name': align_type}).fetch1('trial_offset', 'xlim')
        xlims.append(xlim)
        
        # align_trial_offset is added on the get_trials, which effectively 
        # makes the psth conditioned on the previous {align_trial_offset} trials
        ipsi_hit_trials = psth_foraging.TrialCondition.get_trials(f'{ipsi}_hit{no_early_lick}', offset) & unit_key
        ipsi_hit_unit_psth = psth_foraging.compute_unit_psth_and_raster(unit_key, ipsi_hit_trials, align_type)

        contra_hit_trials = psth_foraging.TrialCondition.get_trials(f'{contra}_hit{no_early_lick}', offset) & unit_key
        contra_hit_unit_psth = psth_foraging.compute_unit_psth_and_raster(unit_key, contra_hit_trials, align_type)

        ipsi_miss_trials = psth_foraging.TrialCondition.get_trials(f'{ipsi}_miss{no_early_lick}', offset) & unit_key
        ipsi_miss_unit_psth = psth_foraging.compute_unit_psth_and_raster(unit_key, ipsi_miss_trials, align_type)

        contra_miss_trials = psth_foraging.TrialCondition.get_trials(f'{contra}_miss{no_early_lick}', offset) & unit_key
        contra_miss_unit_psth = psth_foraging.compute_unit_psth_and_raster(unit_key, contra_miss_trials, align_type)

        # --- plot psths (all 4 in one plot) ---
        ax_psth = axs[1 if if_raster else 0, ax_i]
        period_starts_hit = _get_ephys_trial_event_times(align_types,
                                                         align_to=align_type,
                                                         trial_keys=psth_foraging.TrialCondition.get_trials(f'LR_hit{no_early_lick}') & unit_key,
                                                         # cannot use *_hit_trials because it could have been offset
                                                         )
        # _, period_starts_miss = _get_ephys_trial_event_times([trialstart, 'go', 'choice', 'trialend'], 
        #                                                   ipsi_miss_trials.proj() + contra_miss_trials.proj(), align_event=align_event_type)
        
        _plot_psth_foraging(ipsi_hit_unit_psth, contra_hit_unit_psth,
                   vlines=period_starts_hit, ax=ax_psth, xlim=xlim, label='rew', linestyle='-')
        
        _plot_psth_foraging(ipsi_miss_unit_psth, contra_miss_unit_psth,
                   vlines=[], ax=ax_psth, xlim=xlim, label='norew', linestyle = '--')

        ax_psth.set(title=f'{align_type}')
        if ax_i > 0:
            ax_psth.spines['left'].set_visible(False)
            ax_psth.get_yaxis().set_visible(False)

        # --- plot rasters (optional) ---
        if if_raster:
            ax_raster = axs[0, ax_i]
            _plot_spike_raster_foraging(ipsi_hit_unit_psth, contra_hit_unit_psth, ax=ax_raster,
                                           offset=0,
                                           vlines=period_starts_hit,
                                           title='', xlim=xlim)
            _plot_spike_raster_foraging(ipsi_miss_unit_psth, contra_miss_unit_psth, ax=ax_raster,
                                           offset=len(ipsi_hit_unit_psth['trials']) + len(contra_hit_unit_psth['trials']),
                                           vlines=[],
                                           title='', xlim=xlim)
            ax_raster.invert_yaxis()

    # Add unit info
    unit_info = (f'{(lab.WaterRestriction & unit_key).fetch1("water_restriction_number")}, '
                 f'{(experiment.Session & unit_key).fetch1("session_date")}, '
                 f'imec {unit_key["insertion_number"]-1}\n'
                 f'Unit #: {unit_key["unit"]}, '
                 f'{(((ephys.Unit & unit_key) * histology.ElectrodeCCFPosition.ElectrodePosition) * ccf.CCFAnnotation).fetch1("annotation")}'
                 )
    fig.text(0.1, 0.9, unit_info)

    # Scale axis widths to keep the same horizontal aspect ratio (time) across axs
    _set_same_horizonal_aspect_ratio(axs[1 if if_raster else 0, :], xlims)
    if if_raster:
        _set_same_horizonal_aspect_ratio(axs[0, :], xlims)
    ax_psth.legend(fontsize=8)

    return fig


def plot_unit_psth_value_quantile(unit_key, model_id=11, n_quantile=5,
                                  align_types=['trial_start', 'go_cue', 'first_lick_after_go_cue', 'iti_start', 'next_trial_start'],
                                  axs=None, title=''
                                  ):
    """
    (for foraging task) Plot psth grouped by quantiles of action value from behavioral model fitting
    """
    # for (the very few) sessions without zaber feedback signal, use 'bitcodestart' with manual correction (see compute_unit_psth_and_raster)
    if not ephys.TrialEvent & unit_key & 'trial_event_type = "zaberready"':
        align_types = [a + '_bitcode' if 'trial_start' in a else a for a in align_types]

    hemi = _get_units_hemisphere(unit_key)
    contra = "right" if hemi == "left" else "left"

    # Fetch predictive contra choice probabilities
    df = (foraging_model.FittedSessionModel.PredictiveChoiceProb
          & unit_key
          & {'model_id': model_id}
          & {'water_port': contra}
          ).fetch(format='frame').reset_index()[['trial', 'choice_prob']]

    # TODO: turn choice_prob back to real action value

    # Cut choice probabilities into quantiles
    df['quantile_rank'] = pd.qcut(df.choice_prob, n_quantile, labels=False)

    fig = None
    if axs is None:
        fig = plt.figure(figsize=(len(align_types)/5 * 25, 5))
        axs = fig.subplots(1, len(align_types), sharey='row', sharex='col')
        axs = np.atleast_2d(axs).reshape((1, -1))
        plt.subplots_adjust(top=0.8)
    kargs = [{'color': 'b',
              'alpha': np.linspace(0.2, 1, n_quantile)[rank],
              'label': f'contra value quantile {rank + 1}'} for rank in range(n_quantile)]
    xlims = []

    # -- For each align type --
    for ax_i, align_type in enumerate(align_types):
        offset, xlim = (psth_foraging.AlignType & {'align_type_name': align_type}).fetch1('trial_offset', 'xlim')
        xlims.append(xlim)

        # -- For each quantile group --
        psths = []
        for rank in range(n_quantile):
            # Group trials
            trial_num = df[df.quantile_rank == rank]
            trial_num.trial += -1 + offset    # Important note: by definition, the updated value after choice of trial t is Q(t+1), not Q(t)!
                                              #    behavior & ephys: -->  ITI(t - 1) --> | --> choice(t), reward(t) --> ITI(t) --> |
                                              #    model:          Q(t) --> choice prob(t) --> choice (t), reward(t)  | --> Q(t+1) --> choice prob (t+1)
                                              # Therefore, to *align* PSTH to ITI(t) *conditioned* on Q(t), we should:
                                              #    offset *align* relative to *condition* by -1 trial
                                              # That's why an additional -1 is here. (see also psth_foraging.AlignType)

            # Get psths
            this_trials = (experiment.BehaviorTrial & unit_key & trial_num).proj()
            psths.append(psth_foraging.compute_unit_psth_and_raster(unit_key, this_trials, align_type))

        # -- Plot psths for this align type --
        ax_psth = axs[0, ax_i]
        period_starts_all = _get_ephys_trial_event_times(align_types,
                                                         align_to=align_type,
                                                         trial_keys=experiment.BehaviorTrial & unit_key & df,  # From all trials
                                                         )

        _plot_psths(psths, kargs, ax=ax_psth, xlim=xlim, vlines=period_starts_all)
        ax_psth.set(title=f'{align_type}')

    _set_same_horizonal_aspect_ratio(axs[0, :], xlims)
    ax_psth.legend(fontsize=8)

    # Add unit and model info
    unit_info = (f'{(lab.WaterRestriction & unit_key).fetch1("water_restriction_number")}, '
                 f'{(experiment.Session & unit_key).fetch1("session_date")}, '
                 f'imec {unit_key["insertion_number"]-1}\n'
                 f'Unit #: {unit_key["unit"]}, '
                 f'{(((ephys.Unit & unit_key) * histology.ElectrodeCCFPosition.ElectrodePosition) * ccf.CCFAnnotation).fetch1("annotation")}'
                 )
    id, model_notation, desc, accuracy, n = (foraging_model.FittedSessionModel * foraging_model.Model & unit_key & {'model_id': model_id}).fetch1(
        'model_id', 'model_notation', 'desc', 'cross_valid_accuracy_test', 'n_trials')
    fig.text(0.1, 0.9, unit_info)
    fig.text(0.4, 0.9, f'model #{id} {model_notation}\n{desc}\n{n} trials, prediction accuracy (cross-valid) = {accuracy}')

    return fig
