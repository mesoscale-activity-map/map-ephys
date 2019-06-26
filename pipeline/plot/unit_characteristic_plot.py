import numpy as np
import datajoint as dj

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd

from pipeline import experiment, ephys, psth

m_scale = 1200

def plot_clustering_quality(probe_insert_key):
    amp, snr, spk_rate, isi_violation = (ephys.Unit * ephys.UnitStat
                                         * ephys.ProbeInsertion.InsertionLocation & probe_insert_key).fetch(
        'unit_amp', 'unit_snr', 'avg_firing_rate', 'isi_violation')

    metrics = {'amp': amp,
               'snr': snr,
               'isi': np.array(isi_violation) * 100,  # to percentage
               'rate': np.array(spk_rate)}
    label_mapper = {'amp': 'Amplitude',
                    'snr': 'Signal to noise ratio (SNR)',
                    'isi': 'ISI violation (%)',
                    'rate': 'Firing rate (spike/s)'}

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.4)

    for (m1, m2), ax in zip(itertools.combinations(list(metrics.keys()), 2), axs.flatten()):
        ax.plot(metrics[m1], metrics[m2], '.k')
        ax.set_xlabel(label_mapper[m1])
        ax.set_ylabel(label_mapper[m2])

        # cosmetic
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


def plot_unit_characteristic(probe_insert_key, axs=None):
    amp, snr, spk_rate, x, y, insertion_depth = (
            ephys.Unit * ephys.ProbeInsertion.InsertionLocation * ephys.UnitStat
            & probe_insert_key & 'unit_quality != "all"').fetch(
        'unit_amp', 'unit_snr', 'avg_firing_rate', 'unit_posx', 'unit_posy', 'dv_location')

    insertion_depth = np.where(np.isnan(insertion_depth), 0, insertion_depth)

    metrics = pd.DataFrame(list(zip(*(amp/amp.max(), snr/snr.max(), spk_rate/spk_rate.max(), x, y + insertion_depth))))
    metrics.columns = ['amp', 'snr', 'rate', 'x', 'y']

    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.6)

    assert axs.size == 3

    cosmetic = {'legend': None,
                'linewidth': 1.75,
                'alpha': 0.9,
                'facecolor': 'none', 'edgecolor': 'k'}

    sns.scatterplot(data=metrics, x='x', y='y', s=metrics.amp*m_scale, ax=axs[0], **cosmetic)
    sns.scatterplot(data=metrics, x='x', y='y', s=metrics.snr*m_scale, ax=axs[1], **cosmetic)
    sns.scatterplot(data=metrics, x='x', y='y', s=metrics.rate*m_scale, ax=axs[2], **cosmetic)

    # cosmetic
    for title, ax in zip(('Amplitude', 'SNR', 'Firing rate'), axs.flatten()):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title)
        ax.set_xlim((-10, 60))


def plot_unit_selectivity(probe_insert_key, axs=None):
    attr_names = ['unit', 'period', 'period_selectivity', 'contra_firing_rate',
                       'ipsi_firing_rate', 'unit_posx', 'unit_posy', 'dv_location']
    selective_units = (psth.PeriodSelectivity * ephys.Unit * ephys.ProbeInsertion.InsertionLocation
                       * experiment.Period & probe_insert_key & 'period_selectivity != "non-selective"').fetch(*attr_names)
    selective_units = pd.DataFrame(selective_units).T
    selective_units.columns = attr_names
    selective_units.period_selectivity.astype('category')

    # --- account for insertion depth (manipulator depth)
    selective_units.unit_posy = (selective_units.unit_posy
                                 + np.where(np.isnan(selective_units.dv_location.values.astype(float)),
                                            0, selective_units.dv_location.values.astype(float)))

    # --- get ipsi vs. contra firing rate difference
    f_rate_diff = np.abs(selective_units.ipsi_firing_rate - selective_units.contra_firing_rate)
    selective_units['f_rate_diff'] = f_rate_diff / f_rate_diff.max()

    # --- prepare for plotting
    cosmetic = {'legend': None,
                'linewidth': 0.0001}
    ymax = selective_units.unit_posy.max() + 100

    # a bit of hack to get 'open circle'
    pts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
    vert = np.r_[circ, circ[::-1] * .7]

    open_circle = mpl.path.Path(vert)

    # --- plot
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.6)

    assert axs.size == 3

    for (title, df), ax in zip(((p, selective_units[selective_units.period == p])
                                for p in ('sample', 'delay', 'response')), axs):
        sns.scatterplot(data=df, x='unit_posx', y='unit_posy',
                        s=df.f_rate_diff.values.astype(float)*m_scale,
                        hue='period_selectivity', marker=open_circle,
                        palette={'contra-selective': 'b', 'ipsi-selective': 'r'},
                        ax=ax, **cosmetic)
        contra_p = (df.period_selectivity == 'contra-selective').sum() / len(df) * 100
        # cosmetic
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(f'{title}\n% contra: {contra_p:.2f}\n% ipsi: {100-contra_p:.2f}')
        ax.set_xlim((-10, 60))
        ax.set_ylim((0, ymax))


def plot_unit_bilateral_photostim_effect(probe_insert_key, axs=None):
    cue_onset = (experiment.Period & 'period = "delay"').fetch1('period_start')

    no_stim_cond = (psth.TrialCondition
                    & {'trial_condition_name':
                       'all_noearlylick_both_alm_nostim'}).fetch1('KEY')

    bi_stim_cond = (psth.TrialCondition
                    & {'trial_condition_name':
                       'all_noearlylick_both_alm_stim'}).fetch1('KEY')

    # get photostim duration
    stim_durs = np.unique((experiment.Photostim & experiment.PhotostimEvent
                           * psth.TrialCondition().get_trials('all_noearlylick_both_alm_stim')
                           & probe_insert_key).fetch('duration'))
    stim_dur = _extract_one_stim_dur(stim_durs)

    units = ephys.Unit & probe_insert_key & 'unit_quality != "all"'

    metrics = pd.DataFrame(columns=['unit', 'x', 'y', 'frate_change'])

    # XXX: could be done with 1x fetch+join
    for u_idx, unit in enumerate(units.fetch('KEY')):

        x, y = (ephys.Unit & unit).fetch1('unit_posx', 'unit_posy')

        nostim_psth, nostim_edge = (
            psth.UnitPsth & {**unit, **no_stim_cond}).fetch1('unit_psth')

        bistim_psth, bistim_edge = (
            psth.UnitPsth & {**unit, **bi_stim_cond}).fetch1('unit_psth')

        # compute the firing rate difference between contra vs. ipsi within the stimulation duration
        ctrl_frate = nostim_psth[np.logical_and(nostim_edge[1:] >= cue_onset, nostim_edge[1:] <= cue_onset + stim_dur)]
        stim_frate = bistim_psth[np.logical_and(bistim_edge[1:] >= cue_onset, bistim_edge[1:] <= cue_onset + stim_dur)]

        frate_change = np.abs(stim_frate.mean() - ctrl_frate.mean()) / ctrl_frate.mean()

        metrics.loc[u_idx] = (int(unit['unit']), x, y, frate_change)

    metrics.frate_change = metrics.frate_change / metrics.frate_change.max()

    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(4, 8))

    cosmetic = {'legend': None,
                'linewidth': 1.75,
                'alpha': 0.9,
                'facecolor': 'none', 'edgecolor': 'k'}

    sns.scatterplot(data=metrics, x='x', y='y', s=metrics.frate_change*m_scale,
                    ax=axs, **cosmetic)

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_title('% change')
    axs.set_xlim((-10, 60))


def plot_stacked_contra_ipsi_psth(probe_insert_key, axs=None):

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    assert axs.size == 2

    period_starts = (experiment.Period
                     & 'period in ("sample", "delay", "response")').fetch(
                         'period_start')

    hemi = (ephys.ProbeInsertion.InsertionLocation
            * experiment.BrainLocation & probe_insert_key).fetch1('hemisphere')

    conds_i = (psth.TrialCondition
               & {'trial_condition_name':
                  'good_noearlylick_left_hit' if hemi == 'left' else 'good_noearlylick_right_hit'}).fetch1('KEY')

    conds_c = (psth.TrialCondition
               & {'trial_condition_name':
                  'good_noearlylick_right_hit' if hemi == 'left' else 'good_noearlylick_left_hit'}).fetch1('KEY')

    sel_i = (ephys.Unit * psth.UnitSelectivity
             & 'unit_selectivity = "ipsi-selective"' & probe_insert_key)

    sel_c = (ephys.Unit * psth.UnitSelectivity
             & 'unit_selectivity = "contra-selective"' & probe_insert_key)

    # ipsi selective ipsi trials
    psth_is_it = (psth.UnitPsth * sel_i.proj('unit_posy') & conds_i).fetch(order_by='unit_posy desc')

    # ipsi selective contra trials
    psth_is_ct = (psth.UnitPsth * sel_i.proj('unit_posy') & conds_c).fetch(order_by='unit_posy desc')

    # contra selective contra trials
    psth_cs_ct = (psth.UnitPsth * sel_c.proj('unit_posy') & conds_c).fetch(order_by='unit_posy desc')

    # contra selective ipsi trials
    psth_cs_it = (psth.UnitPsth * sel_c.proj('unit_posy') & conds_i).fetch(order_by='unit_posy desc')

    _plot_stacked_psth_diff(psth_cs_ct, psth_cs_it, ax=axs[0],
                            vlines=period_starts, flip=True)

    axs[0].set_title('Contra-selective Units')
    axs[0].set_ylabel('Unit (by depth)')
    axs[0].set_xlabel('Time to go (s)')

    _plot_stacked_psth_diff(psth_is_it, psth_is_ct, ax=axs[1],
                            vlines=period_starts)

    axs[1].set_title('Ipsi-selective Units')
    axs[1].set_ylabel('Unit (by depth)')
    axs[1].set_xlabel('Time to go (s)')


def plot_avg_contra_ipsi_psth(probe_insert_key, axs=None):

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    assert axs.size == 2

    period_starts = (experiment.Period
                     & 'period in ("sample", "delay", "response")').fetch(
                         'period_start')

    good_unit = ephys.Unit & 'unit_quality != "all"'

    hemi = (ephys.ProbeInsertion.InsertionLocation
            * experiment.BrainLocation & probe_insert_key).fetch1('hemisphere')

    conds_i = (psth.TrialCondition
               & {'trial_condition_name':
                  'good_noearlylick_left_hit' if hemi == 'left' else 'good_noearlylick_right_hit'}).fetch('KEY')

    conds_c = (psth.TrialCondition
               & {'trial_condition_name':
                  'good_noearlylick_right_hit' if hemi == 'left' else 'good_noearlylick_left_hit'}).fetch('KEY')

    sel_i = (ephys.Unit * psth.UnitSelectivity
             & 'unit_selectivity = "ipsi-selective"' & probe_insert_key)

    sel_c = (ephys.Unit * psth.UnitSelectivity
             & 'unit_selectivity = "contra-selective"' & probe_insert_key)

    psth_is_it = (((psth.UnitPsth & conds_i)
                   * ephys.Unit.proj('unit_posy'))
                  & good_unit.proj() & sel_i.proj()).fetch(
                      'unit_psth', order_by='unit_posy desc')

    psth_is_ct = (((psth.UnitPsth & conds_c)
                   * ephys.Unit.proj('unit_posy'))
                  & good_unit.proj() & sel_i.proj()).fetch(
                      'unit_psth', order_by='unit_posy desc')

    psth_cs_ct = (((psth.UnitPsth & conds_c)
                   * ephys.Unit.proj('unit_posy'))
                  & good_unit.proj() & sel_c.proj()).fetch(
                      'unit_psth', order_by='unit_posy desc')

    psth_cs_it = (((psth.UnitPsth & conds_i)
                   * ephys.Unit.proj('unit_posy'))
                  & good_unit.proj() & sel_c.proj()).fetch(
                      'unit_psth', order_by='unit_posy desc')

    _plot_avg_psth(psth_cs_it, psth_cs_ct, period_starts, axs[0],
                   'Contra-selective')
    _plot_avg_psth(psth_is_it, psth_is_ct, period_starts, axs[1],
                   'Ipsi-selective')

    ymax = max([ax.get_ylim()[1] for ax in axs])
    for ax in axs:
        ax.set_ylim((0, ymax))


def plot_psth_bilateral_photostim_effect(probe_insert_key, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    assert axs.size == 2

    insert = (ephys.ProbeInsertion.InsertionLocation
              * experiment.BrainLocation & probe_insert_key).fetch1()

    period_starts = (experiment.Period
                     & 'period in ("sample", "delay", "response")').fetch(
                         'period_start')

    psth_s_l = (psth.UnitPsth * psth.TrialCondition & probe_insert_key
                & {'trial_condition_name':
                   'all_noearlylick_both_alm_stim_left'}).fetch('unit_psth')

    psth_n_l = (psth.UnitPsth * psth.TrialCondition & probe_insert_key
                & {'trial_condition_name':
                   'all_noearlylick_both_alm_nostim_left'}).fetch('unit_psth')

    psth_s_r = (psth.UnitPsth * psth.TrialCondition & probe_insert_key
                & {'trial_condition_name':
                   'all_noearlylick_both_alm_stim_right'}).fetch('unit_psth')

    psth_n_r = (psth.UnitPsth * psth.TrialCondition & probe_insert_key
                & {'trial_condition_name':
                   'all_noearlylick_both_alm_nostim_right'}).fetch('unit_psth')

    # get photostim duration
    stim_durs = np.unique((experiment.Photostim & experiment.PhotostimEvent
                           * psth.TrialCondition().get_trials('all_noearlylick_both_alm_stim')
                           & probe_insert_key).fetch('duration'))
    stim_dur = _extract_one_stim_dur(stim_durs)

    if insert['hemisphere'] == 'left':
        psth_s_i = psth_s_l
        psth_n_i = psth_n_l
        psth_s_c = psth_s_r
        psth_n_c = psth_n_r
    else:
        psth_s_i = psth_s_r
        psth_n_i = psth_n_r
        psth_s_c = psth_s_l
        psth_n_c = psth_n_l

    _plot_avg_psth(psth_n_i, psth_n_c, period_starts, axs[0],
                   'Control')
    _plot_avg_psth(psth_s_i, psth_s_c, period_starts, axs[1],
                   'Bilateral ALM photostim')

    # cosmetic
    ymax = max([ax.get_ylim()[1] for ax in axs])
    for ax in axs:
        ax.set_ylim((0, ymax))

    # add shaded bar for photostim
    delay = (experiment.Period  # TODO: use from period_starts
             & 'period = "delay"').fetch1('period_start')
    axs[1].axvspan(delay, delay + stim_dur, alpha=0.3, color='royalblue')


def plot_coding_direction(units, time_period=None, axs=None):
    _, proj_contra_trial, proj_ipsi_trial, time_stamps = psth.compute_CD_projected_psth(
        units.fetch('KEY'), time_period=time_period)

    period_starts = (experiment.Period & 'period in ("sample", "delay", "response")').fetch('period_start')

    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    # plot
    _plot_with_sem(proj_contra_trial, time_stamps, ax=axs, c='b')
    _plot_with_sem(proj_ipsi_trial, time_stamps, ax=axs, c='r')

    for x in period_starts:
        axs.axvline(x=x, linestyle = '--', color = 'k')
    # cosmetic
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylabel('CD projection (a.u.)')
    axs.set_xlabel('Time (s)')


def plot_paired_coding_direction(unit_g1, unit_g2, labels=None, time_period=None):
    """
    Plot trial-to-trial CD-endpoint correlation between CD-projected trial-psth from two unit-groups (e.g. two brain regions)
    Note: coding direction is calculated on selective units, contra vs. ipsi, within the specified time_period
    """
    _, proj_contra_trial_g1, proj_ipsi_trial_g1, time_stamps = psth.compute_CD_projected_psth(
        unit_g1.fetch('KEY'), time_period=time_period)
    _, proj_contra_trial_g2, proj_ipsi_trial_g2, time_stamps = psth.compute_CD_projected_psth(
        unit_g2.fetch('KEY'), time_period=time_period)

    period_starts = (experiment.Period & 'period in ("sample", "delay", "response")').fetch('period_start')

    if labels:
        assert len(labels) == 2
    else:
        labels = ('unit group 1', 'unit group 2')

    # plot projected trial-psth
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    _plot_with_sem(proj_contra_trial_g1, time_stamps, ax=axs[0], c='b')
    _plot_with_sem(proj_ipsi_trial_g1, time_stamps, ax=axs[0], c='r')
    _plot_with_sem(proj_contra_trial_g2, time_stamps, ax=axs[1], c='b')
    _plot_with_sem(proj_ipsi_trial_g2, time_stamps, ax=axs[1], c='r')

    # cosmetic
    for ax, label in zip(axs, labels):
        for x in period_starts:
            ax.axvline(x=x, linestyle = '--', color = 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('CD projection (a.u.)')
        ax.set_xlabel('Time (s)')
        ax.set_title(label)

    # plot trial CD-endpoint correlation
    p_start, p_end = time_period
    contra_cdend_1 = proj_contra_trial_g1[:, np.logical_and(time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)
    contra_cdend_2 = proj_contra_trial_g2[:, np.logical_and(time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)
    ipsi_cdend_1 = proj_ipsi_trial_g1[:, np.logical_and(time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)
    ipsi_cdend_2 = proj_ipsi_trial_g2[:, np.logical_and(time_stamps >= p_start, time_stamps < p_end)].mean(axis=1)

    c_df = pd.DataFrame([contra_cdend_1, contra_cdend_2]).T
    c_df.columns = labels
    c_df['trial-type'] = 'contra'
    i_df = pd.DataFrame([ipsi_cdend_1, ipsi_cdend_2]).T
    i_df.columns = labels
    i_df['trial-type'] = 'ipsi'
    df = c_df.append(i_df)

    jplot = jointplot_w_hue(data=df, x=labels[0], y=labels[1], hue='trial-type', colormap=['b', 'r'],
                            figsize=(8, 6), fig=None, scatter_kws=None)
    jplot['fig'].show()


# ---------- PLOTTING HELPER FUNCTIONS --------------


def _plot_avg_psth(ipsi_psth, contra_psth, vlines={}, ax=None, title=''):

    avg_contra_psth = np.vstack(
        np.array([i[0] for i in contra_psth])).mean(axis=0)
    contra_edges = contra_psth[0][1][:-1]

    avg_ipsi_psth = np.vstack(
        np.array([i[0] for i in ipsi_psth])).mean(axis=0)
    ipsi_edges = ipsi_psth[0][1][:-1]

    ax.plot(contra_edges, avg_contra_psth, 'b', label='contra')
    ax.plot(ipsi_edges, avg_ipsi_psth, 'r', label='ipsi')

    for x in vlines:
        ax.axvline(x=x, linestyle='--', color='k')

    # cosmetic
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel('Firing Rate (spike/s)')
    ax.set_xlabel('Time (s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def _plot_stacked_psth_diff(psth_a, psth_b, vlines=[], ax=None, flip=False):
    """
    Heatmap of (psth_a - psth_b)
    psth_a, psth_b are the unit_psth(s) resulted from psth.UnitPSTH.fetch()
    """
    plt_xmin, plt_xmax = -3, 3

    assert len(psth_a) == len(psth_b)
    nunits = len(psth_a)
    aspect = 4.5 / nunits  # 4:3 aspect ratio
    extent = [plt_xmin, plt_xmax, 0, nunits]

    a_data = np.array([r[0] for r in psth_a['unit_psth']])
    b_data = np.array([r[0] for r in psth_b['unit_psth']])

    result = a_data - b_data
    result = result / np.repeat(result.max(axis=1)[:, None], result.shape[1], axis=1)

    # color flip
    result = result * -1 if flip else result

    # moving average
    result = np.array([_movmean(i) for i in result])

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # ax.set_axis_off()
    ax.set_xlim([plt_xmin, plt_xmax])
    for x in vlines:
        ax.axvline(x=x, linestyle='--', color='k')

    im = ax.imshow(result, cmap=plt.cm.bwr, aspect=aspect, extent=extent)
    im.set_clim((-1, 1))


def _plot_with_sem(data, t_vec, ax, c='k'):
    v_mean = np.nanmean(data, axis=0)
    v_sem = np.nanstd(data, axis=0) #/ np.sqrt(data.shape[0])
    ax.plot(t_vec, v_mean, c)
    ax.fill_between(t_vec, v_mean - v_sem, v_mean + v_sem, alpha=0.25, facecolor=c)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def _movmean(data, nsamp=5):
    ret = np.cumsum(data, dtype=float)
    ret[nsamp:] = ret[nsamp:] - ret[:-nsamp]
    return ret[nsamp - 1:] / nsamp


def _extract_one_stim_dur(stim_durs):
    """
    In case of multiple photostim durations - pick the shortest duration
    In case of no photostim durations - return the default of 0.5s
    """
    default_stim_dur = 0.5
    if len(stim_durs) == 0:
        return default_stim_dur
    elif len(stim_durs) > 1:
        print(f'Found multiple stim durations: {stim_durs} - select {min(stim_durs)}')
        return float(min(stim_durs))
    else:
        return float(stim_durs[0]) if len(stim_durs) == 1 and stim_durs[0] else default_stim_dur


def jointplot_w_hue(data, x, y, hue=None, colormap=None,
                    figsize=None, fig=None, scatter_kws=None):
    """
    __author__ = "lewis.r.liu@gmail.com"
    __copyright__ = "Copyright 2018, github.com/ruxi"
    __license__ = "MIT"
    __version__ = 0.0
    .1

    # update: Mar 5 , 2018
    # created: Feb 19, 2018
    # desc: seaborn jointplot with 'hue'
    # prepared for issue: https://github.com/mwaskom/seaborn/issues/365

    jointplots with hue groupings.
    minimum working example
    -----------------------
    iris = sns.load_dataset("iris")
    jointplot_w_hue(data=iris, x = 'sepal_length', y = 'sepal_width', hue = 'species')['fig']
    changelog
    ---------
    2018 Mar 5: added legends and colormap
    2018 Feb 19: gist made
    """

    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    # defaults
    if colormap is None:
        colormap = sns.color_palette()  # ['blue','orange']
    if figsize is None:
        figsize = (5, 5)
    if fig is None:
        fig = plt.figure(figsize = figsize)
    if scatter_kws is None:
        scatter_kws = dict(alpha = 0.4, lw = 1)

    # derived variables
    if hue is None:
        return "use normal sns.jointplot"
    hue_groups = data[hue].unique()

    subdata = dict()
    colors = dict()

    active_colormap = colormap[0: len(hue_groups)]
    legend_mapping = []
    for hue_grp, color in zip(hue_groups, active_colormap):
        legend_entry = mpatches.Patch(color = color, label = hue_grp)
        legend_mapping.append(legend_entry)

        subdata[hue_grp] = data[data[hue] == hue_grp]
        colors[hue_grp] = color

    # canvas setup
    grid = gridspec.GridSpec(2, 2,
                             width_ratios = [4, 1],
                             height_ratios = [1, 4],
                             hspace = 0, wspace = 0
                             )
    ax_main = plt.subplot(grid[1, 0])
    ax_xhist = plt.subplot(grid[0, 0], sharex = ax_main)
    ax_yhist = plt.subplot(grid[1, 1])  # , sharey=ax_main)

    ## plotting

    # histplot x-axis
    for hue_grp in hue_groups:
        sns.distplot(subdata[hue_grp][x], color = colors[hue_grp]
                     , ax = ax_xhist)

    # histplot y-axis
    for hue_grp in hue_groups:
        sns.distplot(subdata[hue_grp][y], color = colors[hue_grp]
                     , ax = ax_yhist, vertical = True)

        # main scatterplot
    # note: must be after the histplots else ax_yhist messes up
    for hue_grp in hue_groups:
        sns.regplot(data = subdata[hue_grp], fit_reg = True,
                    x = x, y = y, ax = ax_main, color = colors[hue_grp]
                    , line_kws={'alpha': 0.5}, scatter_kws = scatter_kws
                    )

        # despine
    for myax in [ax_yhist, ax_xhist]:
        sns.despine(ax = myax, bottom = False, top = True, left = False, right = True
                    , trim = False)
        plt.setp(myax.get_xticklabels(), visible = False)
        plt.setp(myax.get_yticklabels(), visible = False)

    # topright
    ax_legend = plt.subplot(grid[0, 1])  # , sharey=ax_main)
    plt.setp(ax_legend.get_xticklabels(), visible = False)
    plt.setp(ax_legend.get_yticklabels(), visible = False)

    ax_legend.legend(handles = legend_mapping)
    return dict(fig = fig, gridspec = grid)

