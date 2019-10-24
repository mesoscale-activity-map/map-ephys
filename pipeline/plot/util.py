# -*- mode: python -*-

import inspect

from IPython.core.display import HTML, display
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pipeline import experiment, ephys, psth


def show_source(function):
    code = inspect.getsource(function)
    style = HtmlFormatter().get_style_defs('.highlight')
    html = highlight(code, PythonLexer(), HtmlFormatter(style='colorful'))
    display(HTML('<style>{}</style>'.format(style)))
    display(HTML(html))


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


def _get_trial_event_times(events, units, trial_cond_name):
    """
    Get median event start times from all unit-trials from the specified "trial_cond_name" and "units" - aligned to GO CUE
    :param events: list of events
    """
    events = list(events) + ['go']

    event_types, event_times = (psth.TrialCondition().get_trials(trial_cond_name)
                                * (experiment.TrialEvent & [{'trial_event_type': eve} for eve in events])
                                & units).fetch('trial_event_type', 'trial_event_time')
    period_starts = [(event_type, np.nanmedian((event_times[event_types == event_type]
                                                - event_times[event_types == 'go']).astype(float)))
                     for event_type in events[:-1] if len(event_times[event_types == event_type])]
    present_events, event_starts = list(zip(*period_starts))
    return present_events, event_starts


def _get_units_hemisphere(units):
    hemispheres = np.unique((ephys.ProbeInsertion.InsertionLocation
                             * experiment.BrainLocation & units).fetch('hemisphere'))
    if len(hemispheres) > 1:
        raise Exception('Error! The specified units belongs to both hemispheres...')
    return hemispheres[0]


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