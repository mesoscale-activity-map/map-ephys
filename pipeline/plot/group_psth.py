
import numpy as np

import matplotlib.pyplot as plt

from pipeline import psth


def movmean(data, nsamp=5):  # TODO: moveout
    ''' moving average over n samples '''
    ret = np.cumsum(data, dtype=float)
    ret[nsamp:] = ret[nsamp:] - ret[:-nsamp]
    return ret[nsamp - 1:] / nsamp


def group_psth_ll(psth_a, psth_b, invert=False):
    plt_xmin, plt_xmax = -3, 3

    assert len(psth_a) == len(psth_b)
    nunits = len(psth_a)
    aspect = 2 / nunits
    extent = [plt_xmin, plt_xmax, 0, nunits]

    a_data = np.array([r[0] for r in psth_a['unit_psth']])
    b_data = np.array([r[0] for r in psth_b['unit_psth']])

    # scale per-unit PSTHS's
    a_data = np.array([movmean(i * (1 / np.max(i))) for i in a_data])
    b_data = np.array([movmean(i * (1 / np.max(i))) for i in b_data])

    if invert:
        result = (a_data - b_data) * -1
    else:
        result = a_data - b_data

    ax = plt.subplot(111)

    # ax.set_axis_off()
    ax.set_xlim([plt_xmin, plt_xmax])
    ax.axvline(0, 0, 1, ls='--', color='k')
    ax.axvline(-1.2, 0, 1, ls='--', color='k')
    ax.axvline(-2.4, 0, 1, ls='--', color='k')

    plt.imshow(result, cmap=plt.cm.bwr, aspect=aspect, extent=extent)


def group_psth(group_condition_key):

    # XXX: currently raises NotImplementedError;
    # see group_psth_rework.ipynb for latest status
    unit_psths = psth.UnitGroupPsth.get(group_condition_key)

    group_psth_ll(unit_psths[:]['unit_psth'])
