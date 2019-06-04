
import numpy as np

import matplotlib.pyplot as plt

from pipeline import psth


def group_psth_ll(units):

    plt_xmin, plt_xmax = -3, 3
    nunits = len(units)
    aspect = 2 / nunits
    extent = [plt_xmin, plt_xmax, 0, nunits]

    units_0 = np.array([u[0] for u in units])
    units_scaled = np.array([i * (1 / np.max(i)) for i in units_0])

    ax = plt.subplot(111)

    # ax.set_axis_off()
    ax.set_xlim([plt_xmin, plt_xmax])
    ax.axvline(0, 0, 1, ls='--', color='k')
    ax.axvline(-1.2, 0, 1, ls='--', color='k')
    ax.axvline(-2.4, 0, 1, ls='--', color='k')

    plt.imshow(units_scaled, cmap=plt.cm.bwr, aspect=aspect, extent=extent)


def group_psth(group_condition_key):

    unit_psths = psth.UnitGroupPsth.get(group_condition_key)

    group_psth_ll(unit_psths[:]['unit_psth'])
