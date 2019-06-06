import numpy as np
import scipy as sp
import datajoint as dj

import matplotlib.pyplot as plt
from scipy import signal

from pipeline import experiment


def plot_behavior_performance(session_key, window_size=None):
    """
    For a particular session (specified by session_key), extract all behavior trials
    Get outcome of each trials, map to (0, 1) - 1 if 'hit'
    Compute the moving average of these outcomes, based on the specified window_size (number of trial to average)
    window_size is set to 10% of the total trial number if not supplied
    """

    trial_outcomes = (experiment.BehaviorTrial & session_key).fetch('outcome')
    trial_outcomes = (trial_outcomes == 'hit').astype(int)

    window_size = int(.1 * len(trial_outcomes)) if not window_size else int(window_size)
    kernel = np.full((window_size, ), 1/window_size)

    mv_outcomes = signal.convolve(trial_outcomes, kernel, mode='same')

    fig, ax = plt.subplots(1, 1)
    ax.bar(range(len(mv_outcomes)), trial_outcomes * mv_outcomes.max(), alpha=0.3)
    ax.plot(range(len(mv_outcomes)), mv_outcomes, 'k', linewidth=3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Proportion correct')
    plt.show()
    

