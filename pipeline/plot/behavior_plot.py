import numpy as np
import scipy as sp
import datajoint as dj

import matplotlib.pyplot as plt
from scipy import signal

from pipeline import experiment, tracking


def plot_correct_proportion(session_key, window_size=None, axis=None):
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

    if not axis:
        fig, axis = plt.subplots(1, 1)

    axis.bar(range(len(mv_outcomes)), trial_outcomes * mv_outcomes.max(), alpha=0.3)
    axis.plot(range(len(mv_outcomes)), mv_outcomes, 'k', linewidth=3)
    axis.set_xlabel('Trial')
    axis.set_ylabel('Proportion correct')


def plot_photostim_effect(session_key, photostim_key, axis=None):
    """
    For all trials in this "session_key", split to 4 groups:
    + control left-lick
    + control right-lick
    + photostim left-lick (specified by "photostim_key")
    + photostim right-lick (specified by "photostim_key")
    Plot correct proportion for each group
    Note: ignore "early lick" trials
    """

    ctrl_trials = experiment.BehaviorTrial - experiment.PhotostimTrial & session_key
    stim_trials = experiment.BehaviorTrial * experiment.PhotostimTrial & session_key

    ctrl_left = ctrl_trials & 'trial_instruction="left"' & 'early_lick="no early"'
    ctrl_right = ctrl_trials & 'trial_instruction="right"' & 'early_lick="no early"'

    stim_left = stim_trials & 'trial_instruction="left"' & 'early_lick="no early"'
    stim_right = stim_trials & 'trial_instruction="right"' & 'early_lick="no early"'

    # Restrict by stim location (from photostim_key)
    stim_left = stim_left * experiment.PhotostimEvent & photostim_key
    stim_right = stim_right * experiment.PhotostimEvent & photostim_key

    def get_correct_proportion(trials):
        correct = (trials.fetch('outcome') == 'hit').astype(int)
        return correct.sum()/len(correct)

    # Extract and compute correct proportion
    cp_ctrl_left = get_correct_proportion(ctrl_left)
    cp_ctrl_right = get_correct_proportion(ctrl_right)
    cp_stim_left = get_correct_proportion(stim_left)
    cp_stim_right = get_correct_proportion(stim_right)

    if not axis:
        fig, axis = plt.subplots(1, 1)

    axis.plot([0, 1], [cp_ctrl_left, cp_stim_left], 'b', label='lick left trials')
    axis.plot([0, 1], [cp_ctrl_right, cp_stim_right], 'r', label='lick right trials')

    # plot cosmetic
    ylim = (min([cp_ctrl_left, cp_stim_left, cp_ctrl_right, cp_stim_right]) - 0.1, 1)
    ylim = (0, 1) if ylim[0] < 0 else ylim

    axis.set_xlim((0, 1))
    axis.set_ylim(ylim)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(['Control', 'Photostim'])
    axis.set_ylabel('Proportion correct')

    axis.legend(loc='lower left')
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)


def plot_jaw_movement(session_key, tongue_thres=430, axis=None):
    trk = (tracking.Tracking.JawTracking * tracking.Tracking.TongueTracking
           * experiment.BehaviorTrial & session_key)
    l_trial_trk = trk & 'trial_instruction="left"' & 'early_lick="no early"'
    r_trial_trk = trk & 'trial_instruction="right"' & 'early_lick="no early"'

    for tr in l_trial_trk.fetch('jaw_y', 'tongue_y', limit=10):
        jaw =
        tongue =
        sample_counts = len(jaw_tracking['jaw_x'])
        tvec = np.arange(sample_counts) / tracking_fs












