import numpy as np
import scipy as sp
import datajoint as dj

import matplotlib.pyplot as plt
from scipy import signal

from pipeline import experiment, tracking, ephys


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

    return axis


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

    return axis


def plot_jaw_movement(session_key, tongue_thres=430, trial_limit=10, axis=None):
    trk = (tracking.Tracking.JawTracking * tracking.Tracking.TongueTracking
           * experiment.BehaviorTrial & session_key & experiment.ActionEvent & ephys.TrialSpikes)
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & session_key).fetch1('sampling_rate'))

    l_trial_trk = trk & 'trial_instruction="left"' & 'early_lick="no early"'
    r_trial_trk = trk & 'trial_instruction="right"' & 'early_lick="no early"'

    def get_trial_track(trial_tracks):
        for tr in trial_tracks.fetch(as_dict=True, limit=trial_limit):
            jaw = tr['jaw_y']
            tongue = tr['tongue_y']
            sample_counts = len(jaw)
            tvec = np.arange(sample_counts) / tracking_fs

            first_lick_time = (experiment.ActionEvent & tr & 'action_event_type in ("left lick", "right lick")').fetch(
                    'action_event_time', order_by = 'action_event_time', limit = 1)[0]

            spike_times = np.hstack((ephys.TrialSpikes & tr).fetch('spike_times', limit=5))

            tvec = tvec - float(first_lick_time)
            tongue_out_bool = tongue >= tongue_thres

            yield jaw, tongue_out_bool, spike_times, tvec

    if not axis or len(axis) < 2:
        fig, axis = plt.subplots(1, 2, figsize=(16, 8))

    h_spacing = 0.5 * tongue_thres
    for trial_tracks, ax, ax_name in zip((l_trial_trk, r_trial_trk), axis, ('left lick trials', 'right lick trials')):
        for tr_id, (jaw, tongue_out_bool, spike_times, tvec) in enumerate(get_trial_track(trial_tracks)):
            ax.plot(tvec, jaw + tr_id * h_spacing, 'k', linewidth=2)
            ax.plot(tvec[tongue_out_bool], jaw[tongue_out_bool] + tr_id * h_spacing, '.', color = 'lime', markersize=2)
            ax.plot(spike_times, np.full_like(spike_times, jaw.mean() + 1.5*jaw.std()) + tr_id * h_spacing,
                    '.r', markersize=2)
            ax.set_title(ax_name)
            ax.axvline(x=0, linestyle='--', color='k')

            # cosmetic
            ax.set_xlim((-0.5, 1.5))
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)














