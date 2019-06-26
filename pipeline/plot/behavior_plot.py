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
    window_size is set to 10% of the total trial number if not specified
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


def plot_jaw_movement(session_key, unit_key, tongue_thres=430, trial_limit=10, axs=None):
    """
    Plot jaw movement per trial, time-locked to cue-onset, with spike times overlay
    :param session_key: session where the trials are from
    :param unit_key: unit for spike times overlay
    :param tongue_thres: y-pos of the toungue to be considered "protruding out of the mouth"
    :param trial_limit: number of trial to plot
    """
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
            go_time = (experiment.TrialEvent & tr & 'trial_event_type="go"').fetch1('trial_event_time')

            spike_times = (ephys.TrialSpikes & tr & unit_key).fetch1('spike_times')
            spike_times = spike_times + float(go_time) - float(first_lick_time)  # realigned to first-lick

            tvec = tvec - float(first_lick_time)
            tongue_out_bool = tongue >= tongue_thres

            yield jaw, tongue_out_bool, spike_times, tvec

    if not axs:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    assert len(axs) == 2

    h_spacing = 0.5 * tongue_thres
    for trial_tracks, ax, ax_name, spk_color in zip((l_trial_trk, r_trial_trk),
                                                    axs, ('left lick trials', 'right lick trials'), ('b', 'r')):
        for tr_id, (jaw, tongue_out_bool, spike_times, tvec) in enumerate(get_trial_track(trial_tracks)):
            ax.plot(tvec, jaw + tr_id * h_spacing, 'k', linewidth=2)
            ax.plot(tvec[tongue_out_bool], jaw[tongue_out_bool] + tr_id * h_spacing, '.', color='lime', markersize=2)
            ax.plot(spike_times, np.full_like(spike_times, jaw[tongue_out_bool].mean()
                                              + 4*jaw[tongue_out_bool].std()) + tr_id * h_spacing,
                    '.', color=spk_color, markersize=4)
            ax.set_title(ax_name)
            ax.axvline(x=0, linestyle='--', color='k')

            # cosmetic
            ax.set_xlim((-0.5, 1.5))
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)


def plot_trial_jaw_movement(trial_key):
    """
    Plot trial-specific Jaw Movement time-locked to "go" cue
    """
    trk = (tracking.Tracking.JawTracking * experiment.BehaviorTrial & trial_key & experiment.TrialEvent)
    if len(trk) == 0:
        return 'The selected trial has no Action Event (e.g. cue start)'

    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & trial_key).fetch1('sampling_rate'))
    jaw = trk.fetch1('jaw_y')
    go_time = (experiment.TrialEvent & trk & 'trial_event_type="go"').fetch1('trial_event_time')
    tvec = np.arange(len(jaw)) / tracking_fs - float(go_time)

    b, a = signal.butter(5, (5, 15), btype='band', fs=tracking_fs)
    filt_jaw = signal.filtfilt(b, a, jaw)

    analytic_signal = signal.hilbert(filt_jaw)
    insta_amp = np.abs(analytic_signal)
    insta_phase = np.angle(analytic_signal)

    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
    fig.subplots_adjust(hspace=0.4)

    axs[0, 0].plot(tvec, jaw, '.k')
    axs[0, 0].set_title('Jaw Movement')
    axs[1, 0].plot(tvec, filt_jaw, '.k')
    axs[1, 0].set_title('Bandpass filtered 5-15Hz')
    axs[1, 0].set_xlabel('Time(s)')
    axs[0, 1].plot(tvec, insta_amp, '.k')
    axs[0, 1].set_title('Amplitude')
    axs[1, 1].plot(tvec, insta_phase, '.k')
    axs[1, 1].set_title('Phase')
    axs[1, 1].set_xlabel('Time(s)')

    # cosmetic
    for ax in axs.flatten():
        ax.set_xlim((-3, 3))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return axs


def plot_windowed_jaw_phase_dist(session_key, xlim=(-0.12, 0.3), w_size=0.01, bin_counts=20):
    trks = (tracking.Tracking.JawTracking * experiment.BehaviorTrial & session_key & experiment.TrialEvent)
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & session_key).fetch1('sampling_rate'))

    def get_trial_track():
        for jaw, go_time in zip(*(trks * experiment.TrialEvent & 'trial_event_type="go"').fetch(
                'jaw_y', 'trial_event_time')):
            tvec = np.arange(len(jaw)) / tracking_fs - float(go_time)
            segmented_jaw = jaw[np.logical_and(tvec >= xlim[0], tvec <= xlim[1])]
            if len(segmented_jaw) == (xlim[1] - xlim[0]) * tracking_fs:
                yield segmented_jaw

    jaw_trackings = np.vstack(get_trial_track())  # trials x times

    b, a = signal.butter(5, (5, 15), btype='band', fs=tracking_fs)

    filt_jaw_trackings = signal.filtfilt(b, a, jaw_trackings, axis=1)

    insta_phase = np.angle(signal.hilbert(filt_jaw_trackings, axis=1))
    insta_phase = np.degrees(insta_phase) % 360  # convert to degree [0, 360]

    tvec = np.linspace(xlim[0], xlim[1], jaw_trackings.shape[1])
    windows = np.arange(xlim[0], xlim[1], w_size)

    # plot
    col_counts = 8
    fig, axs = plt.subplots(int(np.ceil(len(windows) / col_counts)), col_counts,
                            figsize=(16, 16),
                            subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.6, hspace=0.3)

    # non-overlapping windowed histogram
    for w_start, ax in zip(windows, axs.flatten()):
        phase = insta_phase[:, np.logical_and(tvec >= w_start, tvec <= w_start + w_size)].flatten()
        plot_polar_histogram(phase, ax, bin_counts=bin_counts)
        ax.set_xlabel(f'{w_start*1000:.0f} to {(w_start + w_size)*1000:.0f}ms', fontweight='bold')


def plot_jaw_phase_dist(session_key, xlim=(-0.12, 0.3), bin_counts=20):
    trks = (tracking.Tracking.JawTracking * experiment.BehaviorTrial & session_key & experiment.TrialEvent)
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & session_key).fetch1('sampling_rate'))

    l_trial_trk = trks & 'trial_instruction="left"' & 'early_lick="no early"'
    r_trial_trk = trks & 'trial_instruction="right"' & 'early_lick="no early"'

    def get_trial_track(trial_tracks):
        for jaw, go_time in zip(*(trial_tracks * experiment.TrialEvent & 'trial_event_type="go"').fetch(
                'jaw_y', 'trial_event_time')):
            tvec = np.arange(len(jaw)) / tracking_fs - float(go_time)
            segmented_jaw = jaw[np.logical_and(tvec >= xlim[0], tvec <= xlim[1])]
            if len(segmented_jaw) == (xlim[1] - xlim[0]) * tracking_fs:
                yield segmented_jaw

    l_jaw_trackings = np.vstack(get_trial_track(l_trial_trk))  # trials x times
    r_jaw_trackings = np.vstack(get_trial_track(r_trial_trk))  # trials x times

    b, a = signal.butter(5, (5, 15), btype='band', fs=tracking_fs)

    filt_l_jaw_trackings = signal.filtfilt(b, a, l_jaw_trackings, axis=1)
    l_insta_phase = np.angle(signal.hilbert(filt_l_jaw_trackings, axis=1))
    l_insta_phase = np.degrees(l_insta_phase) % 360  # convert to degree [0, 360]

    filt_r_jaw_trackings = signal.filtfilt(b, a, r_jaw_trackings, axis=1)
    r_insta_phase = np.angle(signal.hilbert(filt_r_jaw_trackings, axis=1))
    r_insta_phase = np.degrees(r_insta_phase) % 360  # convert to degree [0, 360]

    fig, axs = plt.subplots(1, 2, figsize=(12, 8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.6)

    plot_polar_histogram(l_insta_phase.flatten(), axs[0], bin_counts=bin_counts)
    axs[0].set_title('left lick trials', loc='left', fontweight='bold')
    plot_polar_histogram(r_insta_phase.flatten(), axs[1], bin_counts=bin_counts)
    axs[1].set_title('right lick trials', loc='left', fontweight='bold')


def plot_polar_histogram(data, ax, bin_counts=30):
    bottom = 2

    theta = np.linspace(0.0, 2 * np.pi, bin_counts, endpoint=False)

    radii, tick = np.histogram(data, bins=bin_counts)

    # width of each bin on the plot
    width = (2 * np.pi) / bin_counts

    # make a polar plot
    bars = ax.bar(theta, radii, width=width, bottom=bottom)

    # set the label starting from East
    ax.set_theta_zero_location("E")
    # clockwise
    ax.set_theta_direction(1)


