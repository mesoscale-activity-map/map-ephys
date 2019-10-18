import numpy as np
import scipy as sp
import datajoint as dj

import matplotlib.pyplot as plt
from scipy import signal

from pipeline import experiment, tracking, ephys


# ======== Define some useful variables ==============

_side_cam = {'tracking_device': 'Camera 0'}

_tracked_nose_features = [n for n in tracking.Tracking.NoseTracking.heading.names if n not in tracking.Tracking.heading.names]
_tracked_tongue_features = [n for n in tracking.Tracking.TongueTracking.heading.names if n not in tracking.Tracking.heading.names]
_tracked_jaw_features = [n for n in tracking.Tracking.JawTracking.heading.names if n not in tracking.Tracking.heading.names]


def plot_correct_proportion(session_key, window_size=None, axs=None, plot=True):
    """
    For a particular session (specified by session_key), extract all behavior trials
    Get outcome of each trials, map to (0, 1) - 1 if 'hit'
    Compute the moving average of these outcomes, based on the specified window_size (number of trial to average)
    window_size is set to 10% of the total trial number if not specified
    Return the figure handle and the performance data array
    """

    trial_outcomes = (experiment.BehaviorTrial & session_key).fetch('outcome')
    trial_outcomes = (trial_outcomes == 'hit').astype(int)

    window_size = int(.1 * len(trial_outcomes)) if not window_size else int(window_size)
    kernel = np.full((window_size, ), 1/window_size)

    mv_outcomes = signal.convolve(trial_outcomes, kernel, mode='same')

    fig = None
    if plot:
        if not axs:
            fig, axs = plt.subplots(1, 1)

        axs.bar(range(len(mv_outcomes)), trial_outcomes * mv_outcomes.max(), alpha=0.3)
        axs.plot(range(len(mv_outcomes)), mv_outcomes, 'k', linewidth=3)
        axs.set_xlabel('Trial')
        axs.set_ylabel('Proportion correct')

    return fig, mv_outcomes


def plot_photostim_effect(session_key, photostim_key, axs=None, title='', plot=True):
    """
    For all trials in this "session_key", split to 4 groups:
    + control left-lick
    + control right-lick
    + photostim left-lick (specified by "photostim_key")
    + photostim right-lick (specified by "photostim_key")
    Plot correct proportion for each group
    Note: ignore "early lick" trials
    Return: fig, (cp_ctrl_left, cp_ctrl_right), (cp_stim_left, cp_stim_right)
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

    fig = None

    if plot:
        if not axs:
            fig, axs = plt.subplots(1, 1)

        axs.plot([0, 1], [cp_ctrl_left, cp_stim_left], 'b', label='lick left trials')
        axs.plot([0, 1], [cp_ctrl_right, cp_stim_right], 'r', label='lick right trials')

        # plot cosmetic
        ylim = (min([cp_ctrl_left, cp_stim_left, cp_ctrl_right, cp_stim_right]) - 0.1, 1)
        ylim = (0, 1) if ylim[0] < 0 else ylim

        axs.set_xlim((0, 1))
        axs.set_ylim(ylim)
        axs.set_xticks([0, 1])
        axs.set_xticklabels(['Control', 'Photostim'])
        axs.set_ylabel('Proportion correct')

        axs.legend(loc='lower left')
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.set_title(title)

    return fig, (cp_ctrl_left, cp_ctrl_right), (cp_stim_left, cp_stim_right)


def plot_tracking(session_key, unit_key,
                  tracking_feature='jaw_y', camera_key=_side_cam,
                  trial_offset=0, trial_limit=10, xlim=(-0.5, 1.5), axs=None):
    """
    Plot jaw movement per trial, time-locked to cue-onset, with spike times overlay
    :param session_key: session where the trials are from
    :param unit_key: unit for spike times overlay
    :param tracking_feature: which tracking feature to plot (default to `jaw_y`)
    :param camera_key: tracking from which camera to plot (default to Camera 0, i.e. the side camera)
    :param trial_offset: number of trial to plot from
    :param trial_limit: number of trial to plot to
    """

    if tracking_feature not in _tracked_nose_features + _tracked_tongue_features + _tracked_jaw_features:
        print(f'Unknown tracking type: {tracking_feature}\nAvailable tracking types are: {_tracked_nose_features + _tracked_tongue_features + _tracked_jaw_features}')
        return

    trk = (tracking.Tracking.JawTracking * tracking.Tracking.TongueTracking * tracking.Tracking.NoseTracking
           * experiment.BehaviorTrial & camera_key & session_key & experiment.ActionEvent & ephys.Unit.TrialSpikes)
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & session_key).fetch1('sampling_rate'))

    l_trial_trk = trk & 'trial_instruction="left"' & 'early_lick="no early"' & 'outcome="hit"'
    r_trial_trk = trk & 'trial_instruction="right"' & 'early_lick="no early"' & 'outcome="hit"'

    def get_trial_track(trial_tracks):
        for tr in trial_tracks.fetch(as_dict=True, offset=trial_offset, limit=trial_limit):
            trk_feat = tr[tracking_feature]
            tongue_out_bool = tr['tongue_likelihood'] > 0.9

            sample_counts = len(trk_feat)
            tvec = np.arange(sample_counts) / tracking_fs

            first_lick_time = (experiment.ActionEvent & tr
                               & {'action_event_type': f'{tr["trial_instruction"]} lick'}).fetch(
                'action_event_time', order_by='action_event_time', limit=1)[0]
            go_time = (experiment.TrialEvent & tr & 'trial_event_type="go"').fetch1('trial_event_time')

            spike_times = (ephys.Unit.TrialSpikes & tr & unit_key).fetch1('spike_times')
            spike_times = spike_times + float(go_time) - float(first_lick_time)  # realigned to first-lick

            tvec = tvec - float(first_lick_time)

            yield trk_feat, tongue_out_bool, spike_times, tvec

    fig = None
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    assert len(axs) == 2

    h_spacing = 150
    for trial_tracks, ax, ax_name, spk_color in zip((l_trial_trk, r_trial_trk), axs,
                                                    ('left lick trials', 'right lick trials'), ('b', 'r')):
        for tr_id, (trk_feat, tongue_out_bool, spike_times, tvec) in enumerate(get_trial_track(trial_tracks)):
            ax.plot(tvec, trk_feat + tr_id * h_spacing, '.k', markersize=1)
            ax.plot(tvec[tongue_out_bool], trk_feat[tongue_out_bool] + tr_id * h_spacing, '.', color='lime', markersize=2)
            ax.plot(spike_times, np.full_like(spike_times, trk_feat[tongue_out_bool].mean() + h_spacing/10) + tr_id * h_spacing,
                    '|', color=spk_color, markersize=4)
            ax.set_title(ax_name)
            ax.axvline(x=0, linestyle='--', color='k')

            # cosmetic
            ax.set_xlim(xlim)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    return fig


def plot_unit_jaw_phase_dist(session_key, unit_key, bin_counts=20):
    trk = (tracking.Tracking.JawTracking * tracking.Tracking.TongueTracking
           * experiment.BehaviorTrial & _side_cam & session_key & experiment.ActionEvent & ephys.Unit.TrialSpikes)
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & session_key).fetch1('sampling_rate'))

    l_trial_trk = trk & 'trial_instruction="left"' & 'early_lick="no early"' & 'outcome="hit"'
    r_trial_trk = trk & 'trial_instruction="right"' & 'early_lick="no early"' & 'outcome="hit"'

    def get_trial_track(trial_tracks):
        jaws, spike_times, go_times = (ephys.Unit.TrialSpikes * trial_tracks * experiment.TrialEvent
                                       & unit_key & 'trial_event_type="go"').fetch(
            'jaw_y', 'spike_times', 'trial_event_time')
        spike_times = spike_times + go_times.astype(float)

        flattened_jaws = np.hstack(jaws)
        jsize = np.cumsum([0] + [j.size for j in jaws])
        _, phase = compute_insta_phase_amp(flattened_jaws, tracking_fs, freq_band = (5, 15))
        stacked_insta_phase = [phase[start: end] for start, end in zip(jsize[:-1], jsize[1:])]

        for spks, jphase in zip(spike_times, stacked_insta_phase):
            j_tvec = np.arange(len(jphase)) / tracking_fs

            # find the tracking timestamps corresponding to the spiketimes; and get the corresponding phase
            nearest_indices = np.searchsorted(j_tvec, spks, side="left")
            nearest_indices = np.where(nearest_indices == len(j_tvec), len(j_tvec) - 1, nearest_indices)

            yield jphase[nearest_indices]

    l_insta_phase = np.hstack(list(get_trial_track(l_trial_trk)))
    r_insta_phase = np.hstack(list(get_trial_track(r_trial_trk)))

    fig, axs = plt.subplots(1, 2, figsize=(12, 8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.6)

    plot_polar_histogram(l_insta_phase, axs[0], bin_counts=bin_counts)
    axs[0].set_title('left lick trials', loc='left', fontweight='bold')
    plot_polar_histogram(r_insta_phase, axs[1], bin_counts=bin_counts)
    axs[1].set_title('right lick trials', loc='left', fontweight='bold')

    return fig


def plot_trial_tracking(trial_key, tracking_feature='jaw_y', camera_key=_side_cam,):
    """
    Plot trial-specific Jaw Movement time-locked to "go" cue
    """
    if tracking_feature not in _tracked_nose_features + _tracked_tongue_features + _tracked_jaw_features:
        print(f'Unknown tracking type: {tracking_feature}\nAvailable tracking types are: {_tracked_nose_features + _tracked_tongue_features + _tracked_jaw_features}')
        return

    trk = (tracking.Tracking.JawTracking * tracking.Tracking.NoseTracking * tracking.Tracking.TongueTracking
           * experiment.BehaviorTrial & camera_key & trial_key & experiment.TrialEvent)
    if len(trk) == 0:
        return 'The selected trial has no Action Event (e.g. cue start)'

    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & trial_key).fetch1('sampling_rate'))
    trk_feat = trk.fetch1(tracking_feature)
    go_time = (experiment.TrialEvent & trk & 'trial_event_type="go"').fetch1('trial_event_time')
    tvec = np.arange(len(trk_feat)) / tracking_fs - float(go_time)

    b, a = signal.butter(5, (5, 15), btype='band', fs=tracking_fs)
    filt_trk_feat = signal.filtfilt(b, a, trk_feat)

    analytic_signal = signal.hilbert(filt_trk_feat)
    insta_amp = np.abs(analytic_signal)
    insta_phase = np.angle(analytic_signal)

    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
    fig.subplots_adjust(hspace=0.4)

    axs[0, 0].plot(tvec, trk_feat, '.k')
    axs[0, 0].set_title(trk_feat)
    axs[1, 0].plot(tvec, filt_trk_feat, '.k')
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

    return fig


def plot_windowed_jaw_phase_dist(session_key, xlim=(-0.12, 0.3), w_size=0.01, bin_counts=20):
    trks = (tracking.Tracking.JawTracking * experiment.BehaviorTrial & _side_cam
            & session_key & experiment.TrialEvent)
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & _side_cam & session_key).fetch1('sampling_rate'))

    tr_ids, jaws, trial_instructs, go_times = (trks * experiment.TrialEvent & 'trial_event_type="go"').fetch(
        'trial', 'jaw_y', 'trial_instruction', 'trial_event_time')

    flattened_jaws = np.hstack(jaws)
    jsize = np.cumsum([0] + [j.size for j in jaws])
    _, phase = compute_insta_phase_amp(flattened_jaws, tracking_fs, freq_band = (5, 15))
    stacked_insta_phase = [phase[start: end] for start, end in zip(jsize[:-1], jsize[1:])]

    # realign and segment - return trials x times
    insta_phase = np.vstack(get_trial_track(session_key, tr_ids, stacked_insta_phase,
                                            trial_instructs, go_times, tracking_fs, xlim))

    tvec = np.linspace(xlim[0], xlim[1], insta_phase.shape[1])
    windows = np.arange(xlim[0], xlim[1], w_size)

    # plot
    col_counts = 8
    row_counts = int(np.ceil(len(windows) / col_counts))
    fig, axs = plt.subplots(row_counts, col_counts,
                            figsize=(16, 2.5*row_counts),
                            subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.6, hspace=0.3)
    [a.axis('off') for a in axs.flatten()]

    # non-overlapping windowed histogram
    for w_start, ax in zip(windows, axs.flatten()):
        phase = insta_phase[:, np.logical_and(tvec >= w_start, tvec <= w_start + w_size)].flatten()
        plot_polar_histogram(phase, ax, bin_counts=bin_counts)
        ax.set_xlabel(f'{w_start*1000:.0f} to {(w_start + w_size)*1000:.0f}ms', fontweight='bold')
        ax.axis('on')

    return fig


def plot_jaw_phase_dist(session_key, xlim=(-0.12, 0.3), bin_counts=20):
    trks = (tracking.Tracking.JawTracking * experiment.BehaviorTrial & _side_cam & session_key & experiment.TrialEvent)
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & _side_cam & session_key).fetch1('sampling_rate'))

    l_trial_trk = trks & 'trial_instruction="left"' & 'early_lick="no early"'
    r_trial_trk = trks & 'trial_instruction="right"' & 'early_lick="no early"'

    insta_phases = []
    for trial_trks in (l_trial_trk, r_trial_trk):
        tr_ids, jaws, trial_instructs, go_times = (trial_trks * experiment.TrialEvent & 'trial_event_type="go"').fetch(
            'trial', 'jaw_y', 'trial_instruction', 'trial_event_time')

        flattened_jaws = np.hstack(jaws)
        jsize = np.cumsum([0] + [j.size for j in jaws])
        _, phase = compute_insta_phase_amp(flattened_jaws, tracking_fs, freq_band = (5, 15))
        stacked_insta_phase = [phase[start: end] for start, end in zip(jsize[:-1], jsize[1:])]

        # realign and segment - return trials x times
        insta_phases.append(np.vstack(get_trial_track(session_key, tr_ids, stacked_insta_phase,
                                                      trial_instructs, go_times, tracking_fs, xlim)))

    l_insta_phase, r_insta_phase = insta_phases

    fig, axs = plt.subplots(1, 2, figsize=(12, 8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.6)

    plot_polar_histogram(l_insta_phase.flatten(), axs[0], bin_counts=bin_counts)
    axs[0].set_title('left lick trials', loc='left', fontweight='bold')
    plot_polar_histogram(r_insta_phase.flatten(), axs[1], bin_counts=bin_counts)
    axs[1].set_title('right lick trials', loc='left', fontweight='bold')

    return fig


def plot_polar_histogram(data, ax=None, bin_counts=30):
    """
    :param data: phase in rad
    :param ax: axes to plot
    :param bin_counts: bin number for histograph
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))

    bottom = 2
    # theta = np.linspace(0.0, 2 * np.pi, bin_counts, endpoint=False)
    radii, tick = np.histogram(data, bins=bin_counts)
    # width of each bin on the plot
    width = (2 * np.pi) / bin_counts
    # make a polar plot
    ax.bar(tick[1:], radii, width=width, bottom=bottom)
    # set the label starting from East
    ax.set_theta_zero_location("E")
    # clockwise
    ax.set_theta_direction(1)


def get_trial_track(session_key, tr_ids, data, trial_instructs, go_times, fs, xlim):
    """
    Realign and segment the "data" - returning trial x time
    Realign to first left lick if 'trial_instructs' is 'left'
            or first rigth lick if 'trial_instructs' is 'right'
            or 'go cue' if no lick found
    Segment based on 'xlim'
    """

    d_length = int(np.floor((xlim[1] - xlim[0]) * fs) - 1)

    for tr_id, jaw, trial_instruct, go_time in zip(tr_ids, data, trial_instructs, go_times):

        first_lick_time = (experiment.ActionEvent & session_key
                           & {'trial': tr_id} & {'action_event_type': f'{trial_instruct} lick'}).fetch(
                'action_event_time', order_by='action_event_time', limit=1)

        align_time = first_lick_time[0] if first_lick_time.size > 0 else go_time

        t = np.arange(len(jaw)) / fs - float(align_time)
        segmented_jaw = jaw[np.logical_and(t >= xlim[0], t <= xlim[1])]
        if len(segmented_jaw) >= d_length:
            yield segmented_jaw[:d_length]


def compute_insta_phase_amp(data, fs, freq_band=(5, 15)):
    """
    :param data: trial x time
    :param fs: sampling rate
    :param freq_band: frequency band for bandpass
    """

    if data.ndim > 1:
        trial_count, time_count = data.shape
        # flatten
        data = data.reshape(-1)

    # band pass
    b, a = signal.butter(5, freq_band, btype='band', fs=fs)
    data = signal.filtfilt(b, a, data)
    # hilbert
    analytic_signal = signal.hilbert(data)
    insta_amp = np.abs(analytic_signal)
    insta_phase = np.angle(analytic_signal)

    if data.ndim > 1:
        return insta_amp.reshape((trial_count, time_count)), insta_phase.reshape((trial_count, time_count))
    else:
        return insta_amp, insta_phase


def get_event_locked_tracking_insta_phase(trials, event, tracking_feature):
    """
    Get instantaneous phase of the jaw movement, at the time of the specified "event", for each of the specified "trials"
    :param trials: query of the SessionTrial - note: the subsequent fetch() will be order_by='trial'
    :param event: "event" can be
     + a list of time equal length to the trials - specifying the time of each trial to extract the insta-phase
     + a single string, representing the event-name to extract the time for each trial
        In such case, the "event" can be the events in
            + experiment.TrialEvent
            + experiment.ActionEvent
        In the case that multiple of such "event_name" are found in a trial, the 1st one will be selected
            (e.g. multiple "lick left", then the 1st "lick left" is selected)
    :param tracking_feature: any attribute name under the tracking.Tracking table - e.g. 'jaw_y', 'tongue_x', etc.
    :return: list of instantaneous phase, in the same order of specified "trials"
    """

    trials = trials.proj()

    tracking_fs = (tracking.TrackingDevice & tracking.Tracking & trials).fetch('sampling_rate')
    if len(set(tracking_fs)) > 1:
        raise Exception('Multiple tracking Fs found!')
    else:
        tracking_fs = float(tracking_fs[0])

    # ---- process the "tracking_feature" input ----

    if tracking_feature not in _tracked_nose_features + _tracked_tongue_features + _tracked_jaw_features:
        print(f'Unknown tracking type: {tracking_feature}\nAvailable tracking types are: {_tracked_nose_features + _tracked_tongue_features + _tracked_jaw_features}')
        return

    for trk_types, trk_tbl in zip((_tracked_nose_features, _tracked_tongue_features, _tracked_jaw_features),
                                  (tracking.Tracking.NoseTracking, tracking.Tracking.TongueTracking, tracking.Tracking.JawTracking)):
        if tracking_feature in trk_types:
            d_tbl = trk_tbl

    # ---- process the "event" input ----
    if isinstance(event, (list, np.ndarray)):
        assert len(event) == len(trials)
        tr_ids, trk_data = trials.aggr(d_tbl, trk_data=tracking_feature, keep_all_rows=True).fetch(
            'trial', 'trk_data', order_by='trial')

        eve_idx = np.array(event).astype(float) * tracking_fs

    elif isinstance(event, str):
        trial_event_types = experiment.TrialEventType.fetch('trial_event_type')
        action_event_types = experiment.ActionEventType.fetch('action_event_type')

        if event in trial_event_types:
            event_tbl = experiment.TrialEvent
            eve_type_attr = 'trial_event_type'
            eve_time_attr = 'trial_event_time'
        elif event in action_event_types:
            event_tbl = experiment.ActionEvent
            eve_type_attr = 'action_event_type'
            eve_time_attr = 'trial_event_time'
        else:
            print(f'Unknown event: {event}\nAvailable events are: {list(trial_event_types) + list(action_event_types)}')
            return

        tr_ids, trk_data, eve_times = trials.aggr(d_tbl, trk_data=tracking_feature, keep_all_rows=True).aggr(
            event_tbl & {eve_type_attr: event}, 'trk_data', event_time=f'min({eve_time_attr})', keep_all_rows=True).fetch(
            'trial', 'trk_data', 'event_time', order_by='trial')

        eve_idx = eve_times.astype(float) * tracking_fs

    else:
        print('Unknown "event" argument!')
        return

    # ---- the computation part ----

    # for trials with no jaw data (None), set to np.nan array
    no_trk_trid = [idx for idx, jaw in enumerate(trk_data) if jaw is None]
    with_trk_trid = np.array(list(set(range(len(trk_data))) ^ set(no_trk_trid))).astype(int)

    if len(with_trk_trid) == 0:
        print(f'The specified trials do not have any {tracking_feature}')
        return

    trk_data = [d for d in trk_data if d is not None]

    flattened_jaws = np.hstack(trk_data)
    jsize = np.cumsum([0] + [j.size for j in trk_data])
    _, phase = compute_insta_phase_amp(flattened_jaws, tracking_fs, freq_band=(5, 15))
    stacked_insta_phase = [phase[start: end] for start, end in zip(jsize[:-1], jsize[1:])]

    trial_eve_insta_phase = [stacked_insta_phase[np.where(with_trk_trid == tr_id)[0][0]][int(e_idx)]
                             if not np.isnan(e_idx) and tr_id in with_trk_trid else np.nan
                             for tr_id, e_idx in enumerate(eve_idx)]

    return trial_eve_insta_phase
