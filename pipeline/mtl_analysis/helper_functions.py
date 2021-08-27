from pipeline import ephys
from pipeline import tracking
from pipeline import experiment
from pipeline.plot import behavior_plot
from pipeline import lab

from scipy import signal
from scipy import optimize

import matplotlib.pyplot as plt
import numpy as np

# ======== Define some useful variables ==============

_side_cam = {'tracking_device': 'Camera 3'}

_tracked_nose_features = tracking.Tracking.NoseTracking.heading.secondary_attributes
_tracked_tongue_features = tracking.Tracking.TongueTracking.heading.secondary_attributes
_tracked_jaw_features = tracking.Tracking.JawTracking.heading.secondary_attributes


def compute_phase_tuning(datax, datay):
    max_fit_y=np.amax(datay)
    min_fit_y=np.amin(datay)
    max_idx=np.where(datay == max_fit_y)
    if np.size(max_idx)>1:
        max_idx=max_idx[0][0]
    # fit curve
    params, pcov = optimize.curve_fit(vonMise_f, datax, datay, p0 = [1, datax[max_idx], max_fit_y-min_fit_y, min_fit_y], bounds=(0, [np.pi/2, 2*np.pi, max_fit_y+min_fit_y, max_fit_y]))
    preferred_phase=params[1]
    
    r_max=vonMise_f(params[1], params[0], params[1], params[2], params[3])
    r_min=vonMise_f(params[1]+np.pi, params[0], params[1], params[2], params[3])
    # get MI
    modulation_index=(r_max-r_min)/r_max
        
    return preferred_phase, modulation_index[0]


def min_dist(x1, x2):
    minD = np.abs(x1 - x2)
   
    minD1 = np.mod(minD, 2*np.pi)
    
    minD1 = np.array([minD1]) if isinstance(minD1, (float, np.float64)) else minD1

    minD1[minD1 > np.pi] = 2*np.pi - minD1[minD1 > np.pi]
    return minD1

def vonMise_f(x, std, mean, amp, baseline):
    return amp * np.exp(-0.5 * (min_dist(x, np.full_like(x, mean))/std)**2) + baseline

def plot_tracking(session_key, unit_key,
                  tracking_feature='jaw_y', camera_key=_side_cam,
                  trial_offset=0, trial_limit=10, xlim=(-0, 5), axs=None):
    """
    Plot jaw movement per trial, time-locked to cue-onset, with spike times overlay
    :param session_key: session where the trials are from
    :param unit_key: unit for spike times overlay
    :param tracking_feature: which tracking feature to plot (default to `jaw_y`)
    :param camera_key: tracking from which camera to plot (default to Camera 0, i.e. the side camera)
    :param trial_offset: index of trial to plot from (if a decimal between 0 and 1, indicates the proportion of total trial to plot from)
    :param trial_limit: number of trial to plot
    """

    if tracking_feature not in _tracked_nose_features + _tracked_tongue_features + _tracked_jaw_features:
        print(f'Unknown tracking type: {tracking_feature}\nAvailable tracking types are: {_tracked_nose_features + _tracked_tongue_features + _tracked_jaw_features}')
        return

    trk = (tracking.Tracking.JawTracking * tracking.Tracking.TongueTracking
           * experiment.BehaviorTrial
           * experiment.MultiTargetLickingSessionBlock.WaterPort
           * experiment.MultiTargetLickingSessionBlock.BlockTrial
           & camera_key & session_key & ephys.Unit.TrialSpikes)
    
    
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & camera_key & session_key).fetch1('sampling_rate'))

    l_trial_trk = trk & 'water_port="mtl-6"'
    r_trial_trk = trk & 'water_port="mtl-5"'

    def get_trial_track(trial_tracks):
        if trial_offset < 1 and isinstance(trial_offset, float):
            offset = int(len(trial_tracks) * trial_offset)
        else:
            offset = trial_offset

        for tr in trial_tracks.fetch(as_dict=True, offset=offset, limit=trial_limit, order_by='trial'):
            trk_feat = tr[tracking_feature]
            tongue_out_bool = tr['tongue_likelihood'] > 0.9

            sample_counts = len(trk_feat)
            tvec = np.arange(sample_counts) / tracking_fs

            spike_times = (ephys.Unit.TrialSpikes & tr & unit_key).fetch1('spike_times')

            yield trk_feat, tongue_out_bool, spike_times, tvec

    fig = None
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    assert len(axs) == 2

    h_spacing = 150
    for trial_tracks, ax, ax_name, spk_color in zip((l_trial_trk, r_trial_trk), axs,
                                                    ('left lick trials', 'right lick trials'), ('k', 'k')):
        for tr_id, (trk_feat, tongue_out_bool, spike_times, tvec) in enumerate(get_trial_track(trial_tracks)):
            ax.plot(tvec, trk_feat + tr_id * h_spacing, '.k', markersize=1)
            ax.plot(tvec[tongue_out_bool], trk_feat[tongue_out_bool] + tr_id * h_spacing, '.', color='lime', markersize=2)
            ax.plot(spike_times, np.full_like(spike_times, trk_feat[tongue_out_bool].mean() + h_spacing/10) + tr_id * h_spacing,
                    '|', color=spk_color, markersize=4)
            #ax.set_title(ax_name)
            ax.axvline(x=0, linestyle='--', color='k')

            # cosmetic
            ax.set_xlim(xlim)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    return fig

def plot_tuning(session_key, unit_key):
    num_frame = 1470
    traces = tracking.Tracking.JawTracking & session_key & {'tracking_device': 'Camera 3'}
    session_traces = traces.fetch('jaw_y', order_by='trial')
    traces_length = [len(d) for d in session_traces]
    sample_number = int(np.median(traces_length))
    good_trial_ind = np.where(np.array(traces_length) == sample_number)[0]
    good_traces = session_traces[good_trial_ind]
    good_traces = np.vstack(good_traces)
    
    fs=(tracking.TrackingDevice() & 'tracking_device="Camera 3"').fetch1('sampling_rate')
    
    amp, phase=behavior_plot.compute_insta_phase_amp(good_traces, float(fs), freq_band=(5, 15))
    phase = phase + np.pi

    all_spikes=(ephys.Unit.TrialSpikes & unit_key).fetch('spike_times', order_by='trial')
    good_spikes = np.array(all_spikes[good_trial_ind]*float(fs)) # get good spikes
    good_spikes = [d.astype(int) for d in good_spikes]

    for i, d in enumerate(good_spikes):
        good_spikes[i] = d[d < num_frame]

    all_phase = []
    for trial_idx in range(len(good_spikes)):
        all_phase.append(phase[trial_idx][good_spikes[trial_idx]])

    all_phase=np.hstack(all_phase)
    
    n_bins = 20
    tofity, tofitx = np.histogram(all_phase, bins=n_bins)
    baseline, tofitx = np.histogram(phase, bins=n_bins)  
    tofitx = tofitx[:-1] + (tofitx[1] - tofitx[0])/2
    tofity = tofity / baseline * float(fs)
    max_fit_y=np.amax(tofity)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(np.append(tofitx,tofitx[0]), np.append(tofity,tofity[0]))
    ax.set_rmax(max_fit_y)
    ax.set_rticks([0, max_fit_y])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(False)
    ax.set_xticklabels([])
    
    return fig

def psth(spikes, trigger):
    
    return psth

def water2subject(water,date):
    subject_id = (lab.WaterRestriction & {'water_restriction_number': water}).fetch('subject_id')
    session_num = (experiment.Session() * lab.WaterRestriction & {'water_restriction_number': water, 'session_date': date}).fetch('session')
    return subject_id, session_num