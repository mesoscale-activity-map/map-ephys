from pipeline import lab, experiment, ephys, tracking, oralfacial_analysis

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
    try:
        params_h, pcov = optimize.curve_fit(vonMise_f, datax, datay, p0 = [1, datax[max_idx], max_fit_y-min_fit_y, min_fit_y], bounds=(0, [np.pi/2, 2*np.pi, max_fit_y+min_fit_y, max_fit_y]))
    except:
        print('fitting error')
        params_h = [1, datax[max_idx], max_fit_y-min_fit_y, min_fit_y]
    residuals = datay - vonMise_f(datax, *params_h) 
    r_2_h = r_squared(datay, residuals)

    try:
        params_0, pcov = optimize.curve_fit(vonMise_f, datax, datay, p0 = [1, 0, max_fit_y-min_fit_y, min_fit_y], bounds=(0, [np.pi/2, 2*np.pi, max_fit_y+min_fit_y, max_fit_y]))
    except:
        print('fitting error')
        params_0 = [1, datax[max_idx], max_fit_y-min_fit_y, min_fit_y]
    residuals = datay - vonMise_f(datax, *params_0) 
    r_2_0 = r_squared(datay, residuals)
    
    if r_2_0>r_2_h:
        params = params_0
    else:
        params = params_h
        
    preferred_phase=params[1]
    
    r_max=vonMise_f(params[1], params[0], params[1], params[2], params[3])
    r_min=vonMise_f(params[1]+np.pi, params[0], params[1], params[2], params[3])
    # get MI
    modulation_index=(r_max-r_min)/r_max
        
    return preferred_phase, modulation_index[0]

def r_squared(datay, residuals):
    ss_res = np.sum(residuals**2) 
    ss_tot = np.sum((datay-np.mean(datay))**2)   
    r_2 = 1 - (ss_res / ss_tot)
    return r_2

def min_dist(x1, x2):
    minD = np.abs(x1 - x2)
   
    minD1 = np.mod(minD, 2*np.pi)
    
    minD1 = np.array([minD1]) if isinstance(minD1, (float, np.float64)) else minD1

    minD1[minD1 > np.pi] = 2*np.pi - minD1[minD1 > np.pi]
    return minD1

def vonMise_f(x, std, mean, amp, baseline):
    return amp * np.exp(-0.5 * (min_dist(x, np.full_like(x, mean))/std)**2) + baseline

def plot_all_traces(session_key, unit_key,
                  tracking_feature='jaw_y', camera_key=_side_cam,
                  trial_offset=0, trial_limit=3, xlim=(0, 5), axs=None):
    """
    Plot jaw movement per trial, time-locked to trial-onset, with spike times overlay
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
    
    if not experiment.Breathing & session_key:
        print('No breathing data')
        return

    breathing = (experiment.Breathing & session_key & ephys.Unit.TrialSpikes)
    
    if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & session_key)) != len(breathing):
        print(f'Mismatch in tracking trial and ephys trial number: {session_key}')
        return
    
    num_frame = 1471
    traces = tracking.Tracking.JawTracking & session_key & {'tracking_device': 'Camera 4'}
    
    if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & session_key)) != len(traces):
        print(f'Mismatch in tracking trial and ephys trial number: {session_key}')
        return
    
    session_traces_w = (oralfacial_analysis.WhiskerSVD & session_key).fetch('mot_svd')
    
    if len(session_traces_w[0][:,0]) % num_frame != 0:
        print('Bad videos in bottom view')
        return
    else:
        num_trial_w = int(len(session_traces_w[0][:,0])/num_frame)
        session_traces_w = np.reshape(session_traces_w[0][:,0], (num_trial_w, num_frame))
        trial_idx_nat = [d.astype(str) for d in np.arange(num_trial_w)]
        trial_idx_nat = sorted(range(len(trial_idx_nat)), key=lambda k: trial_idx_nat[k])
        trial_idx_nat = sorted(range(len(trial_idx_nat)), key=lambda k: trial_idx_nat[k])
        session_traces_w=session_traces_w[trial_idx_nat,:]
    
    tracking_fs = float((tracking.TrackingDevice & tracking.Tracking & camera_key & session_key).fetch1('sampling_rate'))

    trial_trk_1 = trk & 'water_port="mtl-1"'
    trial_trk_2 = trk & 'water_port="mtl-2"'
    trial_trk_3 = trk & 'water_port="mtl-3"'
    trial_trk_4 = trk & 'water_port="mtl-4"'
    trial_trk_5 = trk & 'water_port="mtl-5"'
    trial_trk_6 = trk & 'water_port="mtl-6"'
    trial_trk_7 = trk & 'water_port="mtl-7"'
    trial_trk_8 = trk & 'water_port="mtl-8"'
    trial_trk_9 = trk & 'water_port="mtl-9"'

    def get_trial_all(trial_tracks, trial_breathing, trial_whisking):
        if trial_offset < 1 and isinstance(trial_offset, float):
            offset = int(len(trial_tracks) * trial_offset)
        else:
            offset = trial_offset
            
        br = trial_breathing.fetch(as_dict=True, offset=offset, limit=trial_limit, order_by='trial')
        wh = trial_whisking[offset:offset+trial_limit]
        
        for i_t, tr in enumerate(trial_tracks.fetch(as_dict=True, offset=offset, limit=trial_limit, order_by='trial')):
            trk_feat = tr[tracking_feature]
            tongue_out_bool = tr['tongue_likelihood'] > 0.95

            sample_counts = len(trk_feat)
            tvec = np.arange(sample_counts) / tracking_fs

            spike_times = (ephys.Unit.TrialSpikes & tr & unit_key).fetch1('spike_times')
            
            br_trace = br[i_t]['breathing']       
            tvec_b = br[i_t]['breathing_timestamps']
            wh_trace = wh[i_t]
            yield trk_feat, tongue_out_bool, spike_times, tvec, br_trace, tvec_b, wh_trace   
            
    fig = None
    if axs is None:
        fig, axs = plt.subplots(3, 3, figsize=(16, 16))
    assert len(axs) == 9

    h_spacing = 80
    h_spacing_b = 2500
    h_spacing_w = 600
    for trial_tracks, ax, ax_name, spk_color in zip((trial_trk_3,trial_trk_6,trial_trk_9,trial_trk_2,trial_trk_5,trial_trk_8,trial_trk_1,trial_trk_4,trial_trk_7), axs.flatten(),
                                                    ('top-left trials', 'top-mid trials','top-right trials','mid-left trials','mid trials','mid-right trials','bot-left trials','bot-mid trials','bot-right trials'),
                                                    ('k','k','k','k','k','k','k','k','k')):
        trial_breathing = breathing & trial_tracks
        trials = trial_tracks.fetch('trial', order_by='trial')
        trial_whisking = session_traces_w[trials-1]

        for tr_id, (trk_feat, tongue_out_bool, spike_times, tvec, br, tvec_b, wh) in enumerate(get_trial_all(trial_tracks, trial_breathing, trial_whisking)):

            ax.plot(tvec, trk_feat/h_spacing + tr_id*4-0.5, '.k', markersize=1)
            ax.plot(tvec[tongue_out_bool], trk_feat[tongue_out_bool]/h_spacing + tr_id*4-0.5, '.', color='lime', markersize=2)
            ax.plot(spike_times, np.full_like(spike_times, 1) + tr_id*4+3,
                        color=spk_color, marker='$I$', linestyle='None', markersize=6)
            ax.plot(tvec_b, br/h_spacing_b + tr_id*4+0.7, 'b', linewidth=1)
            ax.plot(tvec, wh/h_spacing_w + tr_id*4+0.8, 'r', linewidth=1)
            ax.set_title(ax_name)
            ax.axvline(x=0, linestyle='--', color='k')           

            # cosmetic
            ax.set_xlim(xlim)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if ax_name == 'bot-mid trials':
                ax.set_xlabel('s')

    return fig

fig1 = plt.figure(figsize=(16, 16))
gs = GridSpec(3, 4)

plot_all_traces(daves_session_key,unit_key,axs=np.array([fig1.add_subplot(gs[row_idx, col_idx])
                              for row_idx, col_idx in itertools.product(
                        range(0,3), range(0,3))]))
mtl_plot.plot_jaw_tuning(unit_key, axs=fig1.add_subplot(gs[0, 3], polar=True))
mtl_plot.plot_whisker_tuning(unit_key, axs=fig1.add_subplot(gs[1, 3], polar=True))
mtl_plot.plot_breathing_tuning(unit_key, axs=fig1.add_subplot(gs[2, 3], polar=True))
fig1.subplots_adjust(wspace=0.2)
fig1.subplots_adjust(hspace=0.8)


def plot_tracking(session_key, unit_key,
                  tracking_feature='jaw_y', camera_key=_side_cam,
                  trial_offset=0, trial_limit=10, xlim=(0, 5), axs=None):
    """
    Plot jaw movement per trial, time-locked to trial-onset, with spike times overlay
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

    trial_trk_1 = trk & 'water_port="mtl-1"'
    trial_trk_2 = trk & 'water_port="mtl-2"'
    trial_trk_3 = trk & 'water_port="mtl-3"'
    trial_trk_4 = trk & 'water_port="mtl-4"'
    trial_trk_5 = trk & 'water_port="mtl-5"'
    trial_trk_6 = trk & 'water_port="mtl-6"'
    trial_trk_7 = trk & 'water_port="mtl-7"'
    trial_trk_8 = trk & 'water_port="mtl-8"'
    trial_trk_9 = trk & 'water_port="mtl-9"'

    def get_trial_track(trial_tracks):
        if trial_offset < 1 and isinstance(trial_offset, float):
            offset = int(len(trial_tracks) * trial_offset)
        else:
            offset = trial_offset

        for tr in trial_tracks.fetch(as_dict=True, offset=offset, limit=trial_limit, order_by='trial'):
            trk_feat = tr[tracking_feature]
            tongue_out_bool = tr['tongue_likelihood'] > 0.95

            sample_counts = len(trk_feat)
            tvec = np.arange(sample_counts) / tracking_fs

            spike_times = (ephys.Unit.TrialSpikes & tr & unit_key).fetch1('spike_times')

            yield trk_feat, tongue_out_bool, spike_times, tvec

    fig = None
    if axs is None:
        fig, axs = plt.subplots(3, 3, figsize=(16, 16))
    assert len(axs) == 9

    h_spacing = 150
    for trial_tracks, ax, ax_name, spk_color in zip(
            (trial_trk_3,trial_trk_6,trial_trk_9,trial_trk_2,trial_trk_5,trial_trk_8,trial_trk_1,trial_trk_4,trial_trk_7),
            axs.flatten(),
            ('top-left trials', 'top-mid trials','top-right trials','mid-left trials','mid trials','mid-right trials','bot-left trials','bot-mid trials','bot-right trials'),
            ('k','k','k','k','k','k','k','k','k')):
        if not len(trial_tracks):
            ax.remove()
            continue
        for tr_id, (trk_feat, tongue_out_bool, spike_times, tvec) in enumerate(get_trial_track(trial_tracks)):
            ax.plot(tvec, trk_feat + tr_id * h_spacing, '.k', markersize=1)
            ax.plot(tvec[tongue_out_bool], trk_feat[tongue_out_bool] + tr_id * h_spacing, '.', color='lime', markersize=2)
            ax.plot(spike_times, np.full_like(spike_times, trk_feat[tongue_out_bool].mean() + h_spacing/3) + tr_id * h_spacing,
                        color=spk_color, marker='$I$', linestyle='None', markersize=6)
            ax.set_title(ax_name)
            ax.axvline(x=0, linestyle='--', color='k')

            # cosmetic
            ax.set_xlim(xlim)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if not ax_name.startswith('bot'):
                ax.set_xticks([])
            if ax_name == 'bot-mid trials':
                ax.set_xlabel('s')

    return fig

def plot_breathing(session_key, unit_key, trial_offset=0, trial_limit=10, xlim=(0, 5), axs=None):
    """
    Plot breathing per trial, time-locked to trial-onset, with spike times overlay
    :param session_key: session where the trials are from
    :param unit_key: unit for spike times overlay
    :param trial_offset: index of trial to plot from (if a decimal between 0 and 1, indicates the proportion of total trial to plot from)
    :param trial_limit: number of trial to plot
    """

    if not experiment.Breathing & session_key:
        print('No breathing data')
        return

    breathing = (experiment.Breathing & session_key & ephys.Unit.TrialSpikes)
    
    if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & session_key)) != len(breathing):
        print(f'Mismatch in tracking trial and ephys trial number: {session_key}')
        return

    def get_trial_breathing(trial_breathing):
        if trial_offset < 1 and isinstance(trial_offset, float):
            offset = int(len(trial_breathing) * trial_offset)
        else:
            offset = trial_offset

        for br in trial_breathing.fetch(as_dict=True, offset=offset, limit=trial_limit, order_by='trial'):
            br_trace = br['breathing']       
            tvec = br['breathing_timestamps']
            spike_times = (ephys.Unit.TrialSpikes & br & unit_key).fetch1('spike_times')
            yield br_trace, spike_times, tvec
    
    fig = None
    if axs is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        
    h_spacing = 3000

    for tr_id, (br, spike_times, tvec) in enumerate(get_trial_breathing(breathing)):
        ax.plot(tvec, br + tr_id * h_spacing, 'k')
        ax.plot(spike_times, np.full_like(spike_times, br.mean() + h_spacing/2) + tr_id * h_spacing, color='k', marker='$I$', linestyle='None', markersize=12)
        ax.set_title('breathing')
        ax.axvline(x=0, linestyle='--', color='k')

        # cosmetic
        ax.set_xlim(xlim)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('s')

    return fig

def plot_whisking(session_key, unit_key, trial_offset=0, trial_limit=10, xlim=(0, 5), axs=None):
    """
    Plot whisking per trial, time-locked to trial-onset, with spike times overlay
    :param session_key: session where the trials are from
    :param unit_key: unit for spike times overlay
    :param trial_offset: index of trial to plot from (if a decimal between 0 and 1, indicates the proportion of total trial to plot from)
    :param trial_limit: number of trial to plot
    """
    
    num_frame = 1471
    traces = tracking.Tracking.JawTracking & session_key & {'tracking_device': 'Camera 4'}
    
    if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & session_key)) != len(traces):
        print(f'Mismatch in tracking trial and ephys trial number: {session_key}')
        return
    
    session_traces_w = (oralfacial_analysis.WhiskerSVD & session_key).fetch('mot_svd')
    
    if len(session_traces_w[0][:,0]) % num_frame != 0:
        print('Bad videos in bottom view')
        return
    else:
        num_trial_w = int(len(session_traces_w[0][:,0])/num_frame)
        session_traces_w = np.reshape(session_traces_w[0][:,0], (num_trial_w, num_frame))
    trial_idx_nat = [d.astype(str) for d in np.arange(num_trial_w)]
    trial_idx_nat = sorted(range(len(trial_idx_nat)), key=lambda k: trial_idx_nat[k])
    trial_idx_nat = sorted(range(len(trial_idx_nat)), key=lambda k: trial_idx_nat[k])
    session_traces_w=session_traces_w[trial_idx_nat,:]
                                   
    fs=(tracking.TrackingDevice & 'tracking_device="Camera 4"').fetch1('sampling_rate')
    
    def get_trial_whisking(trial_whisking):
        if trial_offset < 1 and isinstance(trial_offset, float):
            offset = int(len(trial_whisking) * trial_offset)
        else:
            offset = trial_offset

        for tr_id, wh in enumerate(trial_whisking[offset:offset+trial_limit]):
            wh_trace = wh

            sample_counts = len(wh_trace)
            tvec = np.arange(sample_counts) / fs
            spike_times = (ephys.Unit.TrialSpikes & {'trial': tr_id+1+offset} & unit_key).fetch1('spike_times')
            yield wh_trace, spike_times, tvec
    
    fig = None
    if axs is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        
    h_spacing =1000

    for tr_id, (wh, spike_times, tvec) in enumerate(get_trial_whisking(session_traces_w)):
        ax.plot(tvec, wh + tr_id * h_spacing, 'k')
        ax.plot(spike_times, np.full_like(spike_times, wh.mean() + h_spacing/2) + tr_id * h_spacing, color='k', marker='$I$', linestyle='None', markersize=12)
        ax.set_title('whisking')
        ax.axvline(x=0, linestyle='--', color='k')

        # cosmetic
        ax.set_xlim(xlim)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('s')

    return fig

def plot_jaw_tuning(unit_key, axs=None):
        
    tofitx, tofity = (oralfacial_analysis.JawTuning() & unit_key).fetch1('jaw_x', 'jaw_y')
    max_fit_y = np.round(np.amax(tofity), 1)

    fig = None
    if axs is None:
        fig, axs = plt.subplots(subplot_kw={'projection': 'polar'})

    axs.plot(np.append(tofitx, tofitx[0]), np.append(tofity, tofity[0], color='k'))
    axs.set_rmax(max_fit_y)
    axs.set_rticks([0, max_fit_y])  # Less radial ticks
    axs.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    axs.grid(False)
    axs.set_title('Jaw tuning')
    xT = plt.xticks()[0]
    xL = ['0', '', r'$\frac{\pi}{2}$', '', r'$\pi$', '', r'$\frac{3\pi}{2}$', '']
    plt.xticks(xT, xL)
    
    return fig

def plot_breathing_tuning(unit_key, axs=None):

    tofitx, tofity = (oralfacial_analysis.BreathingTuning() & unit_key).fetch1('breathing_x','breathing_y')
    max_fit_y=np.round(np.amax(tofity),1)

    fig = None
    if axs is None:
        fig, axs = plt.subplots(subplot_kw={'projection': 'polar'})

    axs.plot(np.append(tofitx,tofitx[0]), np.append(tofity,tofity[0]))
    axs.set_rmax(max_fit_y)
    axs.set_rticks([0, max_fit_y])  # Less radial ticks
    axs.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    axs.grid(False)
    axs.set_title('Breathing tuning')
    xT = plt.xticks()[0]
    xL = ['0', '', r'$\frac{\pi}{2}$', '', r'$\pi$', '', r'$\frac{3\pi}{2}$', '']
    plt.xticks(xT, xL)
    
    return fig

def plot_whisker_tuning(unit_key, axs=None):

    tofitx, tofity = (oralfacial_analysis.WhiskerTuning() & unit_key).fetch1('whisker_x','whisker_y')
    max_fit_y=np.round(np.amax(tofity),1)
    
    fig = None
    if axs is None:
        fig, axs = plt.subplots(subplot_kw={'projection': 'polar'})

    axs.plot(np.append(tofitx, tofitx[0]), np.append(tofity, tofity[0], color='r'))
    axs.set_rmax(max_fit_y)
    axs.set_rticks([0, max_fit_y])  # Less radial ticks
    axs.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    axs.grid(False)
    axs.set_title('Whisker tuning')
    xT = plt.xticks()[0]
    xL = ['0', '', r'$\frac{\pi}{2}$', '', r'$\pi$', '', r'$\frac{3\pi}{2}$', '']
    plt.xticks(xT, xL)
    
    return fig

def water2subject(water,date):
    subject_id = (lab.WaterRestriction & {'water_restriction_number': water}).fetch('subject_id')
    session_num = (experiment.Session() * lab.WaterRestriction & {'water_restriction_number': water, 'session_date': date}).fetch('session')
    return subject_id, session_num

def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees", start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """

    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='C0', fill=True, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    max_r=np.round(np.amax(radius),0)
    ax.set_rmax(max_r)
    ax.set_rticks([0, max_r])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)