from pipeline import ephys
from pipeline import tracking
from pipeline import experiment
from pipeline.plot import behavior_plot
from pipeline.mtl_analysis import helper_functions
from pipeline import lab
import datajoint as dj

from scipy import signal
from scipy import optimize

import matplotlib.pyplot as plt
import numpy as np

#%%
session=experiment.Session & 'subject_id="2897"' & {'session': 1}
session_key=session.fetch('KEY')

#%% get traces and phase
good_units=ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & session_key & {'insertion_number': 2} & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'

unit_keys=good_units.fetch('KEY')

traces = tracking.Tracking.JawTracking & session_key & {'tracking_device': 'Camera 3'}
session_traces = traces.fetch('jaw_y')
traces_length = [len(d) for d in session_traces]
sample_number = int(np.median(traces_length))
good_trial_ind = np.where(np.array(traces_length) == sample_number)[0]
good_traces = session_traces[good_trial_ind]
good_traces = np.vstack(good_traces)

fs=(tracking.TrackingDevice() & 'tracking_device="Camera 3"').fetch1('sampling_rate')

amp, phase=behavior_plot.compute_insta_phase_amp(good_traces, float(fs), freq_band=(5, 15))
phase = phase + np.pi

#%% compute phase and MI
for j in range(20,21):
    unit_key=unit_keys[j]
    all_spikes=(ephys.Unit.TrialSpikes & unit_key).fetch('spike_times')
    good_spikes = np.array(all_spikes[good_trial_ind]*float(fs)) # get good spikes
    good_spikes = [d.astype(int) for d in good_spikes]

    for i, d in enumerate(good_spikes):
        good_spikes[i] = d[d < 1470]

    all_phase = []
    for trial_idx in range(len(good_spikes)):
        all_phase.append(phase[trial_idx][good_spikes[trial_idx]])

    all_phase=np.hstack(all_phase)
    
    behavior_plot.plot_polar_histogram(all_phase)
    n_bins = 20
    tofity, tofitx = np.histogram(all_phase, bins=n_bins)
    baseline, tofitx = np.histogram(phase, bins=n_bins)  
    tofitx = tofitx[:-1] + (tofitx[1] - tofitx[0])/2
    tofity = tofity / baseline * float(fs)
    
    # plot raw trace
    helper_functions.plot_tracking(session_key, unit_key)
       
    preferred_phase,modulation_index=helper_functions.compute_phase_tuning(tofitx, tofity)
    
#%% plotting
helper_functions.plot_tuning(session_key, unit_key)
