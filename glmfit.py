from pipeline import ephys
from pipeline import tracking
from pipeline import experiment
from pipeline.plot import behavior_plot
from pipeline.mtl_analysis import helper_functions
# import datajoint as dj
# tracking = dj.create_virtual_module('tracking', 'map_v2_tracking')

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
#%%
water='DL004'
date='2021-03-08'
subject, session = helper_functions.water2subject(water, date)
#%%
session=experiment.Session & 'subject_id="2897"' & {'session': 1}
session_key=session.fetch('KEY')
#%% get traces and phase
good_units=ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & session_key & {'insertion_number': 2} & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'

unit_keys=good_units.fetch('KEY')

traces_s = tracking.Tracking.JawTracking & session_key & {'tracking_device': 'Camera 3'}
traces_b = tracking.Tracking.JawTracking & session_key & {'tracking_device': 'Camera 4'}
session_traces_s_y, session_traces_s_x = traces_s.fetch('jaw_y','jaw_x')
session_traces_b_y, session_traces_b_x = traces_b.fetch('jaw_y','jaw_x')
traces_length_s = [len(d) for d in session_traces_s_y]
traces_length_b = [len(d) for d in session_traces_b_y]
sample_number = int(np.median(traces_length_s))
good_trial_ind = np.where((np.array(traces_length_s) == sample_number) & (np.array(traces_length_b) == sample_number))[0]
good_traces_s_y = session_traces_s_y[good_trial_ind]
good_traces = np.vstack(good_traces)

fs=(tracking.TrackingDevice() & 'tracking_device="Camera 3"').fetch1('sampling_rate')

amp, phase=behavior_plot.compute_insta_phase_amp(good_traces, float(fs), freq_band=(5, 15))
phase = phase + np.pi

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
    
    # response
    y
    
    # stimulus
    V
    
    #% set up GLM
    y = np.concatenate((y, np.ones( [len(y),1] )), axis=1)
    sm_log_Link = sm.genmod.families.links.log
    glm_binom = sm.GLM(sm.add_constant(y), sm.add_constant(V_design_matrix), family=sm.families.Poisson(link=sm_log_Link))
    
    
    ## Run GLM fit
    glm_result = glm_binom.fit()
    weights_py = glm_result.params 
    
    ## Compare the difference
    print(weights_py)