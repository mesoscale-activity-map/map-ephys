from pipeline import ephys, tracking, experiment, oralfacial_analysis
from pipeline.plot import behavior_plot
from pipeline.mtl_analysis import helper_functions
import datajoint as dj
v_tracking = dj.create_virtual_module('tracking', 'map_v2_tracking')

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
#%%
water='DL025'
date='2021-06-25'
subject, session = helper_functions.water2subject(water, date)
#%%
# tongue
#session=experiment.Session & 'subject_id="1114"' & {'session': 9}
# jaw 20
session=experiment.Session & 'subject_id="2897"' & {'session': 1}
session_key=session.fetch('KEY')
#%% get traces and phase
good_units=ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & session_key & {'insertion_number': 2} & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'
unit_keys=good_units.fetch('KEY')
bin_width = 0.017

# from the cameras
tongue_thr = 0.95
traces_s = tracking.Tracking.TongueTracking & session_key & {'tracking_device': 'Camera 3'}
traces_b = tracking.Tracking.TongueTracking & session_key & {'tracking_device': 'Camera 4'}
session_traces_s_l = traces_s.fetch('tongue_likelihood', order_by='trial')
session_traces_b_l = traces_b.fetch('tongue_likelihood', order_by='trial')
trial_key=(v_tracking.TongueTracking3DBot & session_key).fetch('trial', order_by='trial')
test_t = trial_key[::5]
trial_key=np.setdiff1d(trial_key,test_t)
session_traces_s_l = session_traces_s_l[trial_key-1]
session_traces_b_l = session_traces_b_l[trial_key-1]
session_traces_s_l = np.vstack(session_traces_s_l)
session_traces_b_l = np.vstack(session_traces_b_l)
session_traces_t_l = session_traces_b_l
session_traces_t_l[np.where((session_traces_s_l > tongue_thr) & (session_traces_b_l > tongue_thr))] = 1
session_traces_t_l[np.where((session_traces_s_l <= tongue_thr) & (session_traces_b_l <= tongue_thr))] = 0
session_traces_t_l = np.hstack(session_traces_t_l)

# from 3D calibration
traces_s = v_tracking.JawTracking3DSid & session_key & [{'trial': tr} for tr in trial_key]
traces_b = v_tracking.TongueTracking3DBot & session_key & [{'trial': tr} for tr in trial_key]
session_traces_s_y, session_traces_s_x, session_traces_s_z = traces_s.fetch('jaw_y', 'jaw_x', 'jaw_z', order_by='trial')
session_traces_b_y, session_traces_b_x, session_traces_b_z = traces_b.fetch('tongue_y', 'tongue_x', 'tongue_z', order_by='trial')
session_traces_s_y = np.vstack(session_traces_s_y)
session_traces_s_x = np.vstack(session_traces_s_x)
session_traces_s_z = np.vstack(session_traces_s_z)
session_traces_b_y = np.vstack(session_traces_b_y) 
session_traces_b_x = np.vstack(session_traces_b_x)
session_traces_b_z = np.vstack(session_traces_b_z)
traces_len = np.size(session_traces_b_z, axis = 1)
num_trial = np.size(session_traces_b_z, axis = 0)

# phase caculation
fs=(tracking.TrackingDevice() & 'tracking_device="Camera 3"').fetch1('sampling_rate')
amp_x, phase_x=behavior_plot.compute_insta_phase_amp(session_traces_s_x, float(fs), freq_band=(5, 15))
amp_y, phase_y=behavior_plot.compute_insta_phase_amp(session_traces_s_y, float(fs), freq_band=(5, 15))
amp_z, phase_z=behavior_plot.compute_insta_phase_amp(session_traces_s_z, float(fs), freq_band=(5, 15))
phase_x = phase_x + np.pi
phase_y = phase_y + np.pi
phase_z = phase_z + np.pi

# format the video data
session_traces_s_y = np.hstack(session_traces_s_y)
session_traces_s_x = np.hstack(session_traces_s_x)
session_traces_s_z = np.hstack(session_traces_s_z)
session_traces_b_y = np.hstack(session_traces_b_y)
session_traces_b_x = np.hstack(session_traces_b_x)
session_traces_b_z = np.hstack(session_traces_b_z)
# -- moving-average and down-sample
window_size = int(bin_width/0.0034)  # sample
kernel = np.ones(window_size) / window_size
session_traces_s_x = np.convolve(session_traces_s_x, kernel, 'same')
session_traces_s_x = session_traces_s_x[window_size::window_size]
session_traces_s_y = np.convolve(session_traces_s_y, kernel, 'same')
session_traces_s_y = session_traces_s_y[window_size::window_size]
session_traces_s_z = np.convolve(session_traces_s_z, kernel, 'same')
session_traces_s_z = session_traces_s_z[window_size::window_size]
session_traces_b_x = np.convolve(session_traces_b_x, kernel, 'same')
session_traces_b_x = session_traces_b_x[window_size::window_size]
session_traces_b_y = np.convolve(session_traces_b_y, kernel, 'same')
session_traces_b_y = session_traces_b_y[window_size::window_size]
session_traces_b_z = np.convolve(session_traces_b_z, kernel, 'same')
session_traces_b_z = session_traces_b_z[window_size::window_size]
session_traces_t_l = np.convolve(session_traces_t_l, kernel, 'same')
session_traces_t_l = session_traces_t_l[window_size::window_size]
session_traces_t_l[np.where(session_traces_t_l < 1)] = 0
session_traces_s_x = np.reshape(session_traces_s_x,(-1,1))
session_traces_s_y = np.reshape(session_traces_s_y,(-1,1))
session_traces_s_z = np.reshape(session_traces_s_z,(-1,1))
session_traces_b_x = np.reshape(session_traces_b_x * session_traces_t_l, (-1,1))
session_traces_b_y = np.reshape(session_traces_b_y * session_traces_t_l, (-1,1))
session_traces_b_z = np.reshape(session_traces_b_z * session_traces_t_l, (-1,1))

# get breathing
breathing, breathing_ts = (experiment.Breathing & session_key).fetch('breathing', 'breathing_timestamps', order_by='trial')
breathing = breathing[trial_key-1]
breathing_ts = breathing_ts[trial_key-1]
good_breathing = breathing
for i, d in enumerate(breathing):
    good_breathing[i] = d[breathing_ts[i] < traces_len*3.4/1000]
good_breathing = np.vstack(good_breathing)
good_breathing = np.hstack(good_breathing)
# -- moving-average
window_size = int(bin_width/(breathing_ts[0][1]-breathing_ts[0][0]))  # sample
kernel = np.ones(window_size) / window_size
good_breathing = np.convolve(good_breathing, kernel, 'same')
# -- down-sample
good_breathing = good_breathing[window_size::window_size]
good_breathing = np.reshape(good_breathing,(-1,1))

# get whisker
session_traces_w = (oralfacial_analysis.WhiskerSVD & session_key).fetch('mot_svd')
if len(session_traces_w[0][:,0]) % 1471 != 0:
    print('Bad videos in bottom view')
    #return
else:
    num_trial_w = int(len(session_traces_w[0][:,0])/1471)
    session_traces_w = np.reshape(session_traces_w[0][:,0], (num_trial_w, 1471))
    
session_traces_w = session_traces_w[trial_key-1,:]
session_traces_w = np.hstack(session_traces_w)
window_size = int(bin_width/0.0034)  # sample
kernel = np.ones(window_size) / window_size
session_traces_w = np.convolve(session_traces_w, kernel, 'same')
session_traces_w = session_traces_w[window_size::window_size]
session_traces_w = np.reshape(session_traces_w,(-1,1))
#%%
# stimulus
V_design_matrix = np.concatenate((session_traces_s_x, session_traces_s_y, session_traces_s_z, session_traces_b_x, session_traces_b_y, session_traces_b_z, good_breathing, session_traces_w), axis=1)

#set up GLM
sm_log_Link = sm.genmod.families.links.log

for j in range(20,21): # loop for each neuron
    unit_key=unit_keys[j]
    all_spikes=(ephys.Unit.TrialSpikes & unit_key).fetch('spike_times', order_by='trial')
    good_spikes = np.array(all_spikes[trial_key-1]) # get good spikes

    for i, d in enumerate(good_spikes):
        good_spikes[i] = d[d < traces_len*3.4/1000]+traces_len*3.4/1000*i

    good_spikes = np.hstack(good_spikes)
    
    y, bin_edges = np.histogram(good_spikes, np.arange(0, traces_len*3.4/1000*num_trial, bin_width))
    #y = np.reshape(y, (-1, 1))
    #y = np.concatenate((y, np.ones( [len(y),1] )), axis=1)
    
    glm_poiss = sm.GLM(y, sm.add_constant(V_design_matrix), family=sm.families.Poisson(link=sm_log_Link))

    glm_result = glm_poiss.fit()
    weights_py = glm_result.params 
    
    sst_val = sum(map(lambda bins_edges: np.power(bins_edges,2),y-np.mean(y))) 
    sse_val = sum(map(lambda bins_edges: np.power(bins_edges,2),glm_result.resid_response)) 
    r2 = 1.0 - sse_val/sst_val
    
    # Compare the difference
    print(weights_py)
#%%    

