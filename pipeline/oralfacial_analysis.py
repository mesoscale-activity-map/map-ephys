import numpy as np
import statsmodels.api as sm
import datajoint as dj
import pathlib
from scipy import stats
from astropy.stats import kuiper_two
from pipeline import ephys, experiment, tracking
from pipeline.ingest import tracking as tracking_ingest

from pipeline.mtl_analysis import helper_functions
from pipeline.plot import behavior_plot
from . import get_schema_name

schema = dj.schema(get_schema_name('oralfacial_analysis'))

v_oralfacial_analysis = dj.create_virtual_module('oralfacial_analysis', get_schema_name('oralfacial_analysis'))
v_tracking = dj.create_virtual_module('tracking', get_schema_name('tracking'))

@schema
class JawTuning(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    modulation_index: float
    preferred_phase: float
    jaw_x: mediumblob
    jaw_y: mediumblob
    kuiper_test: float
    di_perm: float
    """
    # mtl sessions only
    key_source = experiment.Session & ephys.Unit & tracking.Tracking & 'rig = "RRig-MTL"'
    
    def make(self, key):
        num_frame = 1470
        # get traces and phase
        good_units=ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & key & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'
        
        unit_keys=good_units.fetch('KEY')
        
        traces = tracking.Tracking.JawTracking & key & {'tracking_device': 'Camera 3'}
        
        if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & key)) != len(traces):
            print(f'Mismatch in tracking trial and ephys trial number: {key}')
            return
        
        session_traces = traces.fetch('jaw_y', order_by='trial')
        traces_length = [len(d) for d in session_traces]
        sample_number = int(np.median(traces_length))
        good_trial_ind = np.where(np.array(traces_length) == sample_number)[0]
        good_traces = session_traces[good_trial_ind]
        good_traces = np.vstack(good_traces)
        
        fs=(tracking.TrackingDevice & 'tracking_device="Camera 3"').fetch1('sampling_rate')
        
        amp, phase=behavior_plot.compute_insta_phase_amp(good_traces, float(fs), freq_band=(3, 15))
        phase = phase + np.pi
        phase_s=np.hstack(phase)
        
        # compute phase and MI
        units_jaw_tunings = []
        for unit_key in unit_keys:
            all_spikes=(ephys.Unit.TrialSpikes & unit_key).fetch('spike_times', order_by='trial')
            good_spikes = np.array(all_spikes[good_trial_ind]*float(fs)) # get good spikes and convert to indices
            good_spikes = [d.astype(int) for d in good_spikes] # convert to intergers
        
            for i, d in enumerate(good_spikes):
                good_spikes[i] = d[d < num_frame]
        
            all_phase = []
            for trial_idx in range(len(good_spikes)):
                all_phase.append(phase[trial_idx][good_spikes[trial_idx]])
        
            all_phase=np.hstack(all_phase)
            
            _, kuiper_test = kuiper_two(phase_s, all_phase)
                        
            n_bins = 20
            tofity, tofitx = np.histogram(all_phase, bins=n_bins)
            baseline, tofitx = np.histogram(phase_s, bins=n_bins)  
            tofitx = tofitx[:-1] + (tofitx[1] - tofitx[0])/2
            tofity = tofity / baseline * float(fs)
                           
            preferred_phase,modulation_index=helper_functions.compute_phase_tuning(tofitx, tofity)
            
            n_perm = 100
            n_spk = len(all_phase)
            di_distr = np.zeros(n_perm)
            for i_perm in range(n_perm):
                tofity_p, _ = np.histogram(np.random.choice(phase_s, n_spk), bins=n_bins) 
                tofity_p = tofity_p / baseline * float(fs)
                _, di_distr[i_perm] = helper_functions.compute_phase_tuning(tofitx, tofity_p)
                
            _, di_perm = stats.mannwhitneyu(modulation_index,di_distr,alternative='greater')          
        
            units_jaw_tunings.append({**unit_key, 'modulation_index': modulation_index, 'preferred_phase': preferred_phase, 'jaw_x': tofitx, 'jaw_y': tofity, 'kuiper_test': kuiper_test, 'di_perm': di_perm})
            
        self.insert(units_jaw_tunings, ignore_extra_fields=True)
        
@schema
class BreathingTuning(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    modulation_index: float
    preferred_phase: float
    breathing_x: mediumblob
    breathing_y: mediumblob
    """
    # mtl sessions only
    key_source = experiment.Session & experiment.Breathing & ephys.Unit & 'rig = "RRig-MTL"'
    
    def make(self, key):
    
        # get traces and phase
        good_units=ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & key & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'
        
        unit_keys=good_units.fetch('KEY')
        
        traces = experiment.Breathing & key
        
        if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & key)) != len(traces):
            print(f'Mismatch in tracking trial and ephys trial number: {key}')
            return
        
        session_traces, breathing_ts = traces.fetch('breathing', 'breathing_timestamps', order_by='trial')
        fs=25000
        ds=100
        good_traces = session_traces
        for i, d in enumerate(session_traces):
            good_traces[i] = d[breathing_ts[i] < 5][::ds]
        traces_length = [len(d) for d in good_traces]
        good_trial_ind = np.where(np.array(traces_length) == 5*fs/ds)[0]
        good_traces = good_traces[good_trial_ind]
        good_traces = np.vstack(good_traces)
        
        amp, phase=behavior_plot.compute_insta_phase_amp(good_traces, float(fs/ds), freq_band=(1, 15))
        phase = phase + np.pi
        
        # compute phase and MI
        units_breathing_tunings = []
        for unit_key in unit_keys:
            all_spikes=(ephys.Unit.TrialSpikes & unit_key).fetch('spike_times', order_by='trial')
            good_spikes = np.array(all_spikes[good_trial_ind]*float(fs/ds)) # get good spikes and convert to indices
            good_spikes = [d.astype(int) for d in good_spikes] # convert to intergers
        
            for i, d in enumerate(good_spikes):
                good_spikes[i] = d[d < int(5*fs/ds)]
        
            all_phase = []
            for trial_idx in range(len(good_spikes)):
                all_phase.append(phase[trial_idx][good_spikes[trial_idx]])
        
            all_phase=np.hstack(all_phase)
            
            n_bins = 20
            tofity, tofitx = np.histogram(all_phase, bins=n_bins)
            baseline, tofitx = np.histogram(phase, bins=n_bins)  
            tofitx = tofitx[:-1] + (tofitx[1] - tofitx[0])/2
            tofity = tofity / baseline * float(fs/ds)
            
            preferred_phase,modulation_index=helper_functions.compute_phase_tuning(tofitx, tofity)             
        
            units_breathing_tunings.append({**unit_key, 'modulation_index': modulation_index, 'preferred_phase': preferred_phase, 'breathing_x': tofitx, 'breathing_y': tofity})
            
        self.insert(units_breathing_tunings, ignore_extra_fields=True)

@schema
class WhiskerTuning(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    modulation_index: float
    preferred_phase: float
    whisker_x: mediumblob
    whisker_y: mediumblob
    """
    # mtl sessions only
    key_source = experiment.Session & v_oralfacial_analysis.WhiskerSVD & ephys.Unit & 'rig = "RRig-MTL"'
    
    def make(self, key):
        num_frame = 1471
        # get traces and phase
        good_units=ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & key & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'
        
        unit_keys=good_units.fetch('KEY')
        
        traces = tracking.Tracking.JawTracking & key & {'tracking_device': 'Camera 4'}
        
        if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & key)) != len(traces):
            print(f'Mismatch in tracking trial and ephys trial number: {key}')
            return
        
        session_traces_w = (v_oralfacial_analysis.WhiskerSVD & key).fetch('mot_svd')
        
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
        
        amp, phase=behavior_plot.compute_insta_phase_amp(session_traces_w, float(fs), freq_band=(5, 20))
        phase = phase + np.pi
        
        # compute phase and MI
        units_whisker_tunings = []
        for unit_key in unit_keys:
            all_spikes=(ephys.Unit.TrialSpikes & unit_key).fetch('spike_times', order_by='trial')
            good_spikes = np.array(all_spikes*float(fs)) # get good spikes and convert to indices
            good_spikes = [d.astype(int) for d in good_spikes] # convert to intergers
        
            for i, d in enumerate(good_spikes):
                good_spikes[i] = d[d < int(5*fs)]
        
            all_phase = []
            for trial_idx in range(len(good_spikes)):
                all_phase.append(phase[trial_idx][good_spikes[trial_idx]])
        
            all_phase=np.hstack(all_phase)
            
            n_bins = 20
            tofity, tofitx = np.histogram(all_phase, bins=n_bins)
            baseline, tofitx = np.histogram(phase, bins=n_bins)  
            tofitx = tofitx[:-1] + (tofitx[1] - tofitx[0])/2
            tofity = tofity / baseline * float(fs)
            
            #print(unit_key)
            preferred_phase,modulation_index=helper_functions.compute_phase_tuning(tofitx, tofity)             
        
            units_whisker_tunings.append({**unit_key, 'modulation_index': modulation_index, 'preferred_phase': preferred_phase, 'whisker_x': tofitx, 'whisker_y': tofity})
            
        self.insert(units_whisker_tunings, ignore_extra_fields=True)

@schema
class GLMFit(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    r2: mediumblob
    r2_t: mediumblob
    weights: mediumblob
    test_y: longblob
    predict_y: longblob
    test_x: longblob
    """
    # mtl sessions only
    key_source = experiment.Session & v_tracking.TongueTracking3DBot & experiment.Breathing & v_oralfacial_analysis.WhiskerSVD & ephys.Unit & 'rig = "RRig-MTL"'
    
    def make(self, key):
        num_frame = 1471
        good_units=ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & key & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'
        unit_keys=good_units.fetch('KEY')
        bin_width = 0.017

        # from the cameras
        tongue_thr = 0.95
        traces_s = tracking.Tracking.TongueTracking & key & {'tracking_device': 'Camera 3'}
        traces_b = tracking.Tracking.TongueTracking & key & {'tracking_device': 'Camera 4'}
        
        if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & key)) != len(traces_s):
            print(f'Mismatch in tracking trial and ephys trial number: {key}')
            return
        if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & key)) != len(traces_b):
            print(f'Mismatch in tracking trial and ephys trial number: {key}')
            return
        
        session_traces_s_l_o = traces_s.fetch('tongue_likelihood', order_by='trial')
        session_traces_b_l_o = traces_b.fetch('tongue_likelihood', order_by='trial')
        trial_key=(v_tracking.TongueTracking3DBot & key).fetch('trial', order_by='trial')
        test_t = trial_key[::5] # test trials
        trial_key=np.setdiff1d(trial_key,test_t)
        
        session_traces_s_l = session_traces_s_l_o[trial_key-1]
        session_traces_b_l = session_traces_b_l_o[trial_key-1]
        session_traces_s_l = np.vstack(session_traces_s_l)
        session_traces_b_l = np.vstack(session_traces_b_l)
        session_traces_t_l = session_traces_b_l
        session_traces_t_l[np.where((session_traces_s_l > tongue_thr) & (session_traces_b_l > tongue_thr))] = 1
        session_traces_t_l[np.where((session_traces_s_l <= tongue_thr) | (session_traces_b_l <= tongue_thr))] = 0
        session_traces_t_l = np.hstack(session_traces_t_l)
        
        session_traces_s_l_t = session_traces_s_l_o[test_t-1]
        session_traces_b_l_t = session_traces_b_l_o[test_t-1]
        session_traces_s_l_t = np.vstack(session_traces_s_l_t)
        session_traces_b_l_t = np.vstack(session_traces_b_l_t)
        session_traces_t_l_t = session_traces_b_l_t
        session_traces_t_l_t[np.where((session_traces_s_l_t > tongue_thr) & (session_traces_b_l_t > tongue_thr))] = 1
        session_traces_t_l_t[np.where((session_traces_s_l_t <= tongue_thr) | (session_traces_b_l_t <= tongue_thr))] = 0
        session_traces_t_l_t = np.hstack(session_traces_t_l_t)

        # from 3D calibration
        traces_s = v_tracking.JawTracking3DSid & key
        traces_b = v_tracking.TongueTracking3DBot & key
        session_traces_s_y_o, session_traces_s_x_o, session_traces_s_z_o = traces_s.fetch('jaw_y', 'jaw_x', 'jaw_z', order_by='trial')
        session_traces_b_y_o, session_traces_b_x_o, session_traces_b_z_o = traces_b.fetch('tongue_y', 'tongue_x', 'tongue_z', order_by='trial')
        session_traces_s_y_o = stats.zscore(np.vstack(session_traces_s_y_o),axis=None)
        session_traces_s_x_o = stats.zscore(np.vstack(session_traces_s_x_o),axis=None)
        session_traces_s_z_o = stats.zscore(np.vstack(session_traces_s_z_o),axis=None)
        session_traces_b_y_o = stats.zscore(np.vstack(session_traces_b_y_o),axis=None) 
        session_traces_b_x_o = stats.zscore(np.vstack(session_traces_b_x_o),axis=None)
        session_traces_b_z_o = stats.zscore(np.vstack(session_traces_b_z_o),axis=None)
        
        session_traces_s_y = session_traces_s_y_o[trial_key-1]
        session_traces_s_x = session_traces_s_x_o[trial_key-1]
        session_traces_s_z = session_traces_s_z_o[trial_key-1]
        session_traces_b_y = session_traces_b_y_o[trial_key-1]
        session_traces_b_x = session_traces_b_x_o[trial_key-1]
        session_traces_b_z = session_traces_b_z_o[trial_key-1]
        traces_len = np.size(session_traces_b_z, axis = 1)
        num_trial = np.size(session_traces_b_z, axis = 0)

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
        
        # test trials
        session_traces_s_y_t = session_traces_s_y_o[test_t-1]
        session_traces_s_x_t = session_traces_s_x_o[test_t-1]
        session_traces_s_z_t = session_traces_s_z_o[test_t-1]
        session_traces_b_y_t = session_traces_b_y_o[test_t-1]
        session_traces_b_x_t = session_traces_b_x_o[test_t-1]
        session_traces_b_z_t = session_traces_b_z_o[test_t-1]
        traces_len_t = np.size(session_traces_b_z_t, axis = 1)
        num_trial_t = np.size(session_traces_b_z_t, axis = 0)

        session_traces_s_y_t = np.hstack(session_traces_s_y_t)
        session_traces_s_x_t = np.hstack(session_traces_s_x_t)
        session_traces_s_z_t = np.hstack(session_traces_s_z_t)
        session_traces_b_y_t = np.hstack(session_traces_b_y_t)
        session_traces_b_x_t = np.hstack(session_traces_b_x_t)
        session_traces_b_z_t = np.hstack(session_traces_b_z_t)
        # -- moving-average and down-sample
        session_traces_s_x_t = np.convolve(session_traces_s_x_t, kernel, 'same')
        session_traces_s_x_t = session_traces_s_x_t[window_size::window_size]
        session_traces_s_y_t = np.convolve(session_traces_s_y_t, kernel, 'same')
        session_traces_s_y_t = session_traces_s_y_t[window_size::window_size]
        session_traces_s_z_t = np.convolve(session_traces_s_z_t, kernel, 'same')
        session_traces_s_z_t = session_traces_s_z_t[window_size::window_size]
        session_traces_b_x_t = np.convolve(session_traces_b_x_t, kernel, 'same')
        session_traces_b_x_t = session_traces_b_x_t[window_size::window_size]
        session_traces_b_y_t = np.convolve(session_traces_b_y_t, kernel, 'same')
        session_traces_b_y_t = session_traces_b_y_t[window_size::window_size]
        session_traces_b_z_t = np.convolve(session_traces_b_z_t, kernel, 'same')
        session_traces_b_z_t = session_traces_b_z_t[window_size::window_size]
        session_traces_t_l_t = np.convolve(session_traces_t_l_t, kernel, 'same')
        session_traces_t_l_t = session_traces_t_l_t[window_size::window_size]
        session_traces_t_l_t[np.where(session_traces_t_l_t < 1)] = 0
        session_traces_s_x_t = np.reshape(session_traces_s_x_t,(-1,1))
        session_traces_s_y_t = np.reshape(session_traces_s_y_t,(-1,1))
        session_traces_s_z_t = np.reshape(session_traces_s_z_t,(-1,1))
        session_traces_b_x_t = np.reshape(session_traces_b_x_t * session_traces_t_l_t, (-1,1))
        session_traces_b_y_t = np.reshape(session_traces_b_y_t * session_traces_t_l_t, (-1,1))
        session_traces_b_z_t = np.reshape(session_traces_b_z_t * session_traces_t_l_t, (-1,1))

        # get breathing
        breathing, breathing_ts = (experiment.Breathing & key).fetch('breathing', 'breathing_timestamps', order_by='trial')
        good_breathing = breathing
        for i, d in enumerate(breathing):
            good_breathing[i] = d[breathing_ts[i] < traces_len*3.4/1000]
        good_breathing_o = stats.zscore(np.vstack(good_breathing),axis=None)
        
        good_breathing = np.hstack(good_breathing_o[trial_key-1])
        # -- moving-average
        window_size = int(bin_width/(breathing_ts[0][1]-breathing_ts[0][0]))  # sample
        kernel = np.ones(window_size) / window_size
        good_breathing = np.convolve(good_breathing, kernel, 'same')
        # -- down-sample
        good_breathing = good_breathing[window_size::window_size]
        good_breathing = np.reshape(good_breathing,(-1,1))
        
        # test trials
        good_breathing_t = np.hstack(good_breathing_o[test_t-1])
        # -- moving-average
        good_breathing_t = np.convolve(good_breathing_t, kernel, 'same')
        # -- down-sample
        good_breathing_t = good_breathing_t[window_size::window_size]
        good_breathing_t = np.reshape(good_breathing_t,(-1,1))
        
        # get whisker
        session_traces_w = (v_oralfacial_analysis.WhiskerSVD & key).fetch('mot_svd')
        if len(session_traces_w[0][:,0]) % num_frame != 0:
            print('Bad videos in bottom view')
            return
        else:
            num_trial_w = int(len(session_traces_w[0][:,0])/num_frame)
            session_traces_w = np.reshape(session_traces_w[0][:,0], (num_trial_w, num_frame))
            
        trial_idx_nat = [d.astype(str) for d in np.arange(num_trial_w)]
        trial_idx_nat = sorted(range(len(trial_idx_nat)), key=lambda k: trial_idx_nat[k])
        trial_idx_nat = sorted(range(len(trial_idx_nat)), key=lambda k: trial_idx_nat[k])
        session_traces_w = session_traces_w[trial_idx_nat,:]
        session_traces_w_o = stats.zscore(session_traces_w,axis=None)
        
        session_traces_w = session_traces_w_o[trial_key-1,:]
        session_traces_w = np.hstack(session_traces_w)
        window_size = int(bin_width/0.0034)  # sample
        kernel = np.ones(window_size) / window_size
        session_traces_w = np.convolve(session_traces_w, kernel, 'same')
        session_traces_w = session_traces_w[window_size::window_size]
        session_traces_w = np.reshape(session_traces_w,(-1,1))
        
        session_traces_w_t = session_traces_w_o[test_t-1,:]
        session_traces_w_t = np.hstack(session_traces_w_t)
        session_traces_w_t = np.convolve(session_traces_w_t, kernel, 'same')
        session_traces_w_t = session_traces_w_t[window_size::window_size]
        session_traces_w_t = np.reshape(session_traces_w_t,(-1,1))

        # stimulus
        V_design_matrix = np.concatenate((session_traces_s_x, session_traces_s_y, session_traces_s_z, session_traces_b_x, session_traces_b_y, session_traces_b_z, good_breathing, session_traces_w), axis=1)
        V_design_matrix_t = np.concatenate((session_traces_s_x_t, session_traces_s_y_t, session_traces_s_z_t, session_traces_b_x_t, session_traces_b_y_t, session_traces_b_z_t, good_breathing_t, session_traces_w_t), axis=1)
        
        #set up GLM
        sm_log_Link = sm.genmod.families.links.log

        taus = np.arange(-5,6)
        
        units_glm = []

        for unit_key in unit_keys: # loop for each neuron
            all_spikes=(ephys.Unit.TrialSpikes & unit_key).fetch('spike_times', order_by='trial')
            
            good_spikes = np.array(all_spikes[trial_key-1]) # get good spikes
            for i, d in enumerate(good_spikes):
                good_spikes[i] = d[d < traces_len*3.4/1000]+traces_len*3.4/1000*i
            good_spikes = np.hstack(good_spikes)          
            y, bin_edges = np.histogram(good_spikes, np.arange(0, traces_len*3.4/1000*num_trial, bin_width))
            
            good_spikes_t = np.array(all_spikes[test_t-1]) # get good spikes
            for i, d in enumerate(good_spikes_t):
                good_spikes_t[i] = d[d < traces_len_t*3.4/1000]+traces_len_t*3.4/1000*i
            good_spikes_t = np.hstack(good_spikes_t)
            y_t, bin_edges = np.histogram(good_spikes_t, np.arange(0, traces_len_t*3.4/1000*num_trial_t, bin_width))
            
            r2s=np.zeros(len(taus))
            r2s_t=r2s
            weights_t=np.zeros((len(taus),9))
            predict_ys=np.zeros((len(taus),len(y_t)))
            for i, tau in enumerate(taus):
                y_roll=np.roll(y,tau)
                y_roll_t=np.roll(y_t,tau)
                glm_poiss = sm.GLM(y_roll, sm.add_constant(V_design_matrix), family=sm.families.Poisson(link=sm_log_Link))
            
                try:
                    glm_result = glm_poiss.fit()                   
                    sst_val = sum(map(lambda x: np.power(x,2),y_roll-np.mean(y_roll))) 
                    sse_val = sum(map(lambda x: np.power(x,2),glm_result.resid_response))
                    r2s[i] = 1.0 - sse_val/sst_val
                    
                    y_roll_t_p=glm_result.predict(sm.add_constant(V_design_matrix_t))
                    sst_val = sum(map(lambda x: np.power(x,2),y_roll_t-np.mean(y_roll_t))) 
                    sse_val = sum(map(lambda x: np.power(x,2),y_roll_t-y_roll_t_p)) 
                    r2s_t = 1.0 - sse_val/sst_val
                    
                    weights_t[i,:] = glm_result.params
                    
                except:
                    pass
                
            units_glm.append({**unit_key, 'r2': r2s, 'r2_t': r2s_t, 'weights': weights_t, 'test_y': y_t, 'predict_y': predict_ys, 'test_x': V_design_matrix_t})
            print(unit_key)
            
        self.insert(units_glm, ignore_extra_fields=True)

@schema
class WhiskerSVD(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    mot_svd: longblob
    """
    
    key_source = experiment.Session & 'rig = "RRig-MTL"' & (tracking.Tracking  & 'tracking_device = "Camera 4"')
    
    def make(self, key):
        
        from facemap import process
        
        roi_path = 'H://videos//bottom//DL027//2021_07_01//DL027_2021_07_01_bottom_0_proc.npy'
        roi_data = np.load(roi_path, allow_pickle=True).item()
        
        video_root_dir = pathlib.Path('H:/videos')
        
        trial_path = (tracking_ingest.TrackingIngest.TrackingFile & 'tracking_device = "Camera 4"' & 'trial = 1' & key).fetch1('tracking_file')
        
        video_path = video_root_dir / trial_path
        
        video_path = video_path.parent
        
        video_files = list(video_path.glob('*.mp4'))
        video_files_l = [[video_files[0]]]
        for ind_trial, file in enumerate(video_files[1:]):
            video_files_l.append([file])
            
        proc = process.run(video_files_l, proc=roi_data)
        
        self.insert1({**key, 'mot_svd': proc['motSVD'][1][:, :3]})
        