import numpy as np
import datajoint as dj
import pathlib
from glob import glob
from pipeline import ephys, experiment, tracking
from pipeline.ingest import tracking as tracking_ingest

from pipeline.mtl_analysis import helper_functions
from pipeline.plot import behavior_plot


schema = dj.schema('daveliu_analysis')

v_oralfacial_analysis = dj.create_virtual_module('oralfacial_analysis', 'daveliu_analysis')

@schema
class JawTuning(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    modulation_index: float
    preferred_phase: float
    """
    # mtl sessions only
    # key_source = ephys.Unit * (experiment.Session & 'rig = "RRig-MTL"') * tracking.Tracking
    # key_source = ephys.Unit * (experiment.Session & 'rig = "RRig-MTL"') * tracking.Tracking * ephys.ClusterMetric * ephys.UnitStat & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'
    key_source = experiment.Session & ephys.Unit & tracking.Tracking & 'rig = "RRig-MTL"'
    
    def make(self, key):
        
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
        
        amp, phase=behavior_plot.compute_insta_phase_amp(good_traces, float(fs), freq_band=(5, 15))
        phase = phase + np.pi
        
        # compute phase and MI
        units_jaw_tunings = []
        for unit_key in unit_keys:
            all_spikes=(ephys.Unit.TrialSpikes & unit_key).fetch('spike_times', order_by='trial')
            good_spikes = np.array(all_spikes[good_trial_ind]*float(fs)) # get good spikes and convert to indices
            good_spikes = [d.astype(int) for d in good_spikes] # convert to intergers
        
            for i, d in enumerate(good_spikes):
                good_spikes[i] = d[d < 1470]
        
            all_phase = []
            for trial_idx in range(len(good_spikes)):
                all_phase.append(phase[trial_idx][good_spikes[trial_idx]])
        
            all_phase=np.hstack(all_phase)
            
            n_bins = 20
            tofity, tofitx = np.histogram(all_phase, bins=n_bins)
            baseline, tofitx = np.histogram(phase, bins=n_bins)  
            tofitx = tofitx[:-1] + (tofitx[1] - tofitx[0])/2
            tofity = tofity / baseline * float(fs)
            
               
            preferred_phase,modulation_index=helper_functions.compute_phase_tuning(tofitx, tofity)             
        
            units_jaw_tunings.append({**unit_key, 'modulation_index': modulation_index, 'preferred_phase': preferred_phase})
            
        self.insert(units_jaw_tunings, ignore_extra_fields=True)
        
        
@schema
class BreathingTuning(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    modulation_index: float
    preferred_phase: float
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
        
            units_breathing_tunings.append({**unit_key, 'modulation_index': modulation_index, 'preferred_phase': preferred_phase})
            
        self.insert(units_breathing_tunings, ignore_extra_fields=True)

@schema
class WhiskerTuning(dj.Computed):
    definition = """
    -> ephys.Unit
    ---
    modulation_index: float
    preferred_phase: float
    """
    # mtl sessions only
    key_source = experiment.Session & v_oralfacial_analysis.WhiskerSVD & ephys.Unit & 'rig = "RRig-MTL"'
    
    def make(self, key):
    
        # get traces and phase
        good_units=ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & key & 'presence_ratio > 0.9' & 'amplitude_cutoff < 0.15' & 'avg_firing_rate > 0.2' & 'isi_violation < 10' & 'unit_amp > 150'
        
        unit_keys=good_units.fetch('KEY')
        
        traces = tracking.Tracking.JawTracking & key & {'tracking_device': 'Camera 4'}
        
        if len(experiment.SessionTrial & (ephys.Unit.TrialSpikes & key)) != len(traces):
            print(f'Mismatch in tracking trial and ephys trial number: {key}')
            return
        
        session_traces_w = (v_oralfacial_analysis.WhiskerSVD & key).fetch('mot_svd')
        
        if len(session_traces_w[0][:,0]) % 1471 != 0:
            print('Bad videos in bottom view')
            return
        else:
            num_trial_w = int(len(session_traces_w[0][:,0])/1471)
            session_traces_w = np.reshape(session_traces_w[0][:,0], (num_trial_w, 1471))
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
        
            units_whisker_tunings.append({**unit_key, 'modulation_index': modulation_index, 'preferred_phase': preferred_phase})
            
        self.insert(units_whisker_tunings, ignore_extra_fields=True)


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
        
        video_files = glob(str(video_path) + "\\*.mp4")
        video_files_l = [[video_files[0]]]
        for ind_trial, file in enumerate(video_files[1:]):
            video_files_l.append([file])
            
        proc = process.run(video_files_l, proc=roi_data)
        
        self.insert1({**key, 'mot_svd': proc['motSVD'][1][:, :3]})
        