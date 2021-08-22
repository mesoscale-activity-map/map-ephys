import pathlib
import numpy as np
import matplotlib.pyplot as plt

from pipeline.mtl_analysis import helper_functions
from pipeline import oralfacial_analysis, experiment, ephys, histology
#%%
water='DL008'
date='2021-04-11'
subject, session = helper_functions.water2subject(water, date)
#%% load data
proc_path = pathlib.Path('H://videos//bottom//DL027//2021_07_01//DL027_2021_07_01_bottom_0_proc.npy')
data = np.load(proc_path, allow_pickle=True).item()
svd = data['motSVD'][1]

#%%
a = ephys.ProbeInsertion & (experiment.Session & 'rig="RRig-MTL"') & histology.ElectrodeCCFPosition
a
#%%
wave_path = pathlib.Path('A://kilosort_datatemp//test//catgt_20210818_g0//20210818_g0_imec0//imec0_ks2//mean_waveforms.npy')
waveforms = np.load(wave_path)
#%%
wave_path_f = pathlib.Path('A://kilosort_datatemp//test//catgt_20210818_g0//20210818_g0_imec0//imec0_ks2//mean_waveforms.npy')
waveforms_f = np.load(wave_path)
#%%
t = np.arange(0,len(waveforms[3,0,:])/30,1/30)
plt.plot(t,waveforms[3,0,:]-np.mean(waveforms[3,0,:]),'b',label='no filter')
plt.plot(t,waveforms_f[3,0,:],'r', label='[300 none]')
plt.legend(loc='lower right')
plt.xlabel('ms')
plt.show()
#%%
qc_units = ephys.Unit * ephys.ClusterMetric * ephys.UnitStat & 'presence_ratio > 0.95' & 'amplitude_cutoff < 0.1' & 'avg_firing_rate > 0.2' & 'isi_violation < 0.1' & 'unit_amp > 100'
duration = (ephys.WaveformMetric & qc_units & (ephys.ProbeInsertion.RecordableBrainRegion & 'brain_area = "ALM"') & (ephys.ProbeInsertion & 'probe_type ="neuropixels 1.0 - 3B"')).fetch('duration')
plt.hist(duration, bins=100)
plt.xlabel('Peak-to-trough (ms)')
plt.ylabel('Number of units')
#%% check motSVD
key = experiment.Session & {'subject_id': '3454', 'session': 2}
session_traces_w = (oralfacial_analysis.WhiskerSVD & key).fetch('mot_svd')
plt.plot(session_traces_w[0][:2000,0])
#%%
proc_path = pathlib.Path('H://videos//bottom//DL008//2021_04_11//DL008_2021_04_11_bottom_0_proc.npy')
data = np.load(proc_path, allow_pickle=True).item()
svd = data['motSVD'][1]
#%%
plt.plot(svd[500:1100,0])