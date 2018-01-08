import datajoint as dj
import ccf, experiment
import numpy as np
import h5py

schema = dj.schema(dj.config['ephys.database'], locals())


@schema
class Probe(dj.Lookup):
    definition = """
    # Ephys probe
    probe_part_no  :  varchar(20)
    ---
    probe_description :  varchar(1023)
    """
    contents = [
        ('15131808323', 'neuropixels probe O3')
    ]


@schema
class ElectrodeGroup(dj.Manual):
    definition = """
    # Electrode
    -> experiment.Session
    electrode_group : tinyint # Electrode_group is like the probe
    ---
    -> Probe
    """
    
    class Electrode(dj.Part): # Can we force insert the electrode here? I am running into a lot of problems with the datatype cannot be inserted
        definition = """
        -> ElectrodeGroup
        electrode : smallint # sites on the electrode
        """
    def make(self, key):
        part_no = (ElectrodeGroup() & key).fetch1('probe_part_no')
        probe = (Probe() & part_no).fetch1()
        if probe['probe_description'] == 'neuropixels probe O3':
            # Fetch the Probe corresponding to this session. If Neuropixel probe in the probe_description, then 374 electrodes for 1 electrode group
            ElectrodeGroup.Electrode().insert(list(dict(key, electrode = x) for x in range (1,375)))
        
@schema
class LabeledTrack(dj.Manual):
    definition = """
    -> ElectrodeGroup
    ---
    dye_color  : varchar(12)
    """
    
    class Point(dj.Part):
        definition = """
        -> LabeledTrack
        -> ccf.CCF
        """

@schema
class Ephys(dj.Imported):
    definition = """
    -> ElectrodeGroup
    """    
    
    class Unit(dj.Part):
        definition = """
        # Sorted unit
        -> Ephys
        unit  : smallint
        ---        
        spike_times  : longblob  #  (s)
        """
        
    class TrialUnit(dj.Part):
        definition = """
        # Entries for trials a unit is in
        -> Ephys.Unit
        -> experiment.Session.Trial
        """
    
    class Spike(dj.Part):
        definition = """
        # Time stamp of each spike relative to the trial start
        -> Ephys.Unit
        spike_time : decimal(9,4)   # (s)
        ---
        -> ElectrodeGroup.Electrode
        -> experiment.Session.Trial
        """

    def make(self, key):
        self.insert1(key) # insert self first if part table
        filepath = (ElectrodeGroup() & key).fetch1('ephys_filepath')
        f = h5py.File(filepath,'r')
        ind = np.argsort(f['S_clu']['viClu'][0]) # index sorted by cluster
        cluster_ids = f['S_clu']['viClu'][0][ind] # cluster (unit) number
        spike_times = f['viTime_spk'][0][ind] # spike times
        viSite_spk = f['viSite_spk'][0][ind] # electrode site for the spike
        viT_offset_file = f['viT_offset_file'][:] # start of each trial, subtract this number for each trial
        sRateHz = f['P']['sRateHz'][0] # sampling rate
        spike_trials = np.ones(len(spike_times)) * (len(viT_offset_file) - 1) # every spike is in the last trial
        spike_times2 = np.copy(spike_times)
        for i in range(len(viT_offset_file) - 1, 0, -1): #find the trials each unit has a spike in
            spike_trials[spike_times < viT_offset_file[i]] = i-1 # Get the trial number of each spike
            spike_times2[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] = spike_times[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] - viT_offset_file[i - 1] # subtract the viT_offset_file from each trial
        spike_times2[np.where(spike_times2 >= viT_offset_file[-1])] = spike_times[np.where(spike_times2 >= viT_offset_file[-1])] - viT_offset_file[-1] # subtract the viT_offset_file from each trial
        spike_times2 = spike_times2 / sRateHz # divide the sampling rate, sRateHz
        clu_ids_diff = np.diff(cluster_ids) # where the units seperate
        clu_ids_diff = np.where(clu_ids_diff != 0)[0] + 1 # seperate the spike_times
        units = np.split(spike_times, clu_ids_diff) / sRateHz # sub arrays of spike_times
        trialunits = np.split(spike_trials, clu_ids_diff) # sub arrays of spike_trials
        unit_ids = np.arange(len(clu_ids_diff) + 1) # unit number
        trialunits1 = [] # array of unit number
        trialunits2 = [] # array of trial number
        for i in range(0,len(trialunits)):
            trialunits2 = np.append(trialunits2, np.unique(trialunits[i]))
            trialunits1 = np.append(trialunits1, np.zeros(len(np.unique(trialunits[i])))+i)
        Ephys.Unit().insert(list(dict(key, unit = x, spike_times = units[x]) for x in unit_ids)) # batch insert the units
        #experiment.Session.Trial() #TODO: fetch the trial from experiment.Session.Trial and realign
        Ephys.TrialUnit().insert(list(dict(key, unit = trialunits1[x], trial = trialunits2[x]) for x in range(0, len(trialunits2)))) # batch insert the TrialUnit (key, unit, trial)
        Ephys.Spike().insert(list(dict(key, unit = cluster_ids[x], spike_time = spike_times2[x], electrode = viSite_spk[x], trial = spike_trials[x]) for x in range(0, len(spike_times2))), skip_duplicates=True) # batch insert the Spikes (key, unit, spike_time, electrode, trial)

@schema
class ElectrodePosition(dj.Manual):
    definition = """
    -> ElectrodeGroup.Electrode
    ---
    -> ccf.CCF
    """
