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
        ('123123123', 'neuropixel probe')
    ]


@schema
class ElectrodeGroup(dj.Manual):
    definition = """
    # Electrode
    -> experiment.Session
    #electrode_group :  tinyint
    electrode_group :  int # Electrode_group is like the probe
    ---
    -> Probe
    ephys_filepath  : varchar(255)   #  
    """
    
    class Electrode(dj.Part): # Can we force insert the electrode here? I am running into a lot of problems with the datatype cannot be inserted
        definition = """
        -> ElectrodeGroup
        #electrode : smallint # sites on the electrode (int for now)
        electrode : int
        """
    def make(self, key):
        print (key)
        # TODO: For now, 374 electrodes for 1 electrode group
        
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
        spike_time : decimal(9,3)   # (s)
        ---
        -> ElectrodeGroup.Electrode
        -> experiment.Session.Trial
        """

    def make(self, key):
        f = h5py.File(key['ephys_filepath'],'r')
        ind = np.argsort(f['S_clu']['viClu'][0]) # index sorted by cluster
        cluster_ids = f['S_clu']['viClu'][0][ind] # cluster (unit) number
        spike_times = f['viTime_spk'][0][ind] # spike times
        viT_offset_file = f['viT_offset_file'][:] # start of each trial, subtract this number for each trial
        sRateHz = f['P']['sRateHz'][0] # sampling rate
        #TODO: Get the position of each spike: either from mrPos_spk or cviSpk_site
        clu_ids_diff = np.diff(cluster_ids)
        clu_ids_diff = np.where(clu_ids_diff != 0)[0] + 1 #
        units = np.split(spike_times, clu_ids_diff) # sub arrays of spike_times
        #TODO: subtract the viT_offset_file from each trial, and divide the sampling rate
        unit_ids = np.arange(len(clu_ids_diff)+1) # unit number
        #TODO: insert the unit_ids with the units
        #Unit().insert((unit_ids, units)) # batch insert the units
		
        #TODO: fetch the trial from experiment.Session.Trial
        #TrialUnit().insert() # batch insert the TrialUnit
        #Spike().insert() # batch insert the Spike

@schema
class ElectrodePosition(dj.Manual):
    definition = """
    -> ElectrodeGroup.Electrode
    ---
    -> ccf.CCF
    """
