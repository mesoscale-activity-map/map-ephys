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
    
    class Electrode(dj.Part):
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


@schema
class ElectrodePosition(dj.Manual):
    definition = """
    -> ElectrodeGroup.Electrode
    ---
    -> ccf.CCF
    """
