import datajoint as dj
import ccf, experiment

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
    electrode_group :  tinyint
    ---
    -> Probe
    ephys_filepath  : varchar(255)   #  
    """
    
    class Electrode(dj.Part):
        definition = """
        -> ElectrodeGroup
        electrode : smallint
        """
        
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
        -> Ephys.Unit
        spike_time : decimal(9,3)   # (s)
        ---
        -> ElectrodeGroup.Electrode
        -> experiment.Session.Trial
        """

    def make(self, key):
        print(key)


@schema
class ElectrodePosition(dj.Manual):
    definition = """
    -> ElectrodeGroup.Electrode
    ---
    -> ccf.CCF
    """
