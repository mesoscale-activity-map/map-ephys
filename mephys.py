import datajoint as dj


schema = dj.schema('map_ccf', locals())


@schema
class CCF(dj.Lookup):
    definition = """
    # Common Coordinate Framework
    x   :  int   # (um)
    y   :  int   # (um)
    z   :  int   # (um)
    ---
    label  : varchar(128)   # TODO
    """

@schema
class AnnotationType(dj.Lookup):
    definition = """
    annotation_type  : varchar(16)  
    """
    
@schema
class CCFAnnotation(dj.Manual):
    definition = """
    -> CCF
    -> AnnotationType
    ---
    annotation  : varchar(1200)
    """

@schema
class Animal(dj.Manual):
    definition = """
    animal  : int    # Janelia ANM ID (6 digits)
    --- 
    dob    : date
    """
        
@schema
class Session(dj.Manual):
    definition = """
    -> Animal
    session : smallint 
    ---
    session_date  : date
    """
    
    class Trial(dj.Part):
        definition = """
        -> Session
        trial   : smallint
        ---
        start_time : decimal(9,3)  # (s)
        """
        
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
    -> Session
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
        -> CCF
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
        -> Session.Trial
        """
    
    class Spike(dj.Part):
        definition = """
        -> Ephys.Unit
        spike_time : decimal(9,3)   # (s)
        ---
        -> ElectrodeGroup.Electrode
        -> Session.Trial
        """

@schema
class ElectrodePosition(dj.Manual):
    definition = """
    -> ElectrodeGroup.Electrode
    ---
    -> CCF
    """
