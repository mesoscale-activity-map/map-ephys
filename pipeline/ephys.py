
import datajoint as dj
import numpy as np

from . import lab
from . import experiment
from . import ccf


schema = dj.schema(dj.config['ephys.database'])


@schema
class Probe(dj.Lookup):
    definition = """
    # Ephys probe
    probe_part_no  :  varchar(20)
    ---
    probe_type : varchar(32)
    probe_comment :  varchar(4000)
    """
    contents = [
        ('15131808323', 'neuropixels probe O3', ''),
        ('H-194', 'janelia2x32', '')
    ]
 
@schema
class UnitQualityType(dj.Lookup):
    definition = """
    # Quality
    unit_quality  :  varchar(100)
    ---
    unit_quality_description :  varchar(4000)
    """
    contents = [
        ('good', 'single unit'),
        ('ok', 'probably a single unit, but could be contaminated'),
        ('multi', 'multi unit'),
        ('all', 'all units'),
        ('ok or good', 'include both ok and good unit')
    ]

@schema
class CellType(dj.Lookup):
    definition = """
    #
    cell_type  :  varchar(100)
    ---
    cell_type_description :  varchar(4000)
    """
    contents = [
        ('Pyr', 'putative pyramidal'),
        ('FS', 'fast spiking'),
        ('not classified', 'intermediate spike-width that falls between spike-width thresholds for FS or Putative pyramidal cells'),
        ('all', 'all types')
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
        part_no = (ElectrodeGroup() & key).fetch('probe_part_no')
        probe = (Probe() & {'probe_part_no': part_no[0]}).fetch1()
        if probe['probe_type'] == 'neuropixels probe O3':
            # Fetch the Probe corresponding to this session. If Neuropixel probe in the probe_description, then 374 electrodes for 1 electrode group
            ElectrodeGroup.Electrode().insert(list(dict(key, electrode = x) for x in range (1,375)))
    
    class ElectrodeGroupPosition(dj.Part):
        definition = """
        -> ElectrodeGroup
        -> ccf.CCF
        ---
        -> lab.SkullReference
        ml_location = null : decimal(8,3) # um from ref ; right is positive; based on manipulator coordinates/reconstructed track
        ap_location = null : decimal(8,3) # um from ref; anterior is positive; based on manipulator coordinates/reconstructed track
        dv_location = null : decimal(8,3) # um from dura; ventral is positive; based on manipulator coordinates/reconstructed track
        ml_angle = null    : decimal(8,3) # Angle between the manipulator/reconstructed track and the Medio-Lateral axis. A tilt towards the right hemishpere is positive.
        ap_angle = null    : decimal(8,3) # Angle between the manipulator/reconstructed track and the Anterior-Posterior axis. An anterior tilt is positive.
        """

@schema
class LabeledTrack(dj.Manual):
    definition = """
    -> ElectrodeGroup
    ---
    labeling_date : date # in case we labeled the track not during a recorded session we can specify the exact date here
    dye_color  : varchar(32)
    """
    
    class Point(dj.Part):
        definition = """
        -> LabeledTrack
        -> ccf.CCF
        """

@schema
class Unit(dj.Imported):
    definition = """
    # Sorted unit
    -> ElectrodeGroup
    unit  : smallint
    ---
    unit_uid : int # unique across sessions/animals
    -> UnitQualityType
    unit_channel = null : float # channel on the electrode for which the unit has the largest amplitude
    spike_times  : longblob  #  (s)
    waveform : blob # average spike waveform
    """

    class UnitTrial(dj.Part):
        definition = """
        # Entries for trials a unit is in
        -> Unit
        -> experiment.SessionTrial
        """
    
    class UnitSpike(dj.Part):
        definition = """
        # Time stamp of each spike relative to the trial start
        -> Unit
        spike_time : decimal(9,4)   # (s)
        ---
        -> ElectrodeGroup.Electrode
        -> experiment.SessionTrial
        """

    class UnitPosition(dj.Part):
        definition = """
        # Estimated unit position in the brain
        -> Unit
        -> ccf.CCF
        ---
        -> lab.Hemisphere
        -> lab.BrainArea
        -> lab.SkullReference
        unit_ml_location = null : decimal(8,3) # um from ref ; right is positive; based on manipulator coordinates/reconstructed track
        unit_ap_location = null : decimal(8,3) # um from ref; anterior is positive; based on manipulator coordinates/reconstructed track
        unit_dv_location = null : decimal(8,3) # um from dura; ventral is positive; based on manipulator coordinates/reconstructed track
        """

@schema
class TrialSpikes(dj.Imported):
    definition = """
    #
    -> Unit
    -> experiment.SessionTrial
    ---
    spike_times : longblob # (s) spike times for each trial, relative to the beginning of the trial
    """

@schema
class ElectrodePosition(dj.Manual):
    definition = """
    -> ElectrodeGroup.Electrode
    ---
    -> ccf.CCF
    """

@schema
class UnitComment(dj.Manual):
    definition = """
    -> Unit
    unit_comment : varchar(767)
    """

@schema
class UnitCellType(dj.Computed):
    definition = """
    -> Unit
    ---
    -> CellType
    """

@schema
class ArchivedEphysFiles(dj.Imported):
    definition = """
    #
    -> experiment.SessionTrial
    ---
    raw_doi : varchar(1000) # doi for each .bin file
    """