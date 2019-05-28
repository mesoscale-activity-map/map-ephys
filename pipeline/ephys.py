
import datajoint as dj

from . import lab
from . import experiment
from . import ccf

schema = dj.schema(dj.config.get('ephys.database', 'map_ephys'))
[lab, experiment, ccf]  # NOQA flake8


@schema
class ProbeInsertion(dj.Manual):
    definition = """
    -> experiment.Session
    insertion_number: int
    ---
    -> lab.Probe
    -> experiment.BrainLocation
    insertion_time : datetime # When this probe was inserted
    ml_location=null: float # um from ref ; right is positive; based on manipulator coordinates/reconstructed track
    ap_location=null: float # um from ref; anterior is positive; based on manipulator coordinates/reconstructed track
    dv_location=null: float # um from dura; ventral is positive; based on manipulator coordinates/reconstructed track
    ml_angle=null: float # Angle between the manipulator/reconstructed track and the Medio-Lateral axis. A tilt towards the right hemishpere is positive.
    ap_angle=null: float # Angle between the manipulator/reconstructed track and the Anterior-Posterior axis. An anterior tilt is positive.
    """


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
class ChannelCCFPosition(dj.Manual):
    definition = """
    -> ProbeInsertion
    """

    class ElectrodePosition(dj.Part):
        definition = """
        -> lab.Probe.Channel
        -> ccf.CCF
        """
    # TODO: not clear the x, y, z below is in what coordinate? CCF as well?
    class ElectrodePositionError(dj.Part):
        definition = """
        -> lab.Probe.Channel
        -> ccf.CCFLabel
        x   :  float   # (um)
        y   :  float   # (um)
        z   :  float   # (um)
        """


@schema
class LabeledTrack(dj.Manual):
    definition = """
    -> ProbeInsertion
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
    -> ProbeInsertion
    unit  : smallint
    ---
    unit_uid : int # unique across sessions/animals
    -> UnitQualityType
    -> lab.Probe.Channel # site on the electrode for which the unit has the largest amplitude
    unit_posx : double # x position of the unit on the probe
    unit_posy : double # y position of the unit on the probe
    spike_times : longblob  #  (s) with respect to the start of the session 
    waveform : blob # average spike waveform
    """

    # TODO: not sure what's the purpose of this UnitTrial here
    class UnitTrial(dj.Part):
        definition = """
        # Entries for trials a unit is in
        -> master
        -> experiment.SessionTrial
        """

    class UnitPosition(dj.Part):
        definition = """
        # Estimated unit position in the brain
        -> master
        -> ccf.CCF
        ---
        -> experiment.BrainLocation
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
class TrialSpikes(dj.Computed):
    definition = """
    #
    -> Unit
    -> experiment.SessionTrial
    ---
    spike_times : longblob # (s) spike times for each trial, relative to go cue
    """
