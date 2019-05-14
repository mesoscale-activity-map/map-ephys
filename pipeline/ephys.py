
import datajoint as dj

from . import lab
from . import experiment
from . import ccf

schema = dj.schema(dj.config.get('ephys.database', 'map_ephys'))
[lab, experiment, ccf]  # NOQA flake8


@schema
class ProbeInsertion(dj.Manual):
    definition = """
    -> reference.Probe
    -> lab.Subject
    insertion_time : datetime # When this probe was inserted
    ---
    -> lab.ActionLocation
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
    -> experiment.Session
    """

    class ElectrodePosition(dj.Part):
        definition = """
        -> lab.Probe.Channel
        -> ccf.CCF
        """

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
    unit_site : smallint # site on the electrode for which the unit has the largest amplitude
    unit_posx : double # x position of the unit on the probe
    unit_posy : double # y position of the unit on the probe
    spike_times : longblob  #  (s)
    waveform : blob # average spike waveform
    """

    class UnitTrial(dj.Part):
        definition = """
        # Entries for trials a unit is in
        -> Unit
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
    spike_times : longblob # (s) spike times for each trial, relative to go cue
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
