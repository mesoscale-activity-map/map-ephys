
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
    """

    class InsertionLocation(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.BrainLocation
        ml_location=null: float # um from ref ; right is positive; based on manipulator coordinates/reconstructed track
        ap_location=null: float # um from ref; anterior is positive; based on manipulator coordinates/reconstructed track
        dv_location=null: float # um from dura; ventral is positive; based on manipulator coordinates/reconstructed track
        ml_angle=null: float # Angle between the manipulator/reconstructed track and the Medio-Lateral axis. A tilt towards the right hemishpere is positive.
        ap_angle=null: float # Angle between the manipulator/reconstructed track and the Anterior-Posterior axis. An anterior tilt is positive. 
        """

    class ElectrodeGroup(dj.Part):
        definition = """
        # grouping of electrodes to be clustered together (e.g. a neuropixel electrode config - 384/960)
        -> master
        electrode_group: int  # electrode group
        """

    class Electrode(dj.Part):
        definition = """
        -> master.ElectrodeGroup
        -> lab.Probe.Electrode
        """


@schema
class LFP(dj.Imported):
    definition = """
    -> ProbeInsertion
    ---
    lfp_sample_rate: float          # (Hz)
    lfp_time_stamps: longblob       # timestamps with respect to the start of the recording (recording_timestamp)
    lfp_mean: longblob              # mean of LFP across electrodes
    """

    class Channel(dj.Part):
        definition = """  
        -> master
        -> ProbeInsertion.Electrode
        ---
        lfp: longblob           # recorded lfp at this electrode
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
class ElectrodeCCFPosition(dj.Manual):
    definition = """
    -> ProbeInsertion
    """

    class ElectrodePosition(dj.Part):
        definition = """
        -> lab.Probe.Electrode
        -> ccf.CCF
        """

    class ElectrodePositionError(dj.Part):
        definition = """
        -> lab.Probe.Electrode
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
    -> ProbeInsertion.Electrode # site on the electrode for which the unit has the largest amplitude
    unit_posx : double # x position of the unit on the probe
    unit_posy : double # y position of the unit on the probe
    spike_times : longblob  #  (s)
    unit_amp : double
    unit_snr : double
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
