
import datajoint as dj

from . import lab, experiment, ccf
from . import get_schema_name

import numpy as np

schema = dj.schema(get_schema_name('ephys'))
[lab, experiment, ccf]  # NOQA flake8


@schema
class ProbeInsertion(dj.Manual):
    definition = """
    -> experiment.Session
    insertion_number: int
    ---
    -> lab.ElectrodeConfig
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
        -> lab.ElectrodeConfig.Electrode
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
        ('all', 'all units')
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
class ClusteringMethod(dj.Lookup):
    definition = """
    clustering_method: varchar(16)
    """

    contents = zip(['jrclust', 'kilosort'])


@schema
class Unit(dj.Imported):
    """
    A certain portion of the recording is used for clustering (could very well be the entire recording)
    Thus, spike-times are relative to the 1st time point in this portion
    E.g. if clustering is performed from trial 8 to trial 200, then spike-times are relative to the start of trial 8
    """
    definition = """
    # Sorted unit
    -> ProbeInsertion
    -> ClusteringMethod
    unit: smallint
    ---
    unit_uid : int # unique across sessions/animals
    -> UnitQualityType
    -> lab.ElectrodeConfig.Electrode # site on the electrode for which the unit has the largest amplitude
    unit_posx : double # (um) estimated x position of the unit relative to probe's (0,0)
    unit_posy : double # (um) estimated y position of the unit relative to probe's (0,0)
    spike_times : longblob  # (s) from the start of the first data point used in clustering
    unit_amp : double
    unit_snr : double
    waveform : blob # average spike waveform
    """

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


@schema
class UnitStat(dj.Computed):
    definition = """
    -> Unit
    ---
    isi_violation=null: float    # 
    avg_firing_rate=null: float  # (Hz)
    """

    isi_violation_thresh = 0.002  # violation threshold of 2 ms

    key_source = ProbeInsertion & experiment.SessionTrial.proj() - (experiment.SessionTrial * Unit - TrialSpikes.proj())

    def make(self, key):
        def make_insert():
            for unit in (Unit & key).fetch('KEY'):
                trial_spikes, tr_start, tr_stop = (TrialSpikes * experiment.SessionTrial & unit).fetch(
                    'spike_times', 'start_time', 'stop_time')
                isi = np.hstack(np.diff(spks) for spks in trial_spikes)
                yield {**unit,
                       'isi_violation': sum((isi < self.isi_violation_thresh).astype(int)) / len(isi) if isi.size else None,
                       'avg_firing_rate': len(np.hstack(trial_spikes)) / sum(tr_stop - tr_start) if isi.size else None}
        self.insert(make_insert())
