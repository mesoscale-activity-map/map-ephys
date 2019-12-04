
import datajoint as dj

from . import lab, experiment, ccf
from . import get_schema_name

import numpy as np
from scipy.interpolate import CubicSpline

schema = dj.schema(get_schema_name('ephys'))
[lab, experiment, ccf]  # NOQA flake8


@schema
class ProbeInsertion(dj.Manual):
    definition = """
    -> experiment.Session
    insertion_number: int
    ---
    -> lab.Probe
    -> lab.ElectrodeConfig
    """

    class InsertionLocation(dj.Part):
        definition = """
        -> master
        ---
        -> lab.SkullReference
        ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
        ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
        depth: decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
        theta:       decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
        phi:         decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
        beta:        decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
        """

    class RecordableBrainRegion(dj.Part):
        definition = """
        -> master
        -> lab.BrainArea
        -> lab.Hemisphere
        """

    class InsertionNote(dj.Part):
        definition = """
        -> master
        ---
        insertion_note: varchar(1000)
        """

    class RecordingSystemSetup(dj.Part):
        definition = """
        -> master
        ---
        sampling_rate: int  # (Hz)
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
    # jrclust_v3 is the version Dave uses
    # jrclust_v4 is the version Susu uses

    contents = zip(['jrclust_v3', 'kilosort', 'jrclust_v4'])


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
    unit_posx : double # (um) estimated x position of the unit relative to probe's tip (0,0)
    unit_posy : double # (um) estimated y position of the unit relative to probe's tip (0,0)
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

    class TrialSpikes(dj.Part):
        definition = """
        #
        -> Unit
        -> experiment.SessionTrial
        ---
        spike_times : longblob # (s) per-trial spike times relative to go-cue
        """


@schema
class BrainAreaDepthCriteria(dj.Manual):
    definition = """
    -> ProbeInsertion
    -> lab.BrainArea
    ---
    depth_upper: float  # (um)
    depth_lower: float  # (um)
    """


@schema
class UnitCoarseBrainLocation(dj.Computed):
    definition = """
    # Estimated unit position in the brain
    -> Unit
    ---
    -> [nullable] lab.BrainArea
    -> [nullable] lab.Hemisphere
    """

    key_source = Unit & BrainAreaDepthCriteria

    def make(self, key):
        posy = (Unit & key).fetch1('unit_posy')

        # get brain location info from this ProbeInsertion
        brain_area, hemi, skull_ref = (experiment.BrainLocation & (ProbeInsertion.InsertionLocation & key)).fetch1(
            'brain_area', 'hemisphere', 'skull_reference')

        brain_area_rules = (BrainAreaDepthCriteria & key).fetch(as_dict=True, order_by='depth_upper')

        # validate rule - non-overlapping depth criteria
        if len(brain_area_rules) > 1:
            upper, lower = zip(*[(v['depth_upper'], v['depth_lower']) for v in brain_area_rules])
            if ((np.array(lower)[:-1] - np.array(upper)[1:]) >= 0).all():
                raise Exception('Overlapping depth criteria')

        coarse_brain_area = None
        for rule in brain_area_rules:
            if rule['depth_upper'] < posy <= rule['depth_lower']:
                coarse_brain_area = rule['brain_area']
                break

        if coarse_brain_area is None:
            self.insert1(key)
        else:
            coarse_brain_location = (experiment.BrainLocation & {'brain_area': coarse_brain_area,
                                                                 'hemisphere': hemi,
                                                                 'skull_reference': skull_ref}).fetch1('KEY')
            self.insert1({**key, **coarse_brain_location})


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

    @property
    def key_source(self):
        return super().key_source & 'unit_quality != "all"'

    def make(self, key):
        upsample_factor = 100

        ave_waveform, fs = (ProbeInsertion.RecordingSystemSetup * Unit & key).fetch1('waveform', 'sampling_rate')
        cs = CubicSpline(range(len(ave_waveform)), ave_waveform)
        ave_waveform = cs(np.linspace(0, len(ave_waveform) - 1, (len(ave_waveform))*upsample_factor))

        fs = fs * upsample_factor
        x_min = np.argmin(ave_waveform) / fs
        x_max = np.argmax(ave_waveform) / fs
        waveform_width = abs(x_max-x_min) * 1000  # convert to ms

        self.insert1(dict(key,
                          cell_type='FS' if waveform_width < 0.4 else 'Pyr'))


@schema
class UnitStat(dj.Computed):
    definition = """
    -> Unit
    ---
    isi_violation=null: float    # 
    avg_firing_rate=null: float  # (Hz)
    """

    isi_violation_thresh = 0.002  # violation threshold of 2 ms

    # NOTE - this key_source logic relies on ALL TrialSpikes ingest all at once in a transaction
    key_source = ProbeInsertion & Unit.TrialSpikes

    def make(self, key):
        def make_insert():
            for unit in (Unit & key).fetch('KEY'):
                trial_spikes, tr_start, tr_stop = (Unit.TrialSpikes * experiment.SessionTrial & unit).fetch(
                    'spike_times', 'start_time', 'stop_time')
                isi = np.hstack(np.diff(spks) for spks in trial_spikes)
                yield {**unit,
                       'isi_violation': sum((isi < self.isi_violation_thresh).astype(int)) / len(isi) if isi.size else None,
                       'avg_firing_rate': len(np.hstack(trial_spikes)) / sum(tr_stop - tr_start) if isi.size else None}
        self.insert(make_insert())


# TODO: confirm the logic/need for this table
@schema
class UnitCCF(dj.Computed):
    definition = """ 
    -> Unit
    ---
    -> ccf.CCF
    """
