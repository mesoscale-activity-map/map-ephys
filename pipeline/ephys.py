import datajoint as dj
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import signal
from scipy.stats import poisson

from . import lab, experiment, ccf
from . import get_schema_name

schema = dj.schema(get_schema_name('ephys'))
[lab, experiment, ccf]  # NOQA flake8

DEFAULT_ARCHIVE_STORE = {
    "protocol": "s3",
    "endpoint": "s3.amazonaws.com",
    "bucket": "map-cluster-archive",
    "location": "/cluster_archive",
    "stage": "./data/archive_stage",
    "access_key": "",
    "secret_key": ""
}


if 'stores' not in dj.config:
    dj.config['stores'] = {}

if 'archive_store' not in dj.config['stores']:
    dj.config['stores']['archive_store'] = DEFAULT_ARCHIVE_STORE


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
        depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
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
class ProbeInsertionQuality(dj.Manual):
    definition = """  # Indication of insertion quality (good/bad) - for various reasons: lack of unit, poor behavior, poor histology
    -> ProbeInsertion
    ---
    insertion_quality: enum('good', 'bad')
    comment: varchar(1000)  # comment/reason for the 'good'/'bad' label
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

    contents = zip(['jrclust_v3', 'kilosort', 'jrclust_v4', 'kilosort2'])


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
    spike_sites : longblob  # array of electrode associated with each spike
    spike_depths : longblob # (um) array of depths associated with each spike
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
class UnitNote(dj.Imported):
    definition = """
    -> Unit
    note_source: varchar(36)  # e.g. "sort", "Davesort", "Han-sort"
    ---
    note_value: varchar(128)
    """

    key_source = ProbeInsertion & Unit

    def make(self, key):
        pass


@schema
class UnitNoiseLabel(dj.Imported):
    definition = """  
    # labeling based on the noiseTemplate module 
    # (https://github.com/jenniferColonell/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/noise_templates)
    -> Unit
    ---
    noise: enum('good', 'noise')
    """

    key_source = ProbeInsertion & Unit

    def make(self, key):
        pass


@schema
class ClusteringLabel(dj.Imported):
    definition = """
    -> Unit
    ---
    clustering_time: datetime  # time of generation of this set of clustering results 
    quality_control: bool  # has this clustering results undergone quality control
    manual_curation: bool  # is manual curation performed on this clustering result
    clustering_note=null: varchar(2000)  
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

    isi_threshold = 0.002  # threshold for isi violation of 2 ms
    min_isi = 0  # threshold for duplicate spikes

    # NOTE - this key_source logic relies on ALL TrialSpikes ingest all at once in a transaction
    key_source = ProbeInsertion & Unit.TrialSpikes

    def make(self, key):
        # Following isi_violations() function
        # Ref: https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py
        def make_insert():
            for unit in (Unit & key).fetch('KEY'):
                trial_spikes, tr_start, tr_stop = (Unit.TrialSpikes * experiment.SessionTrial & unit).fetch(
                    'spike_times', 'start_time', 'stop_time')

                isis = np.hstack(np.diff(spks) for spks in trial_spikes)

                if isis.size > 0:
                    # remove duplicated spikes
                    processed_trial_spikes = []
                    for spike_train in trial_spikes:
                        duplicate_spikes = np.where(np.diff(spike_train) <= self.min_isi)[0]
                        processed_trial_spikes.append(np.delete(spike_train, duplicate_spikes + 1))

                    num_spikes = len(np.hstack(processed_trial_spikes))
                    avg_firing_rate = num_spikes / float(sum(tr_stop - tr_start))

                    num_violations = sum(isis < self.isi_threshold)
                    violation_time = 2 * num_spikes * (self.isi_threshold - self.min_isi)
                    violation_rate = num_violations / violation_time
                    fpRate = violation_rate / avg_firing_rate

                    yield {**unit, 'isi_violation': fpRate, 'avg_firing_rate': avg_firing_rate}

                else:
                    yield {**unit, 'isi_violation': None, 'avg_firing_rate': None}

        self.insert(make_insert())


@schema
class ClusterMetric(dj.Imported):
    definition = """ 
    # Quality metrics for sorted unit
    # Ref: https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/README.md
    -> Unit
    epoch_name_quality_metrics: varchar(64)
    ---
    presence_ratio: float  # Fraction of epoch in which spikes are present
    amplitude_cutoff: float  # Estimate of miss rate based on amplitude histogram
    isolation_distance=null: float  # Distance to nearest cluster in Mahalanobis space
    l_ratio=null: float  # 
    d_prime=null: float  # Classification accuracy based on LDA
    nn_hit_rate=null: float  # 
    nn_miss_rate=null: float
    silhouette_score=null: float  # Standard metric for cluster overlap
    max_drift=null: float  # Maximum change in spike depth throughout recording
    cumulative_drift=null: float  # Cumulative change in spike depth throughout recording 
    """


@schema
class WaveformMetric(dj.Imported):
    definition = """
    -> Unit
    epoch_name_waveform_metrics: varchar(64)
    ---
    duration=null: float
    halfwidth=null: float
    pt_ratio=null: float
    repolarization_slope=null: float
    recovery_slope=null: float
    spread=null: float
    velocity_above=null: float
    velocity_below=null: float   
    """


@schema
class MAPClusterMetric(dj.Computed):
    definition = """
    -> Unit
    """

    class DriftMetric(dj.Part):
        definition = """
        -> master
        ---
        drift_metric: float
        """

    key_source = Unit & UnitStat

    def make(self, key):
        # -- get trial-spikes
        trial_spikes, trial_starts, trial_stops = (
                Unit.TrialSpikes * experiment.SessionTrial & key).fetch(
            'spike_times', 'start_time', 'stop_time', order_by='trial')
        mean_spike_rate = (UnitStat & key).fetch1('avg_firing_rate')
        # -- compute trial spike-rates
        trial_durations = (trial_stops - trial_starts).astype(float)
        trial_spike_rates = [len(s) for s in trial_spikes] / trial_durations  # spikes/sec
        # -- moving-average
        window_size = 6  # sample
        kernel = np.ones(window_size) / window_size
        processed_trial_spike_rates = np.convolve(trial_spike_rates, kernel, 'same')
        # -- down-sample
        ds_factor = 6
        processed_trial_spike_rates = signal.resample(
            processed_trial_spike_rates, int(len(processed_trial_spike_rates) / ds_factor))
        # -- compute drift_qc from poisson distribution
        poisson_cdf = poisson.cdf(processed_trial_spike_rates, mean_spike_rate)
        drift_qc = np.logical_or(poisson_cdf > 0.95, poisson_cdf < 0.05).sum() / len(poisson_cdf)
        # -- insert
        self.insert1(key)
        self.DriftMetric.insert1({**key, 'drift_metric': drift_qc})


# TODO: confirm the logic/need for this table
@schema
class UnitCCF(dj.Computed):
    definition = """ 
    -> Unit
    ---
    -> ccf.CCF
    """


# ======== Archived Clustering ========

@schema
class ArchivedClustering(dj.Imported):
    definition = """
    -> ProbeInsertion
    -> ClusteringMethod
    clustering_time: datetime  # time of generation of this set of clustering results 
    ---
    archival_time: datetime  # time of archiving
    quality_control: bool  # has this clustering results undergone quality control
    manual_curation: bool  # is manual curation performed on this clustering result
    clustering_note=null: varchar(2000)  
    """

    class EphysFile(dj.Part):
        definition = """
        -> master
        ephys_file: varchar(255)    # rig file/dir subpath
        """

    class Unit(dj.Part):
        definition = """
        -> master
        unit: smallint
        ---
        -> UnitQualityType
        -> [nullable] CellType
        -> lab.ElectrodeConfig.Electrode # site on the electrode for which the unit has the largest amplitude
        unit_posx : double # (um) estimated x position of the unit relative to probe's tip (0,0)
        unit_posy : double # (um) estimated y position of the unit relative to probe's tip (0,0)
        spike_times : blob@archive_store  # (s) from the start of the first data point used in clustering
        spike_sites : blob@archive_store  # array of electrode associated with each spike
        spike_depths : blob@archive_store # (um) array of depths associated with each spike
        trial_spike : blob@archive_store  # array of trial numbering per spike - same size as spike_times
        waveform : blob@archive_store     # average spike waveform  
        """

    class UnitStat(dj.Part):
        definition = """
        -> master
        -> ArchivedClustering.Unit
        ---
        unit_amp : float
        unit_snr : float
        isi_violation=null: float     
        avg_firing_rate=null: float  
        """

    class ClusterMetric(dj.Part):
        definition = """ 
        -> master
        -> ArchivedClustering.Unit
        epoch_name_quality_metrics: varchar(64)
        ---
        presence_ratio: float  # Fraction of epoch in which spikes are present
        amplitude_cutoff: float  # Estimate of miss rate based on amplitude histogram
        isolation_distance=null: float  # Distance to nearest cluster in Mahalanobis space
        l_ratio=null: float  # 
        d_prime=null: float  # Classification accuracy based on LDA
        nn_hit_rate=null: float  # 
        nn_miss_rate=null: float
        silhouette_score=null: float  # Standard metric for cluster overlap
        max_drift=null: float  # Maximum change in spike depth throughout recording
        cumulative_drift=null: float  # Cumulative change in spike depth throughout recording 
        """

    class WaveformMetric(dj.Part):
        definition = """
        -> master
        -> ArchivedClustering.Unit
        epoch_name_waveform_metrics: varchar(64)
        ---
        duration=null: float
        halfwidth=null: float
        pt_ratio=null: float
        repolarization_slope=null: float
        recovery_slope=null: float
        spread=null: float
        velocity_above=null: float
        velocity_below=null: float   
        """
