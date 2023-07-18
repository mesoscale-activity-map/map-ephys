import datajoint as dj
import numpy as np
import pathlib
from scipy.interpolate import CubicSpline
from scipy import signal
from scipy.stats import poisson
import logging
import scipy.io as spio

from pipeline import lab, experiment, ccf
from pipeline import get_schema_name, create_schema_settings
from pipeline.ingest.utils.paths import get_sess_dir, gen_probe_insert, match_probe_to_ephys
from pipeline.ingest.utils.spike_sorter_loader import cluster_loader_map

from pipeline.experiment import get_wr_sessdatetime


schema = dj.schema(get_schema_name('ephys'), **create_schema_settings)
[lab, experiment, ccf]  # NOQA flake8

log = logging.getLogger(__name__)

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


# ---- ProbeInsertion ----


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

    @classmethod
    def generate_entries(cls, session_key):
        log.info('------ ProbeInsertion generation for: {} ------'.format(session_key))

        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session.proj(..., '-session_time')) & session_key).fetch1()
        h2o = sinfo['water_restriction_number']

        dpath, dglob, rigpath = get_sess_dir(session_key)

        if dpath is None:
            return

        try:
            clustering_files = match_probe_to_ephys(h2o, dpath, dglob)
        except FileNotFoundError as e:
            log.warning(str(e) + '. Skipping...')
            return

        for probe_no, (f, cluster_method, npx_meta) in clustering_files.items():
            insertion_key = {'subject_id': sinfo['subject_id'],
                             'session': sinfo['session'],
                             'insertion_number': probe_no}
            if cls & insertion_key:
                continue
            gen_probe_insert(sinfo, probe_no, npx_meta)


@schema
class ProbeInsertionQuality(dj.Manual):
    definition = """  # Indication of insertion quality (good/bad) - for various reasons: lack of unit, poor behavior, poor histology
    -> ProbeInsertion
    ---
    drift_presence=0: bool
    number_of_landmarks: int
    alignment_confidence=1: bool
    insertion_comment='': varchar(1000)  # any comment/reason for the 'good'/'bad' label
    """

    class GoodPeriod(dj.Part):
        definition = """
        -> master
        good_period_start: decimal(9, 4)  #  (s) relative to session beginning 
        ---
        good_period_end: decimal(9, 4)  # (s) relative to session beginning 
        """

    class GoodTrial(dj.Part):
        definition = """
        -> master
        -> experiment.SessionTrial
        """


# ---- LFP ----


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

    def make(self, key):
        from .ingest.utils.spike_sorter_loader import SpikeGLX

        log.info('------ LFP ingestion for {} ------'.format(key))

        session_key, h2o = (experiment.Session * lab.WaterRestriction & key).fetch1(
            'KEY', 'water_restriction_number')

        dpath, dglob, rigpath = get_sess_dir(session_key)

        if dpath is None:
            return

        try:
            clustering_files = match_probe_to_ephys(h2o, dpath, dglob)
        except FileNotFoundError as e:
            log.warning(str(e) + '. Skipping...')
            return

        probe_no = key['insertion_number']
        f, cluster_method, npx_meta = clustering_files[probe_no]
        spikeglx_recording = SpikeGLX(pathlib.Path(npx_meta.fname).parent)

        electrode_keys, lfp = [], []

        lfp_channel_ind = spikeglx_recording.lfmeta.recording_channels

        # Extract LFP data at specified channels and convert to uV
        lfp = spikeglx_recording.lf_timeseries[:, lfp_channel_ind]  # (sample x channel)
        lfp = (lfp * spikeglx_recording.get_channel_bit_volts('lf')[lfp_channel_ind]).T  # (channel x sample)

        self.insert1(dict(key,
                          lfp_sample_rate=spikeglx_recording.lfmeta.meta['imSampRate'],
                          lfp_time_stamps=(np.arange(lfp.shape[1])
                                           / spikeglx_recording.lfmeta.meta['imSampRate']),
                          lfp_mean=lfp.mean(axis=0)))

        electrode_query = (lab.ProbeType.Electrode
                           * lab.ElectrodeConfig.Electrode
                           * ProbeInsertion & key)
        probe_electrodes = {
            (shank, shank_col, shank_row): key
            for key, shank, shank_col, shank_row in zip(*electrode_query.fetch(
                'KEY', 'shank', 'shank_col', 'shank_row'))}

        for recorded_site in lfp_channel_ind:
            shank, shank_col, shank_row, _ = spikeglx_recording.apmeta.shankmap['data'][recorded_site]
            electrode_keys.append(probe_electrodes[(shank, shank_col, shank_row)])

        # single insert in loop to mitigate potential memory issue
        for electrode_key, lfp_trace in zip(electrode_keys, lfp):
            self.Channel.insert1({**key, **electrode_key, 'lfp': lfp_trace})


# ---- Clusters/Units/Spiketimes ----


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

    contents = zip(['jrclust_v3', 'kilosort', 'jrclust_v4', 'kilosort2', 'pykilosort2.5'])


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

    key_source = ProbeInsertion()

    def make(self, key):
        from .ingest.ephys import ingest_units

        log.info('------ Units ingestion for: {} ------'.format(key))
        session_key = (experiment.Session & key).fetch1('KEY')
        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session.proj(..., '-session_time')) & session_key).fetch1()
        h2o = sinfo['water_restriction_number']

        dpath, dglob, rigpath = get_sess_dir(session_key)

        if dpath is None:
            return

        try:
            clustering_files = match_probe_to_ephys(h2o, dpath, dglob)
        except FileNotFoundError as e:
            log.warning(str(e) + '. Skipping...')
            return

        probe_no = key['insertion_number']
        f, cluster_method, npx_meta = clustering_files[probe_no]

        data = cluster_loader_map[cluster_method](sinfo, *f)

        data['rigpath'] = rigpath
        ingest_units(key, data, npx_meta)


@schema
class TrialEvent(dj.Imported):
    """
    Trialized events extracted from NIDQ channels with (global) session-based times
    """
    definition = """
    -> experiment.BehaviorTrial
    trial_event_id: smallint
    ---
    -> experiment.TrialEventType
    trial_event_time : Decimal(10, 5)  # (s) from session start (global time)
    """    


@schema
class ActionEvent(dj.Imported):
    """
    Trialized events extracted from NIDQ channels with (global) session-based times
    """
    definition = """
    -> experiment.BehaviorTrial
    action_event_id: smallint
    ---
    -> experiment.ActionEventType
    action_event_time : Decimal(10, 5)  # (s) from session start (global time)
    """


@schema
class UnitNote(dj.Imported):
    definition = """
    -> Unit
    note_source: varchar(36)  # e.g. "sort", "Davesort", "Han-sort"
    ---
    -> UnitQualityType
    """

    key_source = ProbeInsertion & Unit.proj()

    def make(self, key):
        # import here to avoid circular imports
        from pipeline.ingest import ephys as ephys_ingest
        from pipeline.util import _get_clustering_method

        ephys_file = (ephys_ingest.EphysIngest.EphysFile.proj(
            insertion_number='probe_insertion_number') & key).fetch1('ephys_file')
        rigpaths = ephys_ingest.get_ephys_paths()
        for rigpath in rigpaths:
            rigpath = pathlib.Path(rigpath)
            if (rigpath / ephys_file).exists():
                session_ephys_dir = rigpath / ephys_file
                break
        else:
            raise FileNotFoundError(
                'Error - No ephys data directory found for {}'.format(ephys_file))

        key['clustering_method'] = _get_clustering_method(key)
        units = (Unit & key).fetch('unit')
        unit_quality_types = UnitQualityType.fetch('unit_quality')

        ks = ephys_ingest.Kilosort(session_ephys_dir)
        curated_cluster_notes = ks.extract_curated_cluster_notes()

        cluster_notes = []
        for curation_source, cluster_note in curated_cluster_notes.items():
            if curation_source == 'group':
                continue
            cluster_notes.extend([{**key,
                                   'note_source': curation_source,
                                   'unit': u, 'unit_quality': note}
                                  for u, note in zip(cluster_note['cluster_ids'],
                                                     cluster_note['cluster_notes'])
                                  if u in units and note in unit_quality_types])
        self.insert(cluster_notes)


@schema
class UnitNoiseLabel(dj.Imported):
    """
    labeling based on the noiseTemplate module - output to "cluster_group.tsv" file
    (https://github.com/jenniferColonell/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/noise_templates)
    """
    definition = """ 
    # labeling based on the noiseTemplate module - output to cluster_group.tsv file
    -> Unit
    ---
    noise: enum('good', 'noise')
    """

    key_source = ProbeInsertion & Unit.proj()

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
        return Unit & 'unit_quality != "all"'

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

    key_source = ProbeInsertion & Unit.proj()

    def make(self, key):
        from .ingest.ephys import ingest_metrics

        log.info('------ Cluster metrics ingestion for {} ------'.format(key))

        session_key = (experiment.Session & key).fetch1('KEY')
        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session.proj(..., '-session_time')) & session_key).fetch1()
        h2o = sinfo['water_restriction_number']

        dpath, dglob, rigpath = get_sess_dir(session_key)

        if dpath is None:
            return

        try:
            clustering_files = match_probe_to_ephys(h2o, dpath, dglob)
        except FileNotFoundError as e:
            log.warning(str(e) + '. Skipping...')
            return

        probe_no = key['insertion_number']
        f, cluster_method, npx_meta = clustering_files[probe_no]

        data = cluster_loader_map[cluster_method](sinfo, *f)
        ingest_metrics(key, data)


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

    key_source = ProbeInsertion & ClusterMetric.proj()


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

    key_source = Unit & UnitStat & ProbeInsertionQuality

    def make(self, key):
        # -- get trial-spikes - use only trials in ProbeInsertionQuality.GoodTrial
        #    if ProbeInsertionQuality exists but no ProbeInsertionQuality.GoodTrial,
        #    then all trials are considered good trials
        if (ProbeInsertionQuality & key) and (ProbeInsertionQuality.GoodTrial & key):
            trial_spikes_query = (
                Unit.TrialSpikes
                * (experiment.TrialEvent & 'trial_event_type = "trialend"')
                & ProbeInsertionQuality.GoodTrial
                & key)
        else:
            trial_spikes_query = (
                Unit.TrialSpikes
                * (experiment.TrialEvent & 'trial_event_type = "trialend"')
                & key)

        trial_spikes, trial_durations = trial_spikes_query.fetch(
            'spike_times', 'trial_event_time', order_by='trial')

        # -- compute trial spike-rates
        trial_spike_rates = [len(s) for s in trial_spikes] / trial_durations.astype(float)  # spikes/sec
        mean_spike_rate = np.mean(trial_spike_rates)
        # -- moving-average
        window_size = 6  # sample
        kernel = np.ones(window_size) / window_size
        processed_trial_spike_rates = np.convolve(trial_spike_rates, kernel, 'same')
        # -- down-sample
        ds_factor = 6
        processed_trial_spike_rates = processed_trial_spike_rates[::ds_factor]
        # -- compute drift_qc from poisson distribution
        poisson_cdf = poisson.cdf(processed_trial_spike_rates, mean_spike_rate)
        instability = np.logical_or(poisson_cdf > 0.95, poisson_cdf < 0.05).sum() / len(poisson_cdf)
        # -- insert
        self.insert1(key)
        self.DriftMetric.insert1({**key, 'drift_metric': instability})


#TODO: confirm the logic/need for this table
@schema
class UnitCCF(dj.Computed):
    definition = """ 
    -> Unit
    ---
    -> ccf.CCF
    """


# ======== Classification of "good/unlabelled" on a per-region basis ========
# using logistic regression - based on the white paper below
# https://janelia.figshare.com/articles/online_resource/Spike_sorting_and_quality_control_for_the_mesoscale_activity_map_project/22154810/1

@schema
class SingleUnitClassification(dj.Computed):
    definition = """
    -> experiment.Session
    """

    class UnitClassification(dj.Part):
        definition = """
        -> master
        -> Unit
        ---
        classification: enum('good', 'unlabelled')  # single unit classification label ("good"/"unlabelled")
        anno_name='': varchar(128)  # brain region annotation of the classified unit
        """

    key_source = experiment.Session & (experiment.ProjectSession
                                       & {'project_name': 'Brain-wide neural activity underlying memory-guided movement'})

    def make(self, key):
        classified_output_dir = dj.config['custom'].get('single_unit_classification_dir')
        if not classified_output_dir:
            raise FileNotFoundError('Missing specification of "single_unit_classification_dir" directory in dj.config["custom"]')

        water_res_num, sess_datetime = get_wr_sessdatetime(key)
        fname = f"{water_res_num}_{sess_datetime}_GoodUnitsIdx_14regions_cl.mat"
        matlab_filepath = pathlib.Path(classified_output_dir) / fname
        mat = spio.loadmat(matlab_filepath.as_posix(),
                           squeeze_me=True, struct_as_record=False)
        mat_idx = mat['Idx']
        mat_anno = mat['AnnoName']

        # unit Idx from the .mat file are concatenated unit indexes, not the original unit id
        # create a `unit_mapping` to map back to the original unit id
        insertion_numbers, unit_counts = (ProbeInsertion & key).aggr(
            Unit.proj(), unit_count='count(unit)').fetch('insertion_number', 'unit_count',
                                                       order_by='insertion_number')
        unit_count_cumsum = np.concatenate([[0], unit_counts.astype(int).cumsum()[:-1]])
        unit_count_cumsum = {i: c for i, c in zip(insertion_numbers, unit_count_cumsum)}

        unit_mapping = {}
        for insertion_number in insertion_numbers:
            unit_mapping = {**unit_mapping, **{unit_idx + 1 + unit_count_cumsum[insertion_number]: k
                            for unit_idx, k in enumerate((
                        Unit & key & {'insertion_number': insertion_number}).fetch(
                    'KEY', order_by='unit'))}}

        # extract units' classification labels for all regions
        entries = []
        for region_attr in dir(mat_idx):
            if region_attr.startswith('_') or not region_attr.endswith('_qc'):
                continue
            unit_ind = getattr(mat_idx, region_attr)
            unit_anno = getattr(mat_anno, region_attr)
            if isinstance(unit_ind, int):
                unit_ind = [unit_ind]
                unit_anno = [unit_anno]
            entries.extend([{**key, **unit_mapping[u_idx],
                             'classification': 'good',
                             'anno_name': u_anno}
                           for u_idx, u_anno in zip(unit_ind, unit_anno)])

        self.insert1(key)
        self.UnitClassification.insert(entries, skip_duplicates=True)
        self.UnitClassification.insert((Unit - self.UnitClassification & key).proj(
            classification='"unlabelled"', anno_name='""'))


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
        trial_spike=null: blob@archive_store  # array of trial numbering per spike - same size as spike_times
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


@schema
class UnitPassingCriteria(dj.Computed):
    """ This table computes whether a unit_key passes a selection criteria """
    definition = """
    -> Unit
    ---
    criteria_passed : bool      #true or false based on whether a unit passes the criteria
    """

    key_source = Unit & ClusterMetric & UnitStat

    def make(self, key):
        #insert value returned from check_unit_criteria into criteria_passed
        self.insert1(dict(key, criteria_passed=check_unit_criteria(key)))

        
# ---- Unit restriction criteria based on brain regions ----


brain_area_unit_restrictions = {
    'Medulla': 'unit_amp > 150 '
               'AND avg_firing_rate > 0.2 '
               'AND presence_ratio > 0.9 '
               'AND isi_violation < 10 '
               'AND amplitude_cutoff < 0.15',
    'ALM': 'unit_amp > 100 '
           'AND avg_firing_rate > 0.2 '
           'AND presence_ratio > 0.95 '
           'AND isi_violation < 0.1 '
           'AND amplitude_cutoff < 0.1',
    'Midbrain': 'unit_amp > 100 '
                'AND avg_firing_rate > 0.1 '
                'AND presence_ratio > 0.9 '
                'AND isi_violation < 1 '
                'AND amplitude_cutoff < 0.08',
    'Thalamus': 'unit_amp > 90 '
                'AND avg_firing_rate > 0.1 '
                'AND presence_ratio > 0.9 '
                'AND isi_violation < 0.05 '
                'AND amplitude_cutoff < 0.08',
    'Striatum': 'unit_amp > 70 '
                'AND avg_firing_rate > 0.1 '
                'AND presence_ratio > 0.9 '
                'AND isi_violation < 0.5 '
                'AND amplitude_cutoff < 0.1'
}


def check_unit_criteria(unit_key):
    """
    Check if the various statistics/metrics of a given unit passes
     the predefined criteria for the particular brain region
     (defined in "brain_area_unit_restrictions")
    Note: not handling the case where the unit is from a probe that penetrated through
    multiple brain areas (the first one is picked)
    """
    brain_area = (ProbeInsertion.RecordableBrainRegion & unit_key).fetch('brain_area', limit=1)[0]
    if brain_area in brain_area_unit_restrictions:
        unit_query = (Unit * ClusterMetric * UnitStat
                      & unit_key & brain_area_unit_restrictions[brain_area])
        return bool(unit_query)

    return True



 