import logging
import pathlib
from datetime import datetime
from os import path

from tqdm import tqdm
import re
import pandas as pd
import scipy.io as spio
import h5py
import numpy as np
import datajoint as dj

from ... import get_schema_name
from .. import BitCodeError
from . import readSGLX

log = logging.getLogger(__name__)


experiment = dj.create_virtual_module('experiment', get_schema_name('experiment'))

npx_bit_volts = {'neuropixels 1.0': 2.34375, 'neuropixels 2.0': 0.763}  # uV per bit scaling factor for neuropixels probes


# ======== Loaders for clustering results ========
def _decode_notes(fh, notes):
    '''
    dereference and decode unit notes, translate to local labels
    '''
    note_map = {'single': 'good', 'ok': 'ok', 'multi': 'multi',
                '\x00\x00': 'all'}  # 'all' is default / null label
    decoded_notes = []
    for n in notes:
        note_val = str().join(chr(int(c)) for c in fh[n])
        match = [k for k in note_map if re.match(k, note_val)]
        decoded_notes.append(note_map[match[0]] if len(match) > 0 else 'all')

    return decoded_notes


def _load_jrclust_v3(sinfo, fpath):
    '''
    Ephys data loader for JRClust v4 files.

    Arguments:

      - sinfo: lab.WaterRestriction * lab.Subject * experiment.Session
      - rigpath: rig path root
      - dpath: expanded rig data path (rigpath/h2o/YYYY-MM-DD)
      - fpath: file path under dpath

    Returns:
      - tbd
    '''

    h2o = sinfo['water_restriction_number']
    skey = {k: v for k, v in sinfo.items()
            if k in experiment.Session.primary_key}

    ef_path = fpath

    log.info('.. jrclust v3 data load:')
    log.info('.... sinfo: {}'.format(sinfo))
    log.info('.... probe: {}'.format(fpath.parent.name))

    log.info('.... loading ef_path: {}'.format(str(ef_path)))
    ef = h5py.File(str(ef_path), mode='r')  # ephys file

    # -- trial-info from bitcode --
    try:
        sync_behav, sync_ephys, trial_fix, trial_go, trial_start, bitcode_raw = read_bitcode(fpath.parent, h2o, skey)
    except FileNotFoundError as e:
        raise e

    # extract unit data

    hz = ef['P']['sRateHz'][0][0]                   # sampling rate

    spikes = ef['viTime_spk'][0]                    # spike times
    spike_sites = ef['viSite_spk'][0]               # spike electrode
    spike_depths = ef['mrPos_spk'][1]               # spike depths

    units = ef['S_clu']['viClu'][0]                 # spike:unit id
    unit_wav = ef['S_clu']['trWav_raw_clu']         # waveform (unit x channel x sample)

    unit_notes = ef['S_clu']['csNote_clu'][0]       # curation notes
    unit_notes = _decode_notes(ef, unit_notes)

    unit_xpos = ef['S_clu']['vrPosX_clu'][0]        # x position
    unit_ypos = ef['S_clu']['vrPosY_clu'][0]        # y position

    unit_amp = ef['S_clu']['vrVpp_uv_clu'][0]       # amplitude
    unit_snr = ef['S_clu']['vrSnr_clu'][0]          # signal to noise

    vmax_unit_site = ef['S_clu']['viSite_clu']      # max amplitude site
    vmax_unit_site = np.array(vmax_unit_site[:].flatten(), dtype=np.int64)

    creation_time, clustering_label = extract_clustering_info(fpath.parent, 'jrclust_v3')

    metrics = None

    data = {
        'sinfo': sinfo,
        'ef_path': ef_path,
        'skey': skey,
        'method': 'jrclust_v3',
        'hz': hz,
        'spikes': spikes,
        'spike_sites': spike_sites,
        'spike_depths': spike_depths,
        'units': units,
        'unit_wav': unit_wav,
        'unit_notes': unit_notes,
        'unit_xpos': unit_xpos,
        'unit_ypos': unit_ypos,
        'unit_amp': unit_amp,
        'unit_snr': unit_snr,
        'vmax_unit_site': vmax_unit_site,
        'trial_start': trial_start,
        'trial_go': trial_go,
        'sync_ephys': sync_ephys,
        'sync_behav': sync_behav,
        'trial_fix': trial_fix,
        'metrics': metrics,
        'creation_time': creation_time,
        'clustering_label': clustering_label,
        'bitcode_raw': bitcode_raw
    }

    return data


def _load_jrclust_v4(sinfo, fpath):
    '''
    Ephys data loader for JRClust v4 files.
    Arguments:
      - sinfo: lab.WaterRestriction * lab.Subject * experiment.Session
      - rigpath: rig path root
      - dpath: expanded rig data path (rigpath/h2o/YYYY-MM-DD)
      - fpath: file path under dpath
    '''

    h2o = sinfo['water_restriction_number']
    skey = {k: v for k, v in sinfo.items()
            if k in experiment.Session.primary_key}

    ef_path = fpath

    log.info('.. jrclust v4 data load:')
    log.info('.... sinfo: {}'.format(sinfo))
    log.info('.... probe: {}'.format(fpath.parent.name))

    log.info('.... loading ef_path: {}'.format(str(ef_path)))
    ef = h5py.File(str(ef_path), mode='r')  # ephys file

    # -- trial-info from bitcode --
    try:
        sync_behav, sync_ephys, trial_fix, trial_go, trial_start, bitcode_raw = read_bitcode(fpath.parent, h2o, skey)
    except FileNotFoundError as e:
        raise e

    # extract unit data
    hz = None                                       # sampling rate  (N/A from jrclustv4, use from npx_meta)

    spikes = ef['spikeTimes'][0]                    # spikes times
    spike_sites = ef['spikeSites'][0]               # spike electrode
    spike_depths = ef['spikePositions'][0]           # spike depths

    units = ef['spikeClusters'][0]                  # spike:unit id
    unit_wav = ef['meanWfLocalRaw']                 # waveform

    unit_notes = ef['clusterNotes']                 # curation notes
    unit_notes = _decode_notes(ef, unit_notes[:].flatten())

    unit_xpos = ef['clusterCentroids'][0]           # x position
    unit_ypos = ef['clusterCentroids'][1]           # y position

    unit_amp = ef['unitVppRaw'][0]                  # amplitude
    unit_snr = ef['unitSNR'][0]                     # signal to noise

    vmax_unit_site = ef['clusterSites']             # max amplitude site
    vmax_unit_site = np.array(vmax_unit_site[:].flatten(), dtype=np.int64)

    creation_time, clustering_label = extract_clustering_info(fpath.parent, 'jrclust_v4')

    metrics = None

    data = {
        'sinfo': sinfo,
        'ef_path': ef_path,
        'skey': skey,
        'method': 'jrclust_v4',
        'hz': hz,
        'spikes': spikes,
        'spike_sites': spike_sites,
        'spike_depths': spike_depths,
        'units': units,
        'unit_wav': unit_wav,
        'unit_notes': unit_notes,
        'unit_xpos': unit_xpos,
        'unit_ypos': unit_ypos,
        'unit_amp': unit_amp,
        'unit_snr': unit_snr,
        'vmax_unit_site': vmax_unit_site,
        'trial_start': trial_start,
        'trial_go': trial_go,
        'sync_ephys': sync_ephys,
        'sync_behav': sync_behav,
        'trial_fix': trial_fix,
        'metrics': metrics,
        'creation_time': creation_time,
        'clustering_label': clustering_label,
        'bitcode_raw': bitcode_raw
    }

    return data


def _load_kilosort2(sinfo, ks_dir, npx_dir):

    h2o = sinfo['water_restriction_number']
    skey = {k: v for k, v in sinfo.items()
            if k in experiment.Session.primary_key}

    log.info('.. kilosort v2 data load:')
    log.info('.... sinfo: {}'.format(sinfo))

    # -- trial-info from bitcode --
    try:
        sync_behav, sync_ephys, trial_fix, trial_go, trial_start, bitcode_raw = read_bitcode(npx_dir, h2o, skey)
    except FileNotFoundError as e:
        raise e

    # ---- Read Kilosort results ----
    log.info('.... loading kilosort - ks_dir: {}'.format(str(ks_dir)))
    ks = Kilosort(ks_dir)

    # -- Spike-times --
    # spike_times_sec_adj > spike_times_sec > spike_times
    spk_time_key = ('spike_times_sec_adj' if 'spike_times_sec_adj' in ks.data
                    else 'spike_times_sec' if 'spike_times_sec' in ks.data else 'spike_times')
    spike_times = ks.data[spk_time_key]

    # ---- Spike-level results ----
    # -- spike_sites and spike_depths
    ks.extract_spike_depths()

    # ---- Unit-level results ----
    # -- Remove 0-spike units
    withspike_idx = [i for i, u in enumerate(ks.data['cluster_ids']) if (ks.data['spike_clusters'] == u).any()]

    valid_units = ks.data['cluster_ids'][withspike_idx]
    valid_unit_labels = ks.data['cluster_groups'][withspike_idx]
    valid_unit_labels = np.where(valid_unit_labels == 'mua', 'multi', valid_unit_labels)  # rename 'mua' to 'multi'

    # -- vmax_unit_site (peak channel), x, y, amp, waveforms, SNR --
    metric_fp = ks_dir / 'metrics.csv'

    metrics = None
    if metric_fp.exists():
        log.info('.... reading metrics.csv - data dir: {}'.format(str(metric_fp)))
        metrics = pd.read_csv(metric_fp)
        metrics.set_index('cluster_id', inplace=True)
        metrics = metrics[metrics.index == valid_units]
        # -- peak_chn, amp, snr --
        vmax_unit_site = metrics.peak_channel.values  # peak channel
        unit_amp = metrics.amplitude.values  # amp
        unit_snr = metrics.snr.values  # snr
        unit_snr = np.where(np.logical_or(np.isinf(unit_snr), np.isnan(unit_snr)), 0, unit_snr)  # set value to 0 if INF or NaN
        # -- unit x, y --
        vmax_unit_site_idx = [np.where(ks.data['channel_map'] == peak_site)[0][0] for peak_site in vmax_unit_site]
        unit_xpos = [ks.data['channel_positions'][site_idx, 0] for site_idx in vmax_unit_site_idx]
        unit_ypos = [ks.data['channel_positions'][site_idx, 1] for site_idx in vmax_unit_site_idx]
        # -- unit waveforms --
        unit_wav = np.load(ks_dir / 'mean_waveforms.npy')  # all unit x all channel x sample
        # extract unit wavform for valid units and recording channels
        unit_wav = unit_wav[np.ix_(metrics.index, ks.data['channel_map'], range(unit_wav.shape[-1]))]  # unit x channel x sample
    else:
        vmax_unit_site, unit_xpos, unit_ypos, unit_amp = [], [], [], []
        for unit in valid_units:
            template_idx = ks.data['spike_templates'][np.where(ks.data['spike_clusters'] == unit)[0][0]]
            chn_templates = ks.data['templates'][template_idx, :, :]
            site_idx = np.abs(np.abs(chn_templates).max(axis=0)).argmax()
            vmax_unit_site.append(ks.data['channel_map'][site_idx])
            # unit x, y
            unit_xpos.append(ks.data['channel_positions'][site_idx, 0])
            unit_ypos.append(ks.data['channel_positions'][site_idx, 1])
            # unit amp
            amps = ks.data['amplitudes'][ks.data['spike_clusters'] == unit]
            scaled_templates = np.matmul(chn_templates, ks.data['whitening_mat_inv'])
            best_chn_wf = scaled_templates[:, site_idx] * amps.mean()
            unit_amp.append(best_chn_wf.max() - best_chn_wf.min())

        # waveforms and SNR
        log.info('.... extracting waveforms - data dir: {}'.format(str(ks_dir)))
        dj.conn().ping()

        unit_wfs = extract_ks_waveforms(npx_dir, ks, wf_win=[-int(ks.data['templates'].shape[1]/2),
                                                             int(ks.data['templates'].shape[1]/2)])
        unit_wav = np.dstack([unit_wfs[u]['mean_wf']
                              for u in valid_units]).transpose((2, 1, 0))  # unit x channel x sample
        unit_snr = [unit_wfs[u]['snr'][np.where(ks.data['channel_map'] == u_site)[0][0]]
                    for u, u_site in zip(valid_units, vmax_unit_site)]

    # -- Ensuring `spike_times`, `trial_start` and `trial_go` are in `sample` and not `second` --
    # There is still a risk of times in `second` but with 0 decimal values and thus would be detected as `sample` (very very unlikely)
    hz = ks.data['params']['sample_rate']
    if np.mean(spike_times - np.round(spike_times)) != 0:
        log.debug('Kilosort2 spike times in seconds - converting to sample')
        spike_times = np.round(spike_times * hz).astype(int)
    if np.mean(trial_go - np.round(trial_go)) != 0:
        log.debug('Kilosort2 bitcode sTrig in seconds - converting to sample')
        trial_go = np.round(trial_go * hz).astype(int)
    if np.mean(trial_start - np.round(trial_start)) != 0:
        log.debug('Kilosort2 bitcode goCue in seconds - converting to sample')
        trial_start = np.round(trial_start * hz).astype(int)

    creation_time, clustering_label = extract_clustering_info(ks_dir, 'kilosort2')

    cluster_noise_label = ks.extract_cluster_noise_label()

    data = {
        'sinfo': sinfo,
        'ef_path': ks_dir,
        'skey': skey,
        'method': 'kilosort2',
        'hz': hz,
        'spikes': spike_times.astype(int),
        'spike_sites': ks.data['spike_sites'] + 1,  # channel numbering in this pipeline is 1-based indexed
        'spike_depths': ks.data['spike_depths'],
        'units': ks.data['spike_clusters'],
        'unit_wav': unit_wav,
        'unit_notes': valid_unit_labels,
        'unit_xpos': np.array(unit_xpos),
        'unit_ypos': np.array(unit_ypos),
        'unit_amp': np.array(unit_amp),
        'unit_snr': np.array(unit_snr),
        'vmax_unit_site': np.array(vmax_unit_site) + 1,  # channel numbering in this pipeline is 1-based indexed
        'trial_start': trial_start,
        'trial_go': trial_go,
        'sync_ephys': sync_ephys,
        'sync_behav': sync_behav,
        'trial_fix': trial_fix,
        'metrics': metrics,
        'creation_time': creation_time,
        'clustering_label': clustering_label,
        'ks_channel_map': ks.data['channel_map'] + 1,  # channel numbering in this pipeline is 1-based indexed
        'cluster_noise_label': cluster_noise_label,
        'bitcode_raw': bitcode_raw
    }

    return data


cluster_loader_map = {'jrclust_v3': _load_jrclust_v3,
                      'jrclust_v4': _load_jrclust_v4,
                      'kilosort2': _load_kilosort2}


bitcode_rig_mapper = {'RRig-MTL': {'trial_start': '*.XA_0_0.txt', 'bitcode': '*.XA_1_2.txt'}}


def read_bitcode(bitcode_dir, h2o, skey):
    """
    Load bitcode file from specified dir - example bitcode format: e.g. 'SC022_030319_Imec3_bitcode.mat'
    :return: sync_behav, sync_ephys, trial_fix, trial_go, trial_start
    """
    rig = (experiment.Session & skey).fetch1('rig')
    bitcode_dir = pathlib.Path(bitcode_dir)
    try:
        bf_path = next(bitcode_dir.glob('{}_*bitcode.mat'.format(h2o)))
        bitcode_format = 'bitcode.mat'
        log.info('.... loading bitcode file: {}'.format(str(bf_path)))
    except StopIteration:
        try:
            next(bitcode_dir.parent.glob('*.XA_0_0*.txt'))
            bitcode_format = 'nidq.XA.txt'
            log.info('.... loading bitcodes from "nidq.XA.txt" files')
        except StopIteration:
            raise FileNotFoundError(
                'Reading bitcode failed. Neither (*bitcode.mat) nor (nidq.XA*.txt)'
                ' bitcode file for {} found in {}'.format(h2o, bitcode_dir.parent))

    behav_trials, behavior_bitcodes = (experiment.TrialNote
                                       & {**skey, 'trial_note_type': 'bitcode'}).fetch(
        'trial', 'trial_note', order_by='trial')

    if bitcode_format == 'bitcode.mat':
        bf = spio.loadmat(str(bf_path))

        ephys_trial_start_times = bf['sTrig'].flatten()  # trial start
        ephys_trial_ref_times = bf['goCue'].flatten()  # trial go cues

        # check if there are `FreeWater` trials (i.e. no trial_go), if so, set those with trial_go value of NaN
        if len(ephys_trial_ref_times) < len(ephys_trial_start_times):

            if len(experiment.BehaviorTrial & skey) != len(ephys_trial_start_times):
                raise BitCodeError('Mismatch sTrig ({} elements) and total behavior trials ({} trials)'.format(
                    len(ephys_trial_start_times), len(experiment.BehaviorTrial & skey)))

            if len(experiment.BehaviorTrial & skey & 'free_water = 0') != len(ephys_trial_ref_times):
                raise BitCodeError('Mismatch goCue ({} elements) and non-FreeWater trials ({} trials)'.format(
                    len(ephys_trial_ref_times), len(experiment.BehaviorTrial & skey & 'free_water = 0')))

            all_tr = (experiment.BehaviorTrial & skey).fetch('trial', order_by='trial')
            no_free_water_tr = (experiment.BehaviorTrial & skey & 'free_water = 0').fetch('trial', order_by='trial')
            is_go_trial = np.in1d(all_tr, no_free_water_tr)

            trial_go_full = np.full_like(ephys_trial_start_times, np.nan)
            trial_go_full[is_go_trial] = ephys_trial_ref_times
            ephys_trial_ref_times = trial_go_full

        ephys_bitcodes = bf['bitCodeS']  # ephys sync codes
        trial_numbers = bf['trialNum'].flatten() if 'trialNum' in bf else None
    elif bitcode_format == 'nidq.XA.txt':
        bf = {}
        bitcodes, trial_start_times = build_bitcode(bitcode_dir.parent)
        if (experiment.BehaviorTrial & 'task = "multi-target-licking"' & skey):
            # multi-target-licking task: spiketimes w.r.t trial-start
            go_times = np.zeros_like(behav_trials)
        else:
            # delay-response or foraging task: spiketimes w.r.t go-cue
            go_times = (experiment.TrialEvent & skey & 'trial_event_type = "go"').fetch(
                'trial_event_time', order_by='trial').astype(float)
            assert len(go_times) == len(behav_trials)

        if bitcodes is None:
            # no bitcodes (the nidq.XA.txt file for bitcode is not available)
            if rig == 'RRig-MTL' and len(trial_start_times) == len(behav_trials):
                # for MTL rig, this is glitch in the recording system
                # if the number of trial matches with behavior, then this is a 1-to-1 mapping
                log.info('.... Unable to read bitcode! Recognize RRig-MTL session with matching number of behavior and ephys trials, using one-to-one trial mapping')
                ephys_bitcodes = behavior_bitcodes
                trial_numbers = behav_trials
                ephys_trial_ref_times = trial_start_times + go_times
                ephys_trial_start_times = trial_start_times
            else:
                raise FileNotFoundError('Generate bitcode failed. No bitcode file'
                                        ' "*.XA_0_0.txt"'
                                        ' found in {}'.format(bitcode_dir))
        else:
            ephys_bitcodes, trial_numbers, ephys_trial_start_times, ephys_trial_ref_times = [], [], [], []
            for bitcode, start_time in zip(bitcodes, trial_start_times):
                matched_trial_idx = np.where(behavior_bitcodes == bitcode)[0]
                if len(matched_trial_idx):
                    matched_trial_idx = matched_trial_idx[0]
                    ephys_trial_start_times.append(start_time)
                    ephys_bitcodes.append(bitcode)
                    trial_numbers.append(behav_trials[matched_trial_idx])
                    ephys_trial_ref_times.append(start_time + go_times[matched_trial_idx])

            ephys_trial_ref_times = np.array(ephys_trial_ref_times)
            trial_numbers = np.array(trial_numbers)
            ephys_trial_start_times = np.array(ephys_trial_start_times)

    else:
        raise ValueError('Unknown bitcode format: {}'.format(bitcode_format))

    return behavior_bitcodes, ephys_bitcodes, trial_numbers, ephys_trial_ref_times, ephys_trial_start_times, bf


def build_bitcode(bitcode_dir):
    time_to_first_high_bit = 0.05
    inter_bit_interval = 0.007

    bitcode_dir = pathlib.Path(bitcode_dir)

    # trial-start file
    try:
        trial_start_filepath = next(bitcode_dir.glob('*.XA_0_0.adj.txt'))
    except StopIteration:
        try:
            trial_start_filepath = next(bitcode_dir.glob('*.XA_0_0.txt'))
        except StopIteration:
            raise FileNotFoundError('Generate bitcode failed. No trial-start file'
                                    ' "*.XA_1_2.txt"'
                                    ' found in {}'.format(bitcode_dir))
    with open(trial_start_filepath, 'r') as f:
        trial_starts = f.read()
        trial_starts = trial_starts.strip().split('\n')
        trial_starts = np.array(trial_starts).astype(float)

    # bitcode file
    try:
        bitcode_filepath = next(bitcode_dir.glob('*.XA_1_2.adj.txt'))
    except StopIteration:
        try:
            bitcode_filepath = next(bitcode_dir.glob('*.XA_1_2.txt'))
        except StopIteration:
            log.info('Generate bitcode failed. No bitcode file "*.XA_1_2.txt"'
                     ' found in {}'.format(bitcode_dir))
            bitcode_filepath = None

    if bitcode_filepath is None or bitcode_filepath.stat().st_size == 0:
        log.info('Generate bitcode failed. 0kb bitcode file "*.XA_1_2.txt"'
                 ' found in {}'.format(bitcode_dir))
        bitcodes = None
    else:
        with open(bitcode_filepath, 'r') as f:
            bitcode_times = f.read()
            bitcode_times = bitcode_times.strip().split('\n')
            bitcode_times = np.array(bitcode_times).astype(float)

        trial_ends = np.concatenate([trial_starts[1:], [trial_starts[-1] + 99]])  # the last trial to be arbitrarily long
        bitcodes = []
        for trial_start, trial_end in zip(trial_starts, trial_ends):
            trial_bitcode_times = bitcode_times[np.logical_and(bitcode_times >= trial_start,
                                                               bitcode_times < trial_end)]
            trial_bitcode_times = np.concatenate(
                [[trial_start + time_to_first_high_bit - inter_bit_interval], trial_bitcode_times])
            high_bit_ind = np.cumsum(np.round(np.diff(trial_bitcode_times) / inter_bit_interval)).astype(int) - 1
            bitcode = np.zeros(10).astype(int)
            bitcode[high_bit_ind] = 1
            bitcode = ''.join(bitcode.astype(str))
            bitcodes.append(bitcode)

    return bitcodes, trial_starts


def handle_string(value):
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
    return value


# --------- SpikeGLX loaders --------------

AP_GAIN = 80  # For NP 2.0 probes; APGain = 80 for all AP (LF is computed from AP)

# Imax values for different probe types - see metaguides (http://billkarsh.github.io/SpikeGLX/#metadata-guides)
IMAX = {'neuropixels 1.0 - 3A': 512,
        'neuropixels 1.0 - 3B': 512,
        'neuropixels 2.0 - SS': 8192,
        'neuropixels 2.0 - MS': 8192}


class SpikeGLXMeta:

    def __init__(self, meta_filepath):
        # a good processing reference: https://github.com/jenniferColonell/Neuropixels_evaluation_tools/blob/master/SGLXMetaToCoords.m

        self.fname = meta_filepath
        self.meta = _read_meta(meta_filepath)

        # Infer npx probe model (e.g. 1.0 (3A, 3B) or 2.0)
        probe_model = self.meta.get('imDatPrb_type', 1)
        if probe_model <= 1:
            if 'typeEnabled' in self.meta:
                self.probe_model = 'neuropixels 1.0 - 3A'
            elif 'typeImEnabled' in self.meta:
                self.probe_model = 'neuropixels 1.0 - 3B'
        elif probe_model == 21:
            self.probe_model = 'neuropixels 2.0 - SS'
        elif probe_model == 24:
            self.probe_model = 'neuropixels 2.0 - MS'
        else:
            self.probe_model = str(probe_model)

        # Get recording time
        self.recording_time = datetime.strptime(self.meta.get('fileCreateTime_original', self.meta['fileCreateTime']),
                                                '%Y-%m-%dT%H:%M:%S')

        # Get probe serial number - 'imProbeSN' for 3A and 'imDatPrb_sn' for 3B
        try:
            self.probe_SN = self.meta.get('imProbeSN', self.meta.get('imDatPrb_sn'))
        except KeyError:
            raise KeyError('Probe Serial Number not found in either "imProbeSN" or "imDatPrb_sn"')

        self.chanmap = self._parse_chanmap(self.meta['~snsChanMap']) if '~snsChanMap' in self.meta else None
        self.shankmap = self._parse_shankmap(self.meta['~snsShankMap']) if '~snsShankMap' in self.meta else None
        self.imroTbl = self._parse_imrotbl(self.meta['~imroTbl']) if '~imroTbl' in self.meta else None

    @staticmethod
    def _parse_chanmap(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#channel-map
        Parse channel map header structure. Converts:

            '(x,y,z)(c0,x:y)...(cI,x:y),(sy0;x:y)'

        e.g:

            '(384,384,1)(AP0;0:0)...(AP383;383:383)(SY0;768:768)'

        into dict of form:

            {'shape': [x,y,z], 'c0': [x,y], ... }
        '''

        res = {}
        for u in (i.rstrip(')').split(';') for i in raw.split('(') if i != ''):
            if (len(u)) == 1:
                res['shape'] = u[0].split(',')
            else:
                res[u[0]] = u[1].split(':')

        return res

    @staticmethod
    def _parse_shankmap(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#shank-map
        Parse shank map header structure. Converts:

            '(x,y,z)(a:b:c:d)...(a:b:c:d)'

        e.g:

            '(1,2,480)(0:0:192:1)...(0:1:191:1)'

        into dict of form:

            {'shape': [x,y,z], 'data': [[a,b,c,d],...]}

        '''
        res = {'shape': None, 'data': []}

        for u in (i.rstrip(')') for i in raw.split('(') if i != ''):
            if ',' in u:
                res['shape'] = [int(d) for d in u.split(',')]
            else:
                res['data'].append([int(d) for d in u.split(':')])

        return res

    @staticmethod
    def _parse_imrotbl(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#imro-per-channel-settings
        Parse imro tbl structure. Converts:

            '(X,Y,Z)(A B C D E)...(A B C D E)'

        e.g.:

            '(641251209,3,384)(0 1 0 500 250)...(383 0 0 500 250)'

        into dict of form:

            {'shape': (x,y,z), 'data': []}
        '''
        res = {'shape': None, 'data': []}

        for u in (i.rstrip(')') for i in raw.split('(') if i != ''):
            if ',' in u:
                res['shape'] = [int(d) for d in u.split(',')]
            else:
                res['data'].append([int(d) for d in u.split(' ')])

        return res


class SpikeGLX:

    def __init__(self, root_dir):
        '''
        create neuropixels reader from 'root name' - e.g. the recording:

            /data/rec_1/npx_g0_t0.imec.ap.meta
            /data/rec_1/npx_g0_t0.imec.ap.bin
            /data/rec_1/npx_g0_t0.imec.lf.meta
            /data/rec_1/npx_g0_t0.imec.lf.bin

        would have a 'root name' of:

            /data/rec_1/npx_g0_t0.imec

        only a single recording is read/loaded via the root
        name & associated meta - no interpretation of g0_t0.imec, etc is
        performed at this layer.
        '''
        self._apmeta, self._ap_timeseries = None, None
        self._lfmeta, self._lf_timeseries = None, None

        self.root_dir = pathlib.Path(root_dir)

        try:
            meta_filepath = next(pathlib.Path(root_dir).glob('*.ap.meta'))
        except StopIteration:
            raise FileNotFoundError(f'No SpikeGLX file (.ap.meta) found at: {root_dir}')

        self.root_name = meta_filepath.name.replace('.ap.meta', '')

    @property
    def apmeta(self):
        if self._apmeta is None:
            self._apmeta = SpikeGLXMeta(self.root_dir / (self.root_name + '.ap.meta'))
        return self._apmeta

    @property
    def ap_timeseries(self):
        """
        AP data: (sample x channel)
        Data are stored as np.memmap with dtype: int16
        - to convert to microvolts, multiply with self.get_channel_bit_volts('ap')
        """
        if self._ap_timeseries is None:
            self.validate_file('ap')
            self._ap_timeseries = self._read_bin(self.root_dir / (self.root_name + '.ap.bin'))
        return self._ap_timeseries

    @property
    def lfmeta(self):
        if self._lfmeta is None:
            self._lfmeta = SpikeGLXMeta(self.root_dir / (self.root_name + '.lf.meta'))
        return self._lfmeta

    @property
    def lf_timeseries(self):
        """
        LFP data: (sample x channel)
        Data are stored as np.memmap with dtype: int16
        - to convert to microvolts, multiply with self.get_channel_bit_volts('lf')
        """
        if self._lf_timeseries is None:
            self.validate_file('lf')
            self._lf_timeseries = self._read_bin(self.root_dir / (self.root_name + '.lf.bin'))
        return self._lf_timeseries

    def get_channel_bit_volts(self, band='ap'):
        """
        Extract the recorded AP and LF channels' int16 to microvolts - no Sync (SY) channels
        Following the steps specified in: https://billkarsh.github.io/SpikeGLX/Support/SpikeGLX_Datafile_Tools.zip
                dataVolts = dataInt * Vmax / Imax / gain
        """
        vmax = float(self.apmeta.meta['imAiRangeMax'])

        if band == 'ap':
            imax = IMAX[self.apmeta.probe_model]
            imroTbl_data = self.apmeta.imroTbl['data']
            imroTbl_idx = 3
            chn_ind = self.apmeta.get_recording_channels_indices(exclude_sync=True)

        elif band == 'lf':
            imax = IMAX[self.lfmeta.probe_model]
            imroTbl_data = self.lfmeta.imroTbl['data']
            imroTbl_idx = 4
            chn_ind = self.lfmeta.get_recording_channels_indices(exclude_sync=True)
        else:
            raise ValueError(f'Unsupported band: {band} - Must be "ap" or "lf"')

        # extract channels' gains
        if 'imDatPrb_dock' in self.apmeta.meta:
            # NP 2.0; APGain = 80 for all AP (LF is computed from AP)
            chn_gains = [AP_GAIN] * len(imroTbl_data)
        else:
            # 3A, 3B1, 3B2 (NP 1.0)
            chn_gains = [c[imroTbl_idx] for c in imroTbl_data]

        chn_gains = np.array(chn_gains)[chn_ind]

        return vmax / imax / chn_gains * 1e6  # convert to uV as well

    def _read_bin(self, fname):
        nchan = self.apmeta.meta['nSavedChans']
        dtype = np.dtype((np.int16, nchan))
        return np.memmap(fname, dtype, 'r')

    def extract_spike_waveforms(self, spikes, channel_ind, n_wf=500, wf_win=(-32, 32)):
        """
        :param spikes: spike times (in second) to extract waveforms
        :param channel_ind: channel indices (of shankmap) to extract the waveforms from
        :param n_wf: number of spikes per unit to extract the waveforms
        :param wf_win: number of sample pre and post a spike
        :return: waveforms (in uV) - shape: (sample x channel x spike)
        """
        channel_bit_volts = self.get_channel_bit_volts('ap')[channel_ind]

        data = self.ap_timeseries

        spikes = np.round(spikes * self.apmeta.meta['imSampRate']).astype(int)  # convert to sample
        # ignore spikes at the beginning or end of raw data
        spikes = spikes[np.logical_and(spikes > -wf_win[0],
                                       spikes < data.shape[0] - wf_win[-1])]

        np.random.shuffle(spikes)
        spikes = spikes[:n_wf]
        if len(spikes) > 0:
            # waveform at each spike: (sample x channel x spike)
            spike_wfs = np.dstack([data[int(spk + wf_win[0]):int(spk + wf_win[-1]),
                                   channel_ind] * channel_bit_volts
                                   for spk in spikes])
            return spike_wfs
        else:  # if no spike found, return NaN of size (sample x channel x 1)
            return np.full((len(range(*wf_win)), len(channel_ind), 1), np.nan)

    def validate_file(self, file_type='ap'):
        file_path = self.root_dir / (self.root_name + f'.{file_type}.bin')
        file_size = file_path.stat().st_size

        meta_mapping = {
            'ap': self.apmeta,
            'lf': self.lfmeta}
        meta = meta_mapping[file_type]

        if file_size != meta.meta['fileSizeBytes']:
            raise IOError(f'File size error! {file_path} may be corrupted or in transfer?')


def _read_meta(meta_filepath):
    """
    Read metadata in 'k = v' format.

    The fields '~snsChanMap' and '~snsShankMap' are further parsed into
    'snsChanMap' and 'snsShankMap' dictionaries via calls to
    SpikeGLX._parse_chanmap and SpikeGLX._parse_shankmap.
    """

    res = {}
    with open(meta_filepath) as f:
        for l in (l.rstrip() for l in f):
            if '=' in l:
                try:
                    k, v = l.split('=')
                    v = handle_string(v)
                    res[k] = v
                except ValueError:
                    pass
    return res


# --------- Kilosort loaders --------------


class Kilosort:

    ks_files = [
        'params.py',
        'amplitudes.npy',
        'channel_map.npy',
        'channel_positions.npy',
        'pc_features.npy',
        'pc_feature_ind.npy',
        'similar_templates.npy',
        'spike_templates.npy',
        'spike_times.npy',
        'spike_times_sec.npy',
        'spike_times_sec_adj.npy',
        'template_features.npy',
        'template_feature_ind.npy',
        'templates.npy',
        'templates_ind.npy',
        'whitening_mat.npy',
        'whitening_mat_inv.npy',
        'spike_clusters.npy'
    ]

    ks_cluster_files = [
        'cluster_Amplitude.tsv',
        'cluster_ContamPct.tsv',
        'cluster_group.tsv',
        'cluster_KSLabel.tsv',
        'cluster_info.tsv'
    ]

    # keys to self.files, .data are file name e.g. self.data['params'], etc.
    ks_keys = [path.splitext(i)[0] for i in ks_files]

    def __init__(self, kilosort_dir):
        self._kilosort_dir = kilosort_dir
        self._files = {}
        self._data = None
        self._clusters = None

        self._info = {'time_created': datetime.fromtimestamp((kilosort_dir / 'params.py').stat().st_ctime),
                      'time_modified': datetime.fromtimestamp((kilosort_dir / 'params.py').stat().st_mtime)}

    @property
    def data(self):
        if self._data is None:
            self._stat()
        return self._data

    @property
    def info(self):
        return self._info

    def _stat(self):
        self._data = {}
        for i in Kilosort.ks_files:
            f = self._kilosort_dir / i

            if not f.exists():
                log.debug('skipping {} - doesnt exist'.format(f))
                continue

            base, ext = path.splitext(i)
            self._files[base] = f

            if i == 'params.py':
                log.debug('loading params.py {}'.format(f))
                # params.py is a 'key = val' file
                prm = {}
                for line in open(f, 'r').readlines():
                    k, v = line.strip('\n').split('=')
                    prm[k.strip()] = handle_string(v.strip())
                log.debug('prm: {}'.format(prm))
                self._data[base] = prm

            if ext == '.npy':
                log.debug('loading npy {}'.format(f))
                d = np.load(f, mmap_mode='r', allow_pickle=False, fix_imports=False)
                self._data[base] = d.squeeze()

        # Read the Cluster Groups
        if (self._kilosort_dir / 'cluster_groups.csv').exists():
            df = pd.read_csv(self._kilosort_dir / 'cluster_groups.csv', delimiter='\t')
            self._data['cluster_groups'] = np.array(df['group'].values)
            self._data['cluster_ids'] = np.array(df['cluster_id'].values)
        elif (self._kilosort_dir / 'cluster_KSLabel.tsv').exists():
            df = pd.read_csv(self._kilosort_dir / 'cluster_KSLabel.tsv', sep="\t", header=0)
            self._data['cluster_groups'] = np.array(df['KSLabel'].values)
            self._data['cluster_ids'] = np.array(df['cluster_id'].values)
        else:
            raise FileNotFoundError('Neither cluster_groups.csv nor cluster_KSLabel.tsv found!')

    def extract_curated_cluster_notes(self):
        curated_cluster_notes = {}
        for cluster_file in pathlib.Path(self._kilosort_dir).glob('cluster_*.tsv'):
            if cluster_file.name not in self.ks_cluster_files:
                curation_source = ''.join(cluster_file.stem.split('_')[1:])
                df = pd.read_csv(cluster_file, sep="\t", header=0)
                curated_cluster_notes[curation_source] = dict(
                    cluster_ids=np.array(df['cluster_id'].values),
                    cluster_notes=np.array([v.strip() for v in df[curation_source].values]))
        return curated_cluster_notes

    def extract_cluster_noise_label(self):
        """
        # labeling based on the noiseTemplate module - output to "cluster_group.tsv" file
        # (https://github.com/jenniferColonell/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/noise_templates)
        """
        cluster_group_tsv = pathlib.Path(self._kilosort_dir) / 'cluster_group.tsv'
        if cluster_group_tsv.exists():
            df = pd.read_csv(cluster_group_tsv, sep="\t", header=0)
            return dict(cluster_ids=np.array(df['cluster_id'].values),
                        cluster_notes=np.array(df['group'].values))
        else:
            return {}

    def extract_spike_depths(self):
        """ Reimplemented from https://github.com/cortex-lab/spikes/blob/master/analysis/ksDriftmap.m """
        ycoords = self.data['channel_positions'][:, 1]
        pc_features = self.data['pc_features'][:, 0, :]  # 1st PC only
        pc_features = np.where(pc_features < 0, 0, pc_features)

        # ---- compute center of mass of these features (spike depths) ----

        # which channels for each spike?
        spk_feature_ind = self.data['pc_feature_ind'][self.data['spike_templates'], :]
        # ycoords of those channels?
        spk_feature_ycoord = ycoords[spk_feature_ind]
        # center of mass is sum(coords.*features)/sum(features)
        self._data['spike_depths'] = np.sum(spk_feature_ycoord * pc_features**2, axis=1) / np.sum(pc_features**2, axis=1)

        # ---- extract spike sites ----
        max_site_ind = np.argmax(np.abs(self.data['templates']).max(axis=1), axis=1)
        spike_site_ind = max_site_ind[self.data['spike_templates']]
        self._data['spike_sites'] = self.data['channel_map'][spike_site_ind]


def extract_ks_waveforms(npx_dir, ks, n_wf=500, wf_win=(-41, 41), bit_volts=None):
    """
    :param npx_dir: directory to the ap.bin and ap.meta
    :param ks: instance of Kilosort
    :param n_wf: number of spikes per unit to extract the waveforms
    :param wf_win: number of sample pre and post a spike
    :param bit_volts: scalar required to convert int16 values into microvolts
    :return: dictionary of the clusters' waveform (sample x channel x spike) and snr per channel for each cluster
    """
    bin_fp = next(pathlib.Path(npx_dir).glob('*.ap.bin'))
    meta_fp = next(pathlib.Path(npx_dir).glob('*.ap.meta'))

    meta = SpikeGLXMeta(meta_fp)
    channel_num = meta.meta['nSavedChans']

    if bit_volts is None:
        bit_volts = npx_bit_volts[re.match('neuropixels (\d.0)', meta.probe_model).group()]

    raw_data = np.memmap(bin_fp, dtype='int16', mode='r')
    data = np.reshape(raw_data, (int(raw_data.size / channel_num), channel_num))

    chan_map = ks.data['channel_map']

    unit_wfs = {}
    for unit in tqdm(ks.data['cluster_ids']):
        spikes = ks.data['spike_times'][ks.data['spike_clusters'] == unit]
        np.random.shuffle(spikes)
        spikes = spikes[:n_wf]
        # ignore spikes at the beginning or end of raw data
        spikes = spikes[np.logical_and(spikes > wf_win[0], spikes < data.shape[0] - wf_win[-1])]

        unit_wfs[unit] = {}
        if len(spikes) > 0:
            # waveform at each spike: (sample x channel x spike)
            spike_wfs = np.dstack([data[int(spk+wf_win[0]):int(spk+wf_win[-1]), chan_map] for spk in spikes])
            spike_wfs = spike_wfs * bit_volts
            unit_wfs[unit]['snr'] = [calculate_wf_snr(chn_wfs)
                                     for chn_wfs in spike_wfs.transpose((1, 2, 0))]  # (channel x spike x sample)
            unit_wfs[unit]['mean_wf'] = np.nanmean(spike_wfs, axis=2)
        else:  # if no spike found, return NaN of size (sample x channel x 1)
            unit_wfs[unit]['snr'] = np.full((1, len(chan_map)), np.nan)
            unit_wfs[unit]['mean_wf'] = np.full((len(range(*wf_win)), len(chan_map)), np.nan)

    return unit_wfs


def calculate_wf_snr(W):
    """
    Calculate SNR of spike waveforms.
    Converted from Matlab by Xiaoxuan Jia
    ref: (Nordhausen et al., 1996; Suner et al., 2005)
    credit: https://github.com/AllenInstitute/ecephys_spike_sorting/blob/0643bfdc7e6fe87cd8c433ce5a4107d979687029/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py#L100

    Input:
    W : array of N waveforms (N x samples)

    Output:
    snr : signal-to-noise ratio for unit (scalar)
    """

    W_bar = np.nanmean(W, axis=0)
    A = np.max(W_bar) - np.min(W_bar)
    e = W - np.tile(W_bar, (np.shape(W)[0], 1))
    snr = A / (2 * np.nanstd(e.flatten()))
    return snr if not np.isinf(snr) else 0


def extract_clustering_info(cluster_output_dir, cluster_method):
    creation_time = None

    phy_curation_indicators = ['Merge clusters', 'Split cluster', 'Change metadata_group']
    # ---- Manual curation? ----
    phylog_fp = cluster_output_dir / 'phy.log'
    if phylog_fp.exists():
        phylog = pd.read_fwf(phylog_fp, colspecs=[(6, 40), (41, 250)])
        phylog.columns = ['meta', 'detail']
        curation_row = [bool(re.match('|'.join(phy_curation_indicators), str(s))) for s in phylog.detail]
        curation_prefix = 'curated_' if np.any(curation_row) else ''
        if creation_time is None and curation_prefix == 'curated_':
            row_meta = phylog.meta[np.where(curation_row)[0].max()]
            datetime_str = re.search('\d{2}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}', row_meta)
            if datetime_str:
                creation_time = datetime.strptime(datetime_str.group(), '%Y-%m-%d %H:%M:%S')
            else:
                creation_time = datetime.fromtimestamp(phylog_fp.stat().st_ctime)
                time_str = re.search('\d{2}:\d{2}:\d{2}', row_meta)
                if time_str:
                    creation_time = datetime.combine(creation_time.date(),
                                                     datetime.strptime(time_str.group(), '%H:%M:%S').time())
    else:
        curation_prefix = ''

    # ---- Quality control? ----
    metric_fp = cluster_output_dir / 'metrics.csv'
    if metric_fp.exists():
        qc_prefix = 'qc_'
        if creation_time is None:
            creation_time = datetime.fromtimestamp(metric_fp.stat().st_ctime)
    else:
        qc_prefix = 'raw_'

    if creation_time is None:
        if cluster_method == 'jrclust_v3':
            jr_fp = next(cluster_output_dir.glob('*jrc.mat'))
            creation_time = datetime.fromtimestamp(jr_fp.stat().st_ctime)
        elif cluster_method == 'jrclust_v4':
            jr_fp = next(cluster_output_dir.glob('*.ap_res.mat'))
            creation_time = datetime.fromtimestamp(jr_fp.stat().st_ctime)
        elif cluster_method == 'kilosort2':
            spk_fp = next(cluster_output_dir.glob('spike_times.npy'))
            creation_time = datetime.fromtimestamp(spk_fp.stat().st_ctime)

    label = ''.join([curation_prefix, qc_prefix])

    return creation_time, label


def read_SGLX_bin(sglx_bin_fp, chan_list):
    meta = readSGLX.readMeta(sglx_bin_fp)
    sampling_rate = readSGLX.SampRate(meta)
    raw_data = readSGLX.makeMemMapRaw(sglx_bin_fp, meta)
    data = raw_data[chan_list, :]
    if meta['typeThis'] == 'imec':
        # apply gain correction and convert to uV
        data = 1e6 * readSGLX.GainCorrectIM(data, chan_list, meta)
    else:
        # apply gain correction and convert to mV
        data = 1e3 * readSGLX.GainCorrectNI(data, chan_list, meta)
    return data, sampling_rate
