#! /usr/bin/env python

import os
import logging
import pathlib
from datetime import datetime
from os import path

from glob import glob
from tqdm import tqdm
import re
from itertools import repeat
import pandas as pd

import scipy.io as spio
import h5py
import numpy as np

import datajoint as dj
# import pdb

from pipeline import lab, experiment, ephys
from pipeline import dict_to_hash

from pipeline.ingest import ProbeInsertionError, ClusterMetricError, BitCodeError, IdenticalClusterResultError
from pipeline.ingest.ephys import get_ephys_paths
from pipeline.fixes import schema, FixHistory


os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

log = logging.getLogger(__name__)

npx_bit_volts = {'neuropixels 1.0': 2.34375,
                 'neuropixels 2.0': 0.763}  # uV per bit scaling factor for neuropixels probes


@schema
class FixedWaveformUnit(dj.Manual):
    """
    This table accompanies fix_0008, keeping track of the units with waveform being correctly updated.
    This tracking is not extremely important, as a rerun of this fix_0008 will still yield the correct results
    """
    definition = """ # This table accompanies fix_0008
    -> FixHistory
    -> ephys.Unit
    ---
    fixed: bool
    """


def update_waveform(session_keys={}):
    """
    Updating unit-waveform, where a unit's waveform is updated to be from the peak-channel
        and not from the 1-st channel as before
    Applicable to only kilosort2 clustering results and not jrclust
    """
    sessions_2_update = experiment.Session & (ephys.ProbeInsertion.proj()
                                              * ephys.Unit & 'clustering_method = "kilosort2"') & session_keys
    sessions_2_update = sessions_2_update - FixedWaveformUnit

    if not sessions_2_update:
        return

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}
    FixHistory.insert1(fix_hist_key)

    for key in sessions_2_update.fetch('KEY'):
        log.info('\n======================================================')
        log.info('Waveform update for key: {k}'.format(k=key))

        #
        # Find Ephys Recording
        #
        key = (experiment.Session & key).fetch1()
        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session.proj(..., '-session_time')) & key).fetch1()

        rigpaths = get_ephys_paths()
        h2o = sinfo['water_restriction_number']

        sess_time = (datetime.min + key['session_time']).time()
        sess_datetime = datetime.combine(key['session_date'], sess_time)

        for rigpath in rigpaths:
            dpath, dglob = _get_sess_dir(rigpath, h2o, sess_datetime)
            if dpath is not None:
                break

        if dpath is not None:
            log.info('Found session folder: {}'.format(dpath))
        else:
            log.warning('Error - No session folder found for {}/{}'.format(h2o, key['session_date']))
            return

        try:
            clustering_files = _match_probe_to_ephys(h2o, dpath, dglob)
        except FileNotFoundError as e:
            log.warning(str(e) + '. Skipping...')
            return

        with ephys.Unit.connection.transaction:
            for probe_no, (f, cluster_method, npx_meta) in clustering_files.items():
                try:
                    log.info('------ Start loading clustering results for probe: {} ------'.format(probe_no))
                    loader = cluster_loader_map[cluster_method]
                    dj.conn().ping()
                    _wf_update(loader(sinfo, *f), probe_no, npx_meta, rigpath)
                except (ProbeInsertionError, ClusterMetricError, FileNotFoundError) as e:
                    dj.conn().cancel_transaction()  # either successful fix of all probes, or none at all
                    if isinstance(e, ProbeInsertionError):
                        log.warning('Probe Insertion Error: \n{}. \nSkipping...'.format(str(e)))
                    else:
                        log.warning('Error: {}'.format(str(e)))
                    return

            with dj.config(safemode=False):
                (ephys.UnitCellType & key).delete()

            FixedWaveformUnit.insert([{**fix_hist_key, **ukey, 'fixed': 1} for ukey in (ephys.Unit & key).fetch('KEY')])


def _wf_update(data, probe, npx_meta, rigpath):

    sinfo = data['sinfo']
    ef_path = data['ef_path']
    skey = data['skey']
    method = data['method']
    hz = data['hz'] if data['hz'] else npx_meta.meta['imSampRate']
    spikes = data['spikes']
    spike_sites = data['spike_sites']
    units = data['units']
    unit_wav = data['unit_wav']  # (unit x channel x sample)
    unit_notes = data['unit_notes']
    unit_xpos = data['unit_xpos']
    unit_ypos = data['unit_ypos']
    unit_amp = data['unit_amp']
    vmax_unit_site = data['vmax_unit_site']
    trial_start = data['trial_start']
    trial_go = data['trial_go']
    clustering_label = data['clustering_label']

    log.info('-- Start insertions for probe: {} - Clustering method: {} - Label: {}'.format(probe, method,
                                                                                            clustering_label))

    assert len(trial_start) == len(trial_go)

    # probe insertion key
    insertion_key = {'subject_id': sinfo['subject_id'],
                     'session': sinfo['session'],
                     'insertion_number': probe}

    # remove noise clusters
    if method in ['jrclust_v3', 'jrclust_v4']:
        units, spikes, spike_sites = (v[i] for v, i in zip(
            (units, spikes, spike_sites), repeat((units > 0))))

    # scale amplitudes by uV/bit scaling factor (for kilosort2)
    if method in ['kilosort2']:
        bit_volts = npx_bit_volts[re.match('neuropixels (\d.0)', npx_meta.probe_model).group()]
        unit_amp = unit_amp * bit_volts

    # _update() on waveform
    for i, u in enumerate(set(units)):
        wf = unit_wav[i][vmax_unit_site[i]]
        (ephys.Unit & {**skey, **insertion_key, 'clustering_method': method,
                       'unit': u})._update('waveform', wf)


def _gen_probe_insert(sinfo, probe, npx_meta):
    '''
    generate probe insertion for session / probe - for neuropixels recording

    Arguments:

      - sinfo: lab.WaterRestriction * lab.Subject * experiment.Session
      - probe: probe id

    '''

    part_no = npx_meta.probe_SN

    e_config_key = _gen_electrode_config(npx_meta)

    # ------ ProbeInsertion ------
    insertion_key = {'subject_id': sinfo['subject_id'],
                     'session': sinfo['session'],
                     'insertion_number': probe}

    # add probe insertion
    log.info('.. creating probe insertion')

    lab.Probe.insert1({'probe': part_no, 'probe_type': e_config_key['probe_type']}, skip_duplicates=True)

    ephys.ProbeInsertion.insert1({**insertion_key, **e_config_key, 'probe': part_no})

    ephys.ProbeInsertion.RecordingSystemSetup.insert1({**insertion_key, 'sampling_rate': npx_meta.meta['imSampRate']})

    return insertion_key, e_config_key


def _gen_electrode_config(npx_meta):
    """
    Generate and insert (if needed) an ElectrodeConfiguration based on the specified neuropixels meta information
    """

    if re.search('(1.0|2.0)', npx_meta.probe_model):
        eg_members = []
        probe_type = {'probe_type': npx_meta.probe_model}
        q_electrodes = lab.ProbeType.Electrode & probe_type
        for shank, shank_col, shank_row, is_used in npx_meta.shankmap['data']:
            electrode = (q_electrodes & {'shank': shank + 1,  # shank is 1-indexed in this pipeline
                                         'shank_col': shank_col + 1,
                                         'shank_row': shank_row + 1}).fetch1('KEY')
            eg_members.append({**electrode, 'is_used': is_used, 'electrode_group': 0})
    else:
        raise NotImplementedError('Processing for neuropixels probe model {} not yet implemented'.format(
            npx_meta.probe_model))

    # ---- compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode) ----
    ec_hash = dict_to_hash({k['electrode']: k for k in eg_members})

    el_list = sorted([k['electrode'] for k in eg_members])
    el_jumps = [-1] + np.where(np.diff(el_list) > 1)[0].tolist() + [len(el_list) - 1]
    ec_name = '; '.join([f'{el_list[s + 1]}-{el_list[e]}' for s, e in zip(el_jumps[:-1], el_jumps[1:])])

    e_config = {**probe_type, 'electrode_config_name': ec_name}

    # ---- make new ElectrodeConfig if needed ----
    if not (lab.ElectrodeConfig & {'electrode_config_hash': ec_hash}):
        log.info('.. Probe type: {} - creating lab.ElectrodeConfig: {}'.format(npx_meta.probe_model, ec_name))

        lab.ElectrodeConfig.insert1({**e_config, 'electrode_config_hash': ec_hash})

        lab.ElectrodeConfig.ElectrodeGroup.insert1({**e_config, 'electrode_group': 0})

        lab.ElectrodeConfig.Electrode.insert({**e_config, **m} for m in eg_members)

    return e_config


# ======== Loaders for clustering results ========
def _decode_notes(fh, notes):
    '''
    dereference and decode unit notes, translate to local labels
    '''
    note_map = {'single': 'good', 'ok': 'ok', 'multi': 'multi',
                '\x00\x00': 'all'}  # 'all' is default / null label
    decoded_notes = []
    for n in notes:
        note_val = str().join(chr(c) for c in fh[n])
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
    ef = h5py.File(str(ef_path), mode = 'r')  # ephys file

    # -- trial-info from bitcode --
    try:
        sync_behav, sync_ephys, trial_fix, trial_go, trial_start = read_bitcode(fpath.parent, h2o, skey)
    except FileNotFoundError as e:
        raise e

    # extract unit data

    hz = ef['P']['sRateHz'][0][0]  # sampling rate

    spikes = ef['viTime_spk'][0]  # spike times
    spike_sites = ef['viSite_spk'][0]  # spike electrode

    units = ef['S_clu']['viClu'][0]  # spike:unit id
    unit_wav = ef['S_clu']['trWav_raw_clu']  # waveform (unit x channel x sample)

    unit_notes = ef['S_clu']['csNote_clu'][0]  # curation notes
    unit_notes = _decode_notes(ef, unit_notes)

    unit_xpos = ef['S_clu']['vrPosX_clu'][0]  # x position
    unit_ypos = ef['S_clu']['vrPosY_clu'][0]  # y position

    unit_amp = ef['S_clu']['vrVpp_uv_clu'][0]  # amplitude
    unit_snr = ef['S_clu']['vrSnr_clu'][0]  # signal to noise

    vmax_unit_site = ef['S_clu']['viSite_clu']  # max amplitude site
    vmax_unit_site = np.array(vmax_unit_site[:].flatten(), dtype = np.int64)

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
        'clustering_label': clustering_label
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
    ef = h5py.File(str(ef_path), mode = 'r')  # ephys file

    # -- trial-info from bitcode --
    try:
        sync_behav, sync_ephys, trial_fix, trial_go, trial_start = read_bitcode(fpath.parent, h2o, skey)
    except FileNotFoundError as e:
        raise e

    # extract unit data
    hz = None  # sampling rate  (N/A from jrclustv4, use from npx_meta)

    spikes = ef['spikeTimes'][0]  # spikes times
    spike_sites = ef['spikeSites'][0]  # spike electrode

    units = ef['spikeClusters'][0]  # spike:unit id
    unit_wav = ef['meanWfLocalRaw']  # waveform

    unit_notes = ef['clusterNotes']  # curation notes
    unit_notes = _decode_notes(ef, unit_notes[:].flatten())

    unit_xpos = ef['clusterCentroids'][0]  # x position
    unit_ypos = ef['clusterCentroids'][1]  # y position

    unit_amp = ef['unitVppRaw'][0]  # amplitude
    unit_snr = ef['unitSNR'][0]  # signal to noise

    vmax_unit_site = ef['clusterSites']  # max amplitude site
    vmax_unit_site = np.array(vmax_unit_site[:].flatten(), dtype = np.int64)

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
        'clustering_label': clustering_label
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
        sync_behav, sync_ephys, trial_fix, trial_go, trial_start = read_bitcode(npx_dir, h2o, skey)
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
    # -- spike_sites --
    # reimplemented from: https://github.com/JaneliaSciComp/JRCLUST/blob/master/%2Bjrclust/%2Bimport/kilosort.m
    spike_sites = np.full(spike_times.shape, np.nan)
    for template_idx, template in enumerate(ks.data['templates']):
        site_idx = np.abs(np.abs(template).max(axis = 0)).argmax()
        spike_sites[ks.data['spike_templates'] == template_idx] = ks.data['channel_map'][site_idx]

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
        metrics.set_index('cluster_id', inplace = True)
        metrics = metrics[metrics.index == valid_units]

        # peak_chn, amp, snr
        vmax_unit_site = metrics.peak_channel.values  # peak channel
        unit_amp = metrics.amplitude.values  # amp
        unit_snr = metrics.snr.values  # snr
        unit_snr = np.where(np.logical_or(np.isinf(unit_snr), np.isnan(unit_snr)), 0,
                            unit_snr)  # set value to 0 if INF or NaN
        # unit x, y
        vmax_unit_site_idx = [np.where(ks.data['channel_map'] == peak_site)[0][0] for peak_site in vmax_unit_site]
        unit_xpos = [ks.data['channel_positions'][site_idx, 0] for site_idx in vmax_unit_site_idx]
        unit_ypos = [ks.data['channel_positions'][site_idx, 1] for site_idx in vmax_unit_site_idx]
        # unit waveforms
        unit_wav = np.load(ks_dir / 'mean_waveforms.npy')
        unit_wav = unit_wav[valid_units, :, :]  # unit x channel x sample
    else:
        vmax_unit_site, unit_xpos, unit_ypos, unit_amp = [], [], [], []
        for unit in valid_units:
            template_idx = ks.data['spike_templates'][np.where(ks.data['spike_clusters'] == unit)[0][0]]
            chn_templates = ks.data['templates'][template_idx, :, :]
            site_idx = np.abs(np.abs(chn_templates).max(axis = 0)).argmax()
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

        unit_wfs = extract_ks_waveforms(npx_dir, ks, wf_win = [-int(ks.data['templates'].shape[1] / 2),
                                                               int(ks.data['templates'].shape[1] / 2)])
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

    data = {
        'sinfo': sinfo,
        'ef_path': ks_dir,
        'skey': skey,
        'method': 'kilosort2',
        'hz': hz,
        'spikes': spike_times.astype(int),
        'spike_sites': spike_sites + 1,  # channel numbering in this pipeline is 1-based indexed
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
        'clustering_label': clustering_label
    }

    return data


cluster_loader_map = {'jrclust_v3': _load_jrclust_v3,
                      'jrclust_v4': _load_jrclust_v4,
                      'kilosort2': _load_kilosort2}


# ======== Helpers for directory navigation ========
def _get_sess_dir(rigpath, h2o, sess_datetime):
    dpath, dglob = None, None
    if pathlib.Path(rigpath, h2o, sess_datetime.date().strftime('%Y%m%d')).exists():
        dpath = pathlib.Path(rigpath, h2o, sess_datetime.date().strftime('%Y%m%d'))
        dglob = '[0-9]/{}'  # probe directory pattern
    else:
        sess_dirs = list(pathlib.Path(rigpath, h2o).glob('*{}_{}_*'.format(
            h2o, sess_datetime.date().strftime('%m%d%y'))))
        for sess_dir in sess_dirs:
            try:
                npx_meta = NeuropixelsMeta(next(sess_dir.rglob('{}_*.ap.meta'.format(h2o))))
            except StopIteration:
                continue
            # ensuring time difference between behavior-start and ephys-start is no more than 2 minutes - this is to handle multiple sessions in a day
            start_time_difference = abs((npx_meta.recording_time - sess_datetime).total_seconds())
            if start_time_difference <= 120:
                dpath = sess_dir
                dglob = '{}_{}_*_imec[0-9]'.format(h2o, sess_datetime.date().strftime(
                    '%m%d%y')) + '/{}'  # probe directory pattern
                break
            else:
                log.info(
                    'Found {} - difference in behavior and ephys start-time: {} seconds (more than 2 minutes). Skipping...'.format(
                        sess_dir, start_time_difference))

    return dpath, dglob


def _match_probe_to_ephys(h2o, dpath, dglob):
    """
    Based on the identified spike sorted file(s), match the probe number (i.e. 1, 2, 3) to the cluster filepath, loader, and npx_meta
    Return a dict, e.g.:
    {
     1: (cluster_fp, cluster_method, npx_meta),
     2: (cluster_fp, cluster_method, npx_meta),
     3: (cluster_fp, cluster_method, npx_meta),
    }
    """
    # npx ap.meta: '{}_*.imec.ap.meta'.format(h2o)
    npx_meta_files = list(dpath.glob(dglob.format('{}_*.ap.meta'.format(h2o))))
    if not npx_meta_files:
        raise FileNotFoundError('Error - no ap.meta files at {}'.format(dpath))

    jrclustv3spec = '{}_*_jrc.mat'.format(h2o)
    jrclustv4spec = '{}_*.ap_res.mat'.format(h2o)
    ks2specs = ('mean_waveforms.npy', 'spike_times.npy')  # prioritize QC output, then orig

    clustered_probes = {}
    for meta_file in npx_meta_files:
        probe_dir = meta_file.parent
        probe_number = re.search('(imec)?\d{1}$', probe_dir.name).group()
        probe_number = int(probe_number.replace('imec', '')) + 1 if 'imec' in probe_number else int(probe_number)

        # JRClust v3
        v3files = [((f,), 'jrclust_v3') for f in probe_dir.glob(jrclustv3spec)]
        # JRClust v4
        v4files = [((f,), 'jrclust_v4') for f in probe_dir.glob(jrclustv4spec)]
        # Kilosort
        ks2spec = ks2specs[0] if len(list(probe_dir.rglob(ks2specs[0]))) > 0 else ks2specs[1]
        ks2files = [((f.parent, probe_dir), 'kilosort2') for f in probe_dir.rglob(ks2spec)]

        if len(ks2files) > 1:
            raise ValueError('Multiple Kilosort outputs found at: {}'.format([str(x[0]) for x in ks2files]))

        clustering_results = v4files + v3files + ks2files

        if len(clustering_results) < 1:
            raise FileNotFoundError('Error - No clustering results found at {}'.format(probe_dir))
        elif len(clustering_results) > 1:
            log.warning(
                'Found multiple clustering results at {probe_dir}. Prioritize JRC4 > JRC3 > KS2'.format(probe_dir))

        fp, loader = clustering_results[0]
        clustered_probes[probe_number] = (fp, loader, NeuropixelsMeta(meta_file))

    return clustered_probes


def read_bitcode(bitcode_dir, h2o, skey):
    """
    Load bitcode file from specified dir - example bitcode format: e.g. 'SC022_030319_Imec3_bitcode.mat'
    :return: sync_behav, sync_ephys, trial_fix, trial_go, trial_start
    """
    bitcode_dir = pathlib.Path(bitcode_dir)
    try:
        bf_path = next(bitcode_dir.glob('{}_*bitcode.mat'.format(h2o)))
    except StopIteration:
        raise FileNotFoundError('No bitcode for {} found in {}'.format(h2o, bitcode_dir))

    log.info('.... loading bitcode file: {}'.format(str(bf_path)))

    bf = spio.loadmat(str(bf_path))

    trial_start = bf['sTrig'].flatten()  # trial start
    trial_go = bf['goCue'].flatten()  # trial go cues

    # check if there are `FreeWater` trials (i.e. no trial_go), if so, set those with trial_go value of NaN
    if len(trial_go) < len(trial_start):

        if len(experiment.BehaviorTrial & skey) != len(trial_start):
            raise BitCodeError('Mismatch sTrig ({} elements) and total behavior trials ({} trials)'.format(
                len(trial_start), len(experiment.BehaviorTrial & skey)))

        if len(experiment.BehaviorTrial & skey & 'free_water = 0') != len(trial_go):
            raise BitCodeError('Mismatch goCue ({} elements) and non-FreeWater trials ({} trials)'.format(
                len(trial_go), len(experiment.BehaviorTrial & skey & 'free_water = 0')))

        all_tr = (experiment.BehaviorTrial & skey).fetch('trial', order_by = 'trial')
        no_free_water_tr = (experiment.BehaviorTrial & skey & 'free_water = 0').fetch('trial', order_by = 'trial')
        is_go_trial = np.in1d(all_tr, no_free_water_tr)

        trial_go_full = np.full_like(trial_start, np.nan)
        trial_go_full[is_go_trial] = trial_go
        trial_go = trial_go_full

    sync_ephys = bf['bitCodeS']  # ephys sync codes
    sync_behav = (experiment.TrialNote()  # behavior sync codes
                  & {**skey, 'trial_note_type': 'bitcode'}).fetch('trial_note', order_by = 'trial')
    trial_fix = bf['trialNum'].flatten() if 'trialNum' in bf else None

    return sync_behav, sync_ephys, trial_fix, trial_go, trial_start


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


class NeuropixelsMeta:

    def __init__(self, meta_filepath):
        # a good processing reference: https://github.com/jenniferColonell/Neuropixels_evaluation_tools/blob/master/SGLXMetaToCoords.m

        self.fname = meta_filepath
        self.meta = self._read_meta()

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

    def _read_meta(self):
        '''
        Read metadata in 'k = v' format.

        The fields '~snsChanMap' and '~snsShankMap' are further parsed into
        'snsChanMap' and 'snsShankMap' dictionaries via calls to
        Neuropixels._parse_chanmap and Neuropixels._parse_shankmap.
        '''

        fname = self.fname

        res = {}
        with open(fname) as f:
            for l in (l.rstrip() for l in f):
                if '=' in l:
                    try:
                        k, v = l.split('=')
                        v = handle_string(v)
                        res[k] = v
                    except ValueError:
                        pass
        return res

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
        'spike_clusters.npy',
        'cluster_groups.csv',
        'cluster_KSLabel.tsv'
    ]

    # keys to self.files, .data are file name e.g. self.data['params'], etc.
    ks_keys = [path.splitext(i)[0] for i in ks_files]

    def __init__(self, dname):
        self._dname = dname
        self._files = {}
        self._data = None
        self._clusters = None

        self._info = {'time_created': datetime.fromtimestamp((dname / 'params.py').stat().st_ctime),
                      'time_modified': datetime.fromtimestamp((dname / 'params.py').stat().st_mtime)}

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
            f = self._dname / i

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
                d = np.load(f, mmap_mode = 'r', allow_pickle = False, fix_imports = False)
                self._data[base] = np.reshape(d, d.shape[0]) if d.ndim == 2 and d.shape[1] == 1 else d

        # Read the Cluster Groups
        if (self._dname / 'cluster_groups.csv').exists():
            df = pd.read_csv(self._dname / 'cluster_groups.csv', delimiter = '\t')
            self._data['cluster_groups'] = np.array(df['group'].values)
            self._data['cluster_ids'] = np.array(df['cluster_id'].values)
        elif (self._dname / 'cluster_KSLabel.tsv').exists():
            df = pd.read_csv(self._dname / 'cluster_KSLabel.tsv', sep = "\t", header = 0)
            self._data['cluster_groups'] = np.array(df['KSLabel'].values)
            self._data['cluster_ids'] = np.array(df['cluster_id'].values)
        else:
            raise FileNotFoundError('Neither cluster_groups.csv nor cluster_KSLabel.tsv found!')


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

    meta = NeuropixelsMeta(meta_fp)
    channel_num = meta.meta['nSavedChans']

    if bit_volts is None:
        bit_volts = npx_bit_volts[re.match('neuropixels (\d.0)', meta.probe_model).group()]

    raw_data = np.memmap(bin_fp, dtype = 'int16', mode = 'r')
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
            spike_wfs = np.dstack([data[int(spk + wf_win[0]):int(spk + wf_win[-1]), chan_map] for spk in spikes])
            spike_wfs = spike_wfs * bit_volts
            unit_wfs[unit]['snr'] = [calculate_wf_snr(chn_wfs)
                                     for chn_wfs in spike_wfs.transpose((1, 2, 0))]  # (channel x spike x sample)
            unit_wfs[unit]['mean_wf'] = np.nanmean(spike_wfs, axis = 2)
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

    W_bar = np.nanmean(W, axis = 0)
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
        phylog = pd.read_fwf(phylog_fp, colspecs = [(6, 40), (41, 250)])
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


if __name__ == '__main__':
    update_waveform()
