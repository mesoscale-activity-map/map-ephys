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
#import pdb

from pipeline import lab, experiment, ephys, report, tracking
from pipeline import InsertBuffer, dict_value_to_hash

from .. import get_schema_name
from . import readSGLX
from . import ProbeInsertionError, ClusterMetricError, BitCodeError, IdenticalClusterResultError

schema = dj.schema(get_schema_name('ingest_ephys'))

os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

log = logging.getLogger(__name__)

npx_bit_volts = {'neuropixels 1.0': 2.34375, 'neuropixels 2.0': 0.763}  # uV per bit scaling factor for neuropixels probes


def get_ephys_paths():
    """
    retrieve ephys paths from dj.config
    config should be in dj.config of the format:

      dj.config = {
        ...,
        'custom': {
          'ephys_data_paths': ['/path/string', '/path2/string']
        }
        ...
      }
    """
    return dj.config.get('custom', {}).get('ephys_data_paths', None)


@schema
class EphysIngest(dj.Imported):
    # subpaths like: \2017-10-21\tw5ap_imec3_opt3_jrc.mat

    definition = """
        -> experiment.Session
    """

    class EphysFile(dj.Part):
        ''' files in rig-specific storage '''
        definition = """
        -> EphysIngest
        probe_insertion_number:         tinyint         # electrode_group
        ephys_file:                     varchar(255)    # rig file subpath
        """

    key_source = experiment.Session - ephys.ProbeInsertion

    def make(self, key):
        '''
        Ephys .make() function
        '''

        log.info('\n======================================================')
        log.info('EphysIngest().make(): key: {k}'.format(k=key))

        do_ephys_ingest(key)

    def _load(self, data, probe, npx_meta, rigpath, probe_insertion_exists=False, into_archive=False):

        sinfo = data['sinfo']
        ef_path = data['ef_path']
        skey = data['skey']
        method = data['method']
        hz = data['hz'] if data['hz'] else npx_meta.meta['imSampRate']
        spikes = data['spikes']
        spike_sites = data['spike_sites']
        spike_depths = data['spike_depths']
        units = data['units']
        unit_wav = data['unit_wav']  # (unit x channel x sample)
        unit_notes = data['unit_notes']
        unit_xpos = data['unit_xpos']
        unit_ypos = data['unit_ypos']
        unit_amp = data['unit_amp']
        unit_snr = data['unit_snr']
        vmax_unit_site = data['vmax_unit_site']
        trial_start = data['trial_start']
        trial_go = data['trial_go']
        sync_ephys = data['sync_ephys']
        sync_behav = data['sync_behav']
        trial_fix = data['trial_fix']
        metrics = data['metrics']  # either None or a pd.DataFrame loaded from 'metrics.csv'
        creation_time = data['creation_time']
        clustering_label = data['clustering_label']
        bitcode_raw = data['bitcode_raw']

        log.info('-- Start insertions for probe: {} - Clustering method: {} - Label: {}'.format(probe, method, clustering_label))

        assert len(trial_start) == len(trial_go), 'Unequal number of bitcode "trial_start" ({}) and "trial_go" ({})'.format(len(trial_start), len(trial_go))

        # create probe insertion records
        if into_archive:
            probe_insertion_exists = True

        try:
            insertion_key, e_config_key = _gen_probe_insert(sinfo, probe, npx_meta, probe_insertion_exists=probe_insertion_exists)
        except (NotImplementedError, dj.DataJointError) as e:
            raise ProbeInsertionError(str(e))

        # account for the buffer period before trial_start
        if 'trgTTLMarginS' in npx_meta.meta:
            buffer_sample_count = np.round(npx_meta.meta['trgTTLMarginS'] * npx_meta.meta['imSampRate']).astype(int)
            buffer_sample_count = np.round(buffer_sample_count).astype(int)
        else:
            buffer_sample_count = 0

        trial_start = trial_start - buffer_sample_count

        # remove noise clusters
        if method in ['jrclust_v3', 'jrclust_v4']:
            units, spikes, spike_sites, spike_depths = (v[i] for v, i in zip(
                (units, spikes, spike_sites, spike_depths), repeat((units > 0))))

        # scale amplitudes by uV/bit scaling factor (for kilosort2)
        if method in ['kilosort2']:
            if 'qc' not in clustering_label:
                bit_volts = npx_bit_volts[re.match('neuropixels (\d.0)', npx_meta.probe_model).group()]
                unit_amp = unit_amp * bit_volts

        # Determine trial (re)numbering for ephys:
        #
        # - if ephys & bitcode match, determine ephys-to-behavior trial shift
        #   when needed for different-length recordings
        # - otherwise, use trial number correction array (bf['trialNum'])

        # First, find the first Ephys trial in behavior data (sometimes ephys is not started in time)
        sync_behav_start = np.where(sync_behav == sync_ephys[0])[0][0]   
       
        # Ephys trial will never start BEFORE behavioral trial, so the above line is always correct.
        # But due to pybpod bug, sometimes the last behavioral trial is invalid, making Ephys even LONGER than behavior. 
        # Therefore, we must find out both sync_behav_range AND sync_ephys_range (otherwise the next `if not np.all` could fail)
        sync_behav_range = sync_behav[sync_behav_start:][:len(sync_ephys)]   # Note that this will not generate error even if len(sync_ephys) > len(behavior)
        shared_trial_num = len(sync_behav_range)
        sync_ephys_range = sync_ephys[:shared_trial_num]   # Now they must have the same length

        if not np.all(np.equal(sync_ephys_range, sync_behav_range)):
            if trial_fix is not None:
                log.info('ephys/bitcode trial mismatch - fix using "trialNum"')
                trials = trial_fix
            else:
                raise Exception('Bitcode Mismatch - Fix with "trialNum" not available')
        else:
            # TODO: recheck the logic here!
            if len(sync_behav) < len(sync_ephys):
                start_behav = np.where(sync_behav[0] == sync_ephys)[0][0]  # TODO: This is problematic because ephys never leads behavior, otherwise the logic above is wrong
            elif len(sync_behav) > len(sync_ephys):
                start_behav = - np.where(sync_ephys[0] == sync_behav)[0][0]
            else:
                start_behav = 0
            trial_indices = np.arange(shared_trial_num) - start_behav

            # mapping to the behav-trial numbering
            # "trials" here is just the 0-based indices of the behavioral trials
            behav_trials = (experiment.SessionTrial & skey).fetch('trial', order_by='trial')
            trials = behav_trials[trial_indices]
            
            # TODO: this is a workaround to deal with the case where ephys stops later than behavior 
            # but with the assumption that ephys will NEVER start earlier than behavior
            trial_start = trial_start[:shared_trial_num]  # Truncate ephys 'trial_start' at the tail
            # And also truncate the ingestion of digital markers (see immediate below)

        assert len(trial_start) == len(trials), 'Unequal number of bitcode "trial_start" ({}) and ingested behavior trials ({})'.format(len(trial_start), len(trials))

        # -- Ingest time markers from NIDQ channels --
        # This is redudant for delay response task because aligning spikes to the go-cue is enough (trial_spikes below)
        # But this is critical for the foraging task, because we need global session-wise times to plot flexibly-aligned PSTHs (in particular, spikes during ITI).
        # However, we CANNOT get this from behavior pybpod .csv files (PC-TIME is inaccurate, whereas BPOD-TIME is trial-based)
        if probe == 1 and 'digMarkerPerTrial' in bitcode_raw:   # Only import once for one session
            insert_ephys_events(skey, bitcode_raw, shared_trial_num)

        # trialize the spikes & subtract go cue
        t, trial_spikes, trial_units = 0, [], []

        spike_trial_num = np.full_like(spikes, np.nan)

        while t < len(trial_start) - 1:

            s0, s1 = trial_start[t], trial_start[t+1]

            trial_idx = np.where((spikes > s0) & (spikes < s1))
            spike_trial_num[trial_idx] = trials[t]   # Assign (behavioral) trial number to each spike

            trial_spikes.append(spikes[trial_idx] - trial_go[t])
            trial_units.append(units[trial_idx])

            t += 1

        # ... including the last trial
        trial_idx = np.where((spikes > s1))
        trial_spikes.append(spikes[trial_idx] - trial_go[t])
        trial_units.append(units[trial_idx])

        trial_spikes = np.array(trial_spikes)
        trial_units = np.array(trial_units)

        # convert spike data to seconds
        spikes = spikes / hz
        trial_start = trial_start / hz
        trial_spikes = trial_spikes / hz

        # build spike arrays
        unit_spikes = np.array([spikes[np.where(units == u)] for u in set(units)]) - trial_start[0]

        unit_trial_spikes = np.array(
            [[trial_spikes[t][np.where(trial_units[t] == u)]
              for t in range(len(trials))] for u in set(units)])

        q_electrodes = lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode & e_config_key
        site2electrode_map = {}
        for recorded_site, (shank, shank_col, shank_row, _) in enumerate(npx_meta.shankmap['data']):
            site2electrode_map[recorded_site + 1] = (q_electrodes
                                                     & {'shank': shank + 1,  # this is a 1-indexed pipeline
                                                        'shank_col': shank_col + 1,
                                                        'shank_row': shank_row + 1}).fetch1('KEY')

        spike_sites = np.array([site2electrode_map[s]['electrode'] for s in spike_sites])
        unit_spike_sites = np.array([spike_sites[np.where(units == u)] for u in set(units)])
        unit_spike_depths = np.array([spike_depths[np.where(units == u)] for u in set(units)])

        if into_archive:
            log.info('.. inserting clustering timestamp and label')
            archival_time = datetime.now()

            archive_key = {**skey, 'insertion_number': probe,
                           'clustering_method': method, 'clustering_time': creation_time}

            ephys.ArchivedClustering.insert1({
                **archive_key, 'quality_control': bool('qc' in clustering_label),
                'manual_curation': bool('curated' in clustering_label),
                'archival_time': archival_time}, allow_direct_insert=True)
            ephys.ArchivedClustering.EphysFile.insert1({
                **archive_key,
                'ephys_file': ef_path.relative_to(rigpath).as_posix()},
                allow_direct_insert=True)

            unit_spike_trial_num = np.array([spike_trial_num[np.where(units == u)] for u in set(units)])

            with InsertBuffer(ephys.ArchivedClustering.Unit, 10, skip_duplicates=True,
                              allow_direct_insert=True) as ib:

                for i, u in enumerate(set(units)):
                    if method in ['jrclust_v3', 'jrclust_v4']:
                        wf_chn_idx = 0
                    elif method in ['kilosort2']:
                        wf_chn_idx = np.where(data['ks_channel_map'] == vmax_unit_site[i])[0][0]
                    ib.insert1({**archive_key,
                                **site2electrode_map[vmax_unit_site[i]],
                                'clustering_method': method,
                                'unit': u,
                                'unit_quality': unit_notes[i],
                                'unit_posx': unit_xpos[i],
                                'unit_posy': unit_ypos[i],
                                'spike_times': unit_spikes[i],
                                'spike_sites': unit_spike_sites[i],
                                'spike_depths': unit_spike_depths[i],
                                'trial_spike': unit_spike_trial_num[i],
                                'waveform': unit_wav[i][wf_chn_idx]})
                    if ib.flush():
                        log.debug('.... {}'.format(u))

            if metrics is not None:
                metrics.columns = [c.lower() for c in metrics.columns]  # lower-case col names
                # -- confirm correct attribute names from the PD
                required_columns = np.setdiff1d(ephys.ClusterMetric.heading.names + ephys.WaveformMetric.heading.names,
                                                ephys.Unit.primary_key)
                missing_columns = np.setdiff1d(required_columns, metrics.columns)

                if len(missing_columns) > 0:
                    raise ClusterMetricError('Missing or misnamed column(s) in metrics.csv: {}'.format(missing_columns))

                metrics = dict(metrics.T)

                log.info('.. inserting cluster metrics and waveform metrics')
                dj.conn().ping()
                ephys.ArchivedClustering.ClusterMetric.insert(
                    [{**archive_key, 'unit': u, **metrics[u]}
                     for u in set(units)], ignore_extra_fields=True, allow_direct_insert=True)
                ephys.ArchivedClustering.WaveformMetric.insert(
                    [{**archive_key, 'unit': u, **metrics[u]}
                     for u in set(units)], ignore_extra_fields=True, allow_direct_insert=True)
                ephys.ArchivedClustering.UnitStat.insert(
                    [{**archive_key, 'unit': u, 'unit_amp': unit_amp[i], 'unit_snr': unit_snr[i],
                      'isi_violation': metrics[u]['isi_viol'], 'avg_firing_rate': metrics[u]['firing_rate']}
                     for i, u in enumerate(set(units))], allow_direct_insert=True)

        else:
            # insert Unit
            log.info('.. ephys.Unit')

            with InsertBuffer(ephys.Unit, 10, skip_duplicates=True,
                              allow_direct_insert=True) as ib:

                for i, u in enumerate(set(units)):
                    if method in ['jrclust_v3', 'jrclust_v4']:
                        wf_chn_idx = 0
                    elif method in ['kilosort2']:
                        wf_chn_idx = np.where(data['ks_channel_map'] == vmax_unit_site[i])[0][0]

                    ib.insert1({**skey, **insertion_key,
                                **site2electrode_map[vmax_unit_site[i]],
                                'clustering_method': method,
                                'unit': u,
                                'unit_uid': u,
                                'unit_quality': unit_notes[i],
                                'unit_posx': unit_xpos[i],
                                'unit_posy': unit_ypos[i],
                                'unit_amp': unit_amp[i],
                                'unit_snr': unit_snr[i],
                                'spike_times': unit_spikes[i],
                                'spike_sites': unit_spike_sites[i],
                                'spike_depths': unit_spike_depths[i],
                                'waveform': unit_wav[i][wf_chn_idx]})

                    if ib.flush():
                        log.debug('.... {}'.format(u))

            # insert Unit.UnitTrial
            log.info('.. ephys.Unit.UnitTrial')
            dj.conn().ping()
            with InsertBuffer(ephys.Unit.UnitTrial, 10000, skip_duplicates=True,
                              allow_direct_insert=True) as ib:

                for i, u in enumerate(set(units)):
                    for t in range(len(trials)):
                        if len(unit_trial_spikes[i][t]):
                            ib.insert1({**skey,
                                        'insertion_number': probe,
                                        'clustering_method': method,
                                        'unit': u,
                                        'trial': trials[t]})
                            if ib.flush():
                                log.debug('.... (u: {}, t: {})'.format(u, t))

            # insert TrialSpikes
            log.info('.. ephys.Unit.TrialSpikes')
            dj.conn().ping()
            with InsertBuffer(ephys.Unit.TrialSpikes, 10000, skip_duplicates=True,
                              allow_direct_insert=True) as ib:
                for i, u in enumerate(set(units)):
                    for t in range(len(trials)):
                        ib.insert1({**skey,
                                    'insertion_number': probe,
                                    'clustering_method': method,
                                    'unit': u,
                                    'trial': trials[t],
                                    'spike_times': unit_trial_spikes[i][t]})
                        if ib.flush():
                            log.debug('.... (u: {}, t: {})'.format(u, t))

            if metrics is not None:
                metrics.columns = [c.lower() for c in metrics.columns]  # lower-case col names
                # -- confirm correct attribute names from the PD
                required_columns = np.setdiff1d(ephys.ClusterMetric.heading.names + ephys.WaveformMetric.heading.names,
                                                ephys.Unit.primary_key)
                missing_columns = np.setdiff1d(required_columns, metrics.columns)

                if len(missing_columns) > 0:
                    raise ClusterMetricError('Missing or misnamed column(s) in metrics.csv: {}'.format(missing_columns))

                metrics = dict(metrics.T)

                log.info('.. inserting cluster metrics and waveform metrics')
                dj.conn().ping()
                ephys.ClusterMetric.insert([{**skey, 'insertion_number': probe,
                                             'clustering_method': method, 'unit': u, **metrics[u]}
                                            for u in set(units)],
                                           ignore_extra_fields=True, allow_direct_insert=True)
                ephys.WaveformMetric.insert([{**skey, 'insertion_number': probe,
                                              'clustering_method': method, 'unit': u, **metrics[u]}
                                             for u in set(units)],
                                            ignore_extra_fields=True, allow_direct_insert=True)
                ephys.UnitStat.insert([{**skey, 'insertion_number': probe,
                                        'clustering_method': method, 'unit': u,
                                        'isi_violation': metrics[u]['isi_viol'],
                                        'avg_firing_rate': metrics[u]['firing_rate']} for u in set(units)],
                                      allow_direct_insert=True)

            dj.conn().ping()
            log.info('.. inserting clustering timestamp and label')

            ephys.ClusteringLabel.insert([{**skey, 'insertion_number': probe,
                                           'clustering_method': method, 'unit': u,
                                           'clustering_time': creation_time,
                                           'quality_control': bool('qc' in clustering_label),
                                           'manual_curation': bool('curated' in clustering_label)} for u in set(units)],
                                         allow_direct_insert=True)

            log.info('.. inserting file load information')

            self.insert1(skey, skip_duplicates=True, allow_direct_insert=True)
            self.EphysFile.insert1(
                {**skey, 'probe_insertion_number': probe,
                 'ephys_file': ef_path.relative_to(rigpath).as_posix()}, allow_direct_insert=True)

            log.info('-- ephys ingest for {} - probe {} complete'.format(skey, probe))


def do_ephys_ingest(session_key, replace=False, probe_insertion_exists=False, into_archive=False):
    """
    Perform ephys-ingestion for a particular session (defined by session_key) to either
        + fresh ingest of new probe insertion and clustering results
        + archive existing clustering results and replace with new one (set 'replace=True')
    """
    # =========== Find Ephys Recording ============
    key = (experiment.Session & session_key).fetch1()
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

    if replace:
        if len(ephys.Unit & session_key) == 0:  # sanity check
            raise ValueError('No units exist for this session. Cannot handle "replace=True"')

        probe_insertion_exists = True
        # ============ Inspect new clustering dir(s) ============
        # if all new clustering data has identical timestamps to ingested ones, throw error
        identical_clustering_results = []
        for probe_no, (f, cluster_method, npx_meta) in clustering_files.items():
            cluster_output_dir = f[0] if f[0].is_dir() else f[0].parent
            creation_time, _ = extract_clustering_info(cluster_output_dir, cluster_method)
            existing_clustering_time = (ephys.ClusteringLabel & session_key & {'insertion_number': probe_no}).fetch(
                'clustering_time', limit=1)[0]

            if abs((existing_clustering_time - creation_time).total_seconds()) <= 1:
                identical_clustering_results.append((probe_no, cluster_output_dir))

        if len(identical_clustering_results) == len(ephys.ProbeInsertion & session_key):
            raise IdenticalClusterResultError(identical_clustering_results)

    def do_insert():
        # do the insertion per probe for all probes
        for probe_no, (f, cluster_method, npx_meta) in clustering_files.items():
            insertion_key = {'subject_id': sinfo['subject_id'], 'session': sinfo['session'], 'insertion_number': probe_no}
            if probe_insertion_exists and (ephys.Unit & insertion_key):
                # if probe_insertion exists and there exists also units for this insertion_key, skip over it
                continue
            try:
                log.info('------ Start loading clustering results for probe: {} ------'.format(probe_no))
                loader = cluster_loader_map[cluster_method]
                dj.conn().ping()
                EphysIngest()._load(loader(sinfo, *f), probe_no, npx_meta, rigpath,
                                    probe_insertion_exists=probe_insertion_exists, into_archive=into_archive)
            except (ProbeInsertionError, ClusterMetricError, FileNotFoundError) as e:
                dj.conn().cancel_transaction()  # either successful ingestion of all probes, or none at all
                if isinstance(e, ProbeInsertionError):
                    log.warning('Probe Insertion Error: \n{}. \nSkipping...'.format(str(e)))
                else:
                    log.warning('Error: {}'.format(str(e)))
                return

    # the insert part
    if dj.conn().in_transaction:
        if replace:
            archive_ingested_clustering_results(session_key)
        do_insert()
    else:
        with dj.conn().transaction:
            if replace:
                archive_ingested_clustering_results(session_key)
            do_insert()


def _gen_probe_insert(sinfo, probe, npx_meta, probe_insertion_exists=False):
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

    if probe_insertion_exists:
        if insertion_key not in ephys.ProbeInsertion.proj():
            raise RuntimeError(f'ProbeInsertion key not present. Expecting: {insertion_key}')
    else:
        # add probe insertion
        log.info('.. creating probe insertion')

        lab.Probe.insert1({'probe': part_no, 'probe_type': e_config_key['probe_type']}, skip_duplicates=True)

        ephys.ProbeInsertion.insert1({**insertion_key,  **e_config_key, 'probe': part_no})

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
    ec_hash = dict_value_to_hash({k['electrode']: k for k in eg_members})

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
        sync_behav, sync_ephys, trial_fix, trial_go, trial_start, _ = read_bitcode(fpath.parent, h2o, skey)
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
    ef = h5py.File(str(ef_path), mode='r')  # ephys file

    # -- trial-info from bitcode --
    try:
        sync_behav, sync_ephys, trial_fix, trial_go, trial_start, _ = read_bitcode(fpath.parent, h2o, skey)
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
        'bitcode_raw': bitcode_raw,
    }

    return data


cluster_loader_map = {'jrclust_v3': _load_jrclust_v3,
                      'jrclust_v4': _load_jrclust_v4,
                      'kilosort2': _load_kilosort2}


# ======== Helpers for directory navigation ========
def _get_sess_dir(rigpath, h2o, sess_datetime):
    rigpath = pathlib.Path(rigpath)
    dpath, dglob = None, None
    if (rigpath / h2o / sess_datetime.date().strftime('%Y%m%d')).exists():
        dpath = rigpath / h2o / sess_datetime.date().strftime('%Y%m%d')
        dglob = '[0-9]/{}'  # probe directory pattern
    elif (rigpath / h2o / 'catgt_{}_g0'.format(
            sess_datetime.date().strftime('%Y%m%d'))).exists():
        dpath = rigpath / h2o / 'catgt_{}_g0'.format(
            sess_datetime.date().strftime('%Y%m%d'))
        dglob = '{}_*_imec[0-9]'.format(
            sess_datetime.date().strftime('%Y%m%d')) + '/{}'
    else:
        date_strings = [sess_datetime.date().strftime('%m%d%y'), sess_datetime.date().strftime('%Y%m%d')]
        for date_string in date_strings:
            sess_dirs = list(pathlib.Path(rigpath, h2o).glob('*{}*{}_*'.format(h2o, date_string)))
            for sess_dir in sess_dirs:
                try:
                    npx_meta = NeuropixelsMeta(next(sess_dir.rglob('{}_*.ap.meta'.format(h2o))))
                except StopIteration:
                    continue
                # ensuring time difference between behavior-start and ephys-start is no more than 2 minutes - this is to handle multiple sessions in a day
                start_time_difference = abs((npx_meta.recording_time - sess_datetime).total_seconds())
                if start_time_difference <= 120:
                    dpath = sess_dir
                    dglob = '{}*{}_*_imec[0-9]'.format(h2o, date_string) + '/{}'  # probe directory pattern
                    break
                else:
                    log.info('Found {} - difference in behavior and ephys start-time: {} seconds (more than 2 minutes). Skipping...'.format(sess_dir, start_time_difference))


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
    npx_meta_files = list(dpath.glob(dglob.format('*.ap.meta')))
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
        v3files = [((f, ), 'jrclust_v3') for f in probe_dir.glob(jrclustv3spec)]
        # JRClust v4
        v4files = [((f, ), 'jrclust_v4') for f in probe_dir.glob(jrclustv4spec)]
        # Kilosort
        ks2spec = ks2specs[0] if len(list(probe_dir.rglob(ks2specs[0]))) > 0 else ks2specs[1]
        ks2files = [((f.parent, probe_dir), 'kilosort2') for f in probe_dir.rglob(ks2spec)]

        if len(ks2files) > 1:
            raise ValueError('Multiple Kilosort outputs found at: {}'.format([str(x[0]) for x in ks2files]))

        clustering_results = v4files + v3files + ks2files

        if len(clustering_results) < 1:
            raise FileNotFoundError('Error - No clustering results found at {}'.format(probe_dir))
        elif len(clustering_results) > 1:
            log.warning('Found multiple clustering results at {probe_dir}.'
                        ' Prioritize JRC4 > JRC3 > KS2'.format(probe_dir))

        fp, loader = clustering_results[0]
        clustered_probes[probe_number] = (fp, loader, NeuropixelsMeta(meta_file))

    return clustered_probes


# ======== Helpers for bitcodes loading/generation ========
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
                ephys_bitcodes = behavior_bitcodes
                trial_numbers = behav_trials
                ephys_trial_ref_times = trial_start_times + go_times
                ephys_trial_start_times = trial_start_times
            else:
                raise FileNotFoundError('Generate bitcode failed. No bitcode file'
                                        ' "*.XA_0_0.txt"'
                                        ' for found in {}'.format(bitcode_dir))
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


def insert_ephys_events(skey, bf, trial_trunc=None):
    '''
    all times are session-based
    '''
    
    # --- Events available both from behavior .csv file (trial time) and ephys NIDQ (session time) ---
    # digMarkerPerTrial from bitcode.mat: [STRIG_, GOCUE_, CHOICEL_, CHOICER_, REWARD_, ITI_, BPOD_START_, ZABER_IN_POS_]
    # <--> ephys.TrialEventType: 'bitcodestart', 'go', 'choice', 'choice', 'reward', 'trialend', 'bpodstart', 'zaberinposition'
    log.info('.... insert_ephys_events() ...')
    log.info('       loading ephys events from NIDQ ...')
    df = pd.DataFrame()
    headings = bf['headings'][0]
    digMarkerPerTrial = bf['digMarkerPerTrial']
    
    if trial_trunc is None: trial_trunc = digMarkerPerTrial.shape[0]
    
    for col, event_type in enumerate(headings):
        times = digMarkerPerTrial[:trial_trunc, col]
        not_nan = np.where(~np.isnan(times))[0]
        trials = not_nan + 1   # Trial all starts from 1
        df = df.append(pd.DataFrame({**skey,
                           'trial': trials,
                           'trial_event_id': col,
                           'trial_event_type': event_type[0],
                           'trial_event_time': times[not_nan]}
                                    )
                       )

    # --- Zaber pulses (only available from ephys NIDQ) ---
    if 'zaberPerTrial' in bf:
        for trial, pulses in enumerate(bf['zaberPerTrial'][0][:trial_trunc]):
            df = df.append(pd.DataFrame({**skey,
                               'trial': trial + 1,   # Trial all starts from 1
                               'trial_event_id': np.arange(len(pulses)) + len(headings),
                               'trial_event_type': 'zaberstep',
                               'trial_event_time': pulses.flatten()}
                                        )
                           )

    # --- Do batch insertion --
    ephys.TrialEvent.insert(df, allow_direct_insert=True)

    # --- Licks from NI --
    df_action = pd.DataFrame()
    lick_wrapper = {'left lick': 'lickLPerTrial', 'right lick': 'lickRPerTrial', 'middle lick': 'lickMPerTrial'}
    exist_lick = [ltype for ltype in lick_wrapper.keys() if lick_wrapper[ltype] in bf]

    if len(exist_lick):
        log.info(f'       loading licks from NIDQ ...')
        
        for trial, *licks in enumerate(zip(*(bf[lick_wrapper[ltype]][0][:trial_trunc] for ltype in exist_lick))):
            lick_times = {ltype: ltime for ltype, ltime in zip(exist_lick, *licks)}
            all_lick_types = np.concatenate(
                [[ltype] * len(ltimes) for ltype, ltimes in lick_times.items()])
            all_lick_times = np.concatenate(
                [ltimes for ltimes in lick_times.values()]).flatten()
            sorted_licks = sorted(zip(all_lick_types, all_lick_times), key=lambda x: x[-1])  # sort by lick times
            df_action = df_action.append(pd.DataFrame([{ **skey,
                                                         'trial': trial + 1,  # Trial all starts from 1
                                                         'action_event_id': idx,  # Event_id starts from 0
                                                         'action_event_type': ltype,
                                                         'action_event_time': ltime
                                                         } for idx, (ltype, ltime)
                                                       in enumerate(sorted_licks)
                                                       ]))

        # --- Do batch insertion --
        ephys.ActionEvent.insert(df_action, allow_direct_insert=True)

    # --- Camera frames (only available from ephys NIDQ) ---
    if 'cameraPerTrial' in bf:
        log.info('       loading camera frames from NIDQ ...')

        _idx = [_idx for _idx, field in enumerate(bf['chan'].dtype.descr) if 'cameraNameInDJ' in field][0]
        cameras = bf['chan'][0,0][_idx][0,:]
        for camera, all_frames in zip(cameras, bf['cameraPerTrial'][0]):
            for trial, frames in enumerate(all_frames[0][:trial_trunc]):
                key = {**skey,
                       'trial': trial + 1,  # Trial all starts from 1
                       'tracking_device': camera[0]}
                tracking.Tracking.insert1({**key,
                                           'tracking_samples': len(frames)},
                                          allow_direct_insert=True)
                tracking.Tracking.Frame.insert1({**key,
                                                  'frame_time': frames.flatten()},
                                                 allow_direct_insert=True)
    log.info('.... insert_ephys_events() Done! ...')


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
                d = np.load(f, mmap_mode='r', allow_pickle=False, fix_imports=False)
                self._data[base] = d.squeeze()

        # Read the Cluster Groups
        if (self._dname / 'cluster_groups.csv').exists():
            df = pd.read_csv(self._dname / 'cluster_groups.csv', delimiter='\t')
            self._data['cluster_groups'] = np.array(df['group'].values)
            self._data['cluster_ids'] = np.array(df['cluster_id'].values)
        elif (self._dname / 'cluster_KSLabel.tsv').exists():
            df = pd.read_csv(self._dname / 'cluster_KSLabel.tsv', sep="\t", header=0)
            self._data['cluster_groups'] = np.array(df['KSLabel'].values)
            self._data['cluster_ids'] = np.array(df['cluster_id'].values)
        else:
            raise FileNotFoundError('Neither cluster_groups.csv nor cluster_KSLabel.tsv found!')

    def extract_curated_cluster_notes(self):
        curated_cluster_notes = {}
        for cluster_file in pathlib.Path(self._dname).glob('cluster_*.tsv'):
            if cluster_file.name not in self.ks_cluster_files:
                curation_source = ''.join(cluster_file.stem.split('_')[1:])
                df = pd.read_csv(cluster_file, sep="\t", header=0)
                curated_cluster_notes[curation_source] = dict(
                    cluster_ids=np.array(df['cluster_id'].values),
                    cluster_notes=np.array(df[curation_source].values))
        return curated_cluster_notes

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

    meta = NeuropixelsMeta(meta_fp)
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


# ====== Methods for reprocessing of ephys ingestion ======
def extend_ephys_ingest(session_key):
    """
    Extend ephys-ingestion for a particular session (defined by session_key) to add clustering results for new probe
    """
    #
    # Find Ephys Recording
    #
    key = (experiment.Session & session_key).fetch1()
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

    with dj.conn().transaction:
        for probe_no, (f, cluster_method, npx_meta) in clustering_files.items():
            loader = cluster_loader_map[cluster_method]
            insertion_key = {'subject_id': sinfo['subject_id'],
                             'session': sinfo['session'],
                             'insertion_number': probe_no}
            if insertion_key in ephys.ProbeInsertion.proj():
                log.info('Probe {} exists, skipping...'.format(probe_no))
                continue
            try:
                EphysIngest()._load(loader(sinfo, *f), probe_no, npx_meta, rigpath)
            except (ProbeInsertionError, FileNotFoundError) as e:
                dj.conn().cancel_transaction()  # either successful ingestion of all probes, or none at all
                if isinstance(e, ProbeInsertionError):
                    log.warning('Probe Insertion Error: \n{}. \nSkipping...'.format(str(e)))
                else:
                    log.warning('Error: {}'.format(str(e)))
                return


def archive_ingested_clustering_results(key):
    """
    The input-argument "key" should be at the level of ProbeInsertion or its anscestor.

    1. Copy to ephys.ArchivedUnit
    2. Delete ephys.Unit
    """
    insertion_keys = (ephys.ProbeInsertion & key).fetch('KEY')
    log.info('Archiving {} probe insertion(s): {}'.format(len(insertion_keys), insertion_keys))

    archival_time = datetime.now()

    q_archived_clusterings, q_ephys_files, q_archived_units, \
    q_archived_units_stat, q_archived_cluster_metrics,\
    q_archived_waveform_metrics = [], [], [], [], [], []

    for insert_key in insertion_keys:
        q_archived_clustering = (ephys.ProbeInsertion.proj() & insert_key).aggr(
            ephys.ClusteringLabel * ephys.ClusteringMethod, ...,
            clustering_method='clustering_method', clustering_time='clustering_time',
            quality_control='quality_control', manual_curation='manual_curation',
            clustering_note='clustering_note', archival_time='cast("{}" as datetime)'.format(archival_time))

        q_files = (EphysIngest.EphysFile.proj(insertion_number='probe_insertion_number') & insert_key)

        q_units = (ephys.Unit & insert_key).aggr(ephys.UnitCellType, ..., cell_type='cell_type', keep_all_rows=True)

        q_units_stat = q_units.proj('unit_amp', 'unit_snr').aggr(ephys.UnitStat, ...,
                                                                 isi_violation='isi_violation',
                                                                 avg_firing_rate='avg_firing_rate', keep_all_rows=True)
        q_units_cluster_metrics = q_units.proj() * ephys.ClusterMetric
        q_units_waveform_metrics = q_units.proj() * ephys.WaveformMetric

        q_archived_clusterings.append(q_archived_clustering)
        q_ephys_files.append(q_archived_clustering * q_files)
        q_archived_units.append(q_archived_clustering * q_units)
        q_archived_units_stat.append(q_archived_clustering * q_units_stat)
        q_archived_cluster_metrics.append(q_archived_clustering * q_units_cluster_metrics)
        q_archived_waveform_metrics.append(q_archived_clustering * q_units_waveform_metrics)

    # skip archiving (only do delete) if this set of results has already been archived
    is_archived = np.all([bool(archived_key in ephys.ArchivedClustering.proj())
                          for archived_key in q_archived_clusterings])

    if is_archived:
        log.info('This set of clustering results has already been archived, skip archiving...')
    else:
        # preparing spike_times and trial_spike
        tr_no, tr_start = (experiment.SessionTrial & key).fetch(
            'trial', 'start_time', order_by='trial')
        tr_stop = np.append(tr_start[1:], np.inf)

        # units
        archived_units = []
        for units in q_archived_units:
            # recompute trial_spike
            log.info('Archiving {} units'.format(len(units)))
            units = units.fetch(as_dict=True)
            for unit in tqdm(units):
                after_start = unit['spike_times'] >= tr_start[:, None]
                before_stop = unit['spike_times'] <= tr_stop[:, None]
                in_trial = ((after_start & before_stop) * tr_no[:, None]).sum(axis=0)
                unit['trial_spike'] = np.where(in_trial == 0, np.nan, in_trial).astype(int)
            # for unit in tqdm(units):
            #     trial_spike = np.full_like(unit['spike_times'], np.nan)
            #     for tr, tstart, tstop in zip(tr_no, tr_start, tr_stop):
            #         trial_idx = np.where((unit['spike_times'] >= tstart) & (unit['spike_times'] <= tstop))
            #         trial_spike[trial_idx] = tr
            #     unit['trial_spike'] = trial_spike
            archived_units.extend(units)

    def copy_and_delete():

        if not is_archived:
            # server-side copy
            log.info('Archiving {} units from {} probe insertions'.format(len(ephys.Unit & key),
                                                                          len(ephys.ProbeInsertion & key)))
            insert_settings = dict(ignore_extra_fields=True, allow_direct_insert=True)

            [ephys.ArchivedClustering.insert(clustering, **insert_settings)
             for clustering in q_archived_clusterings]
            [ephys.ArchivedClustering.EphysFile.insert(ephys_files, **insert_settings)
             for ephys_files in q_ephys_files]
            ephys.ArchivedClustering.Unit.insert(archived_units, **insert_settings)
            [ephys.ArchivedClustering.UnitStat.insert(units_stat, **insert_settings)
             for units_stat in q_archived_units_stat]
            [ephys.ArchivedClustering.ClusterMetric.insert(cluster_metrics, **insert_settings)
             for cluster_metrics in q_archived_cluster_metrics]
            [ephys.ArchivedClustering.WaveformMetric.insert(waveform_metrics, **insert_settings)
             for waveform_metrics in q_archived_waveform_metrics]

        with dj.config(safemode=False):
            log.info('Delete clustering data and associated analysis results')
            (ephys.Unit & key).delete()
            (EphysIngest.EphysFile & key).delete(force=True)
            (report.SessionLevelCDReport & key).delete()
            (report.ProbeLevelPhotostimEffectReport & key).delete()
            (report.ProbeLevelReport & key).delete()
            (report.ProbeLevelDriftMap & key).delete()

    # the copy, delete part
    if dj.conn().in_transaction:
        copy_and_delete()
    else:
        with dj.conn().transaction:
            copy_and_delete()
