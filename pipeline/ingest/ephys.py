#! /usr/bin/env python

import os
import logging
import pathlib
from datetime import datetime
from tqdm import tqdm
import re
from itertools import repeat
import pandas as pd
import numpy as np
import datajoint as dj

from pipeline import InsertBuffer

from .. import get_schema_name, create_schema_settings
from .. import lab, experiment, ephys, tracking, report
from . import ProbeInsertionError, ClusterMetricError, IdenticalClusterResultError
from .utils.spike_sorter_loader import cluster_loader_map, npx_bit_volts, extract_clustering_info
from .utils.paths import get_sess_dir, gen_probe_insert, match_probe_to_ephys

schema = dj.schema(get_schema_name('ingest_ephys'), **create_schema_settings)

os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
logger.setLevel('INFO')
logger.addHandler(stream_handler)


@schema
class EphysIngest(dj.Imported):
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

        logger.info('\n======================================================')
        logger.info('EphysIngest().make(): key: {k}'.format(k=key))

        self.insert1(key)

        ephys.ProbeInsertion.generate_entries(key)

        for insertion_key in (ephys.ProbeInsertion & key).fetch('KEY'):
            ephys.Unit().make(insertion_key)
            ephys.ClusterMetric().make(insertion_key)

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
        cluster_noise_label = data.get('cluster_noise_label')
        bitcode_raw = data['bitcode_raw']

        logger.info('-- Start insertions for probe: {} - Clustering method: {} - Label: {}'.format(probe, method, clustering_label))

        assert len(trial_start) == len(trial_go), 'Unequal number of bitcode "trial_start" ({}) and "trial_go" ({})'.format(len(trial_start), len(trial_go))

        # create probe insertion records
        if into_archive:
            probe_insertion_exists = True

        try:
            insertion_key, e_config_key = gen_probe_insert(sinfo, probe, npx_meta, probe_insertion_exists=probe_insertion_exists)
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
                logger.info('ephys/bitcode trial mismatch - fix using "trialNum"')
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

        # units
        unit_set = set(units)

        # build spike arrays
        unit_spikes = np.array([spikes[np.where(units == u)] for u in unit_set]) - trial_start[0]

        unit_trial_spikes = np.array(
            [[trial_spikes[t][np.where(trial_units[t] == u)]
              for t in range(len(trials))] for u in unit_set])

        q_electrodes = lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode & e_config_key
        site2electrode_map = {}
        for recorded_site, (shank, shank_col, shank_row, _) in enumerate(npx_meta.shankmap['data']):
            site2electrode_map[recorded_site + 1] = (q_electrodes
                                                     & {'shank': shank + 1,  # this is a 1-indexed pipeline
                                                        'shank_col': shank_col + 1,
                                                        'shank_row': shank_row + 1}).fetch1('KEY')

        spike_sites = np.array([site2electrode_map[s]['electrode'] for s in spike_sites])
        unit_spike_sites = np.array([spike_sites[np.where(units == u)] for u in unit_set])
        unit_spike_depths = np.array([spike_depths[np.where(units == u)] for u in unit_set])

        if into_archive:
            logger.info('.. inserting clustering timestamp and label')
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

            unit_spike_trial_num = np.array([spike_trial_num[np.where(units == u)] for u in unit_set])

            with InsertBuffer(ephys.ArchivedClustering.Unit, 10, skip_duplicates=True,
                              allow_direct_insert=True) as ib:

                for i, u in enumerate(unit_set):
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
                        logger.debug('.... {}'.format(u))

            if metrics is not None:
                metrics.columns = [c.lower() for c in metrics.columns]  # lower-case col names
                # -- confirm correct attribute names from the PD
                required_columns = np.setdiff1d(ephys.ClusterMetric.heading.names + ephys.WaveformMetric.heading.names,
                                                ephys.Unit.primary_key)
                missing_columns = np.setdiff1d(required_columns, metrics.columns)

                if len(missing_columns) > 0:
                    raise ClusterMetricError('Missing or misnamed column(s) in metrics.csv: {}'.format(missing_columns))

                metrics = dict(metrics.T)

                logger.info('.. inserting cluster metrics and waveform metrics')
                dj.conn().ping()
                ephys.ArchivedClustering.ClusterMetric.insert(
                    [{**archive_key, 'unit': u, **metrics[u]}
                     for u in unit_set], ignore_extra_fields=True, allow_direct_insert=True)
                ephys.ArchivedClustering.WaveformMetric.insert(
                    [{**archive_key, 'unit': u, **metrics[u]}
                     for u in unit_set], ignore_extra_fields=True, allow_direct_insert=True)
                ephys.ArchivedClustering.UnitStat.insert(
                    [{**archive_key, 'unit': u, 'unit_amp': unit_amp[i], 'unit_snr': unit_snr[i],
                      'isi_violation': metrics[u]['isi_viol'], 'avg_firing_rate': metrics[u]['firing_rate']}
                     for i, u in enumerate(unit_set)], allow_direct_insert=True)
        else:
            # insert Unit
            logger.info('.. ephys.Unit')

            with InsertBuffer(ephys.Unit, 10, skip_duplicates=True,
                              allow_direct_insert=True) as ib:

                for i, u in enumerate(unit_set):
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
                        logger.debug('.... {}'.format(u))

            # insert Unit.UnitTrial
            logger.info('.. ephys.Unit.UnitTrial')
            dj.conn().ping()
            with InsertBuffer(ephys.Unit.UnitTrial, 10000, skip_duplicates=True,
                              allow_direct_insert=True) as ib:

                for i, u in enumerate(unit_set):
                    for t in range(len(trials)):
                        if len(unit_trial_spikes[i][t]):
                            ib.insert1({**skey,
                                        'insertion_number': probe,
                                        'clustering_method': method,
                                        'unit': u,
                                        'trial': trials[t]})
                            if ib.flush():
                                logger.debug('.... (u: {}, t: {})'.format(u, t))

            # insert TrialSpikes
            logger.info('.. ephys.Unit.TrialSpikes')
            dj.conn().ping()
            with InsertBuffer(ephys.Unit.TrialSpikes, 10000, skip_duplicates=True,
                              allow_direct_insert=True) as ib:
                for i, u in enumerate(unit_set):
                    for t in range(len(trials)):
                        ib.insert1({**skey,
                                    'insertion_number': probe,
                                    'clustering_method': method,
                                    'unit': u,
                                    'trial': trials[t],
                                    'spike_times': unit_trial_spikes[i][t]})
                        if ib.flush():
                            logger.debug('.... (u: {}, t: {})'.format(u, t))

            if metrics is not None:
                metrics.columns = [c.lower() for c in metrics.columns]  # lower-case col names
                # -- confirm correct attribute names from the PD
                required_columns = np.setdiff1d(ephys.ClusterMetric.heading.names + ephys.WaveformMetric.heading.names,
                                                ephys.Unit.primary_key)
                missing_columns = np.setdiff1d(required_columns, metrics.columns)

                if len(missing_columns) > 0:
                    raise ClusterMetricError('Missing or misnamed column(s) in metrics.csv: {}'.format(missing_columns))

                metrics = dict(metrics.T)

                logger.info('.. inserting cluster metrics and waveform metrics')
                dj.conn().ping()
                ephys.ClusterMetric.insert([{**skey, 'insertion_number': probe,
                                             'clustering_method': method, 'unit': u, **metrics[u]}
                                            for u in unit_set],
                                           ignore_extra_fields=True, allow_direct_insert=True)
                ephys.WaveformMetric.insert([{**skey, 'insertion_number': probe,
                                              'clustering_method': method, 'unit': u, **metrics[u]}
                                             for u in unit_set],
                                            ignore_extra_fields=True, allow_direct_insert=True)
                ephys.UnitStat.insert([{**skey, 'insertion_number': probe,
                                        'clustering_method': method, 'unit': u,
                                        'isi_violation': metrics[u]['isi_viol'],
                                        'avg_firing_rate': metrics[u]['firing_rate']}
                                       for u in unit_set],
                                      allow_direct_insert=True)

            if cluster_noise_label is not None:
                dj.conn().ping()
                logger.info('.. inserting unit noise label')
                ephys.UnitNoiseLabel.insert((
                    {**skey, 'insertion_number': probe, 'clustering_method': method,
                     'unit': u, 'noise': note}
                    for u, note in zip(cluster_noise_label['cluster_ids'],
                                       cluster_noise_label['cluster_notes']) if u in unit_set),
                    allow_direct_insert=True)

            dj.conn().ping()
            logger.info('.. inserting clustering timestamp and label')

            ephys.ClusteringLabel.insert([{**skey, 'insertion_number': probe,
                                           'clustering_method': method, 'unit': u,
                                           'clustering_time': creation_time,
                                           'quality_control': bool('qc' in clustering_label),
                                           'manual_curation': bool('curated' in clustering_label)}
                                          for u in unit_set],
                                         allow_direct_insert=True)

            logger.info('.. inserting file load information')

            self.insert1(skey, skip_duplicates=True, allow_direct_insert=True)
            self.EphysFile.insert1(
                {**skey, 'probe_insertion_number': probe,
                 'ephys_file': ef_path.relative_to(rigpath).as_posix()}, allow_direct_insert=True)

            logger.info('-- ephys ingest for {} - probe {} complete'.format(skey, probe))


def ingest_units(insertion_key, data, npx_meta):
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
    creation_time = data['creation_time']
    clustering_label = data['clustering_label']
    cluster_noise_label = data.get('cluster_noise_label')
    bitcode_raw = data['bitcode_raw']

    probe_no = insertion_key['insertion_number']

    logger.info('-- Start insertions for probe: {} - Clustering method: {} - Label: {}'.format(
        probe_no, method, clustering_label))

    assert len(trial_start) == len(trial_go), 'Unequal number of bitcode "trial_start" ({}) and "trial_go" ({})'.format(
        len(trial_start), len(trial_go))

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
    sync_behav_range = sync_behav[sync_behav_start:][:len(sync_ephys)]  # Note that this will not generate error even if len(sync_ephys) > len(behavior)
    shared_trial_num = len(sync_behav_range)
    sync_ephys_range = sync_ephys[:shared_trial_num]  # Now they must have the same length

    if not np.all(np.equal(sync_ephys_range, sync_behav_range)):
        if trial_fix is not None:
            logger.info('ephys/bitcode trial mismatch - fix using "trialNum"')
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
        behav_trials = (experiment.SessionTrial & insertion_key).fetch('trial', order_by='trial')
        trials = behav_trials[trial_indices]

        # TODO: this is a workaround to deal with the case where ephys stops later than behavior
        # but with the assumption that ephys will NEVER start earlier than behavior
        trial_start = trial_start[:shared_trial_num]  # Truncate ephys 'trial_start' at the tail
        # And also truncate the ingestion of digital markers (see immediate below)

    assert len(trial_start) == len(trials),\
        'Unequal number of bitcode "trial_start" ({}) and ingested behavior trials ({})'.format(len(trial_start), len(trials))

    # -- Ingest time markers from NIDQ channels --
    # This is redudant for delay response task because aligning spikes to the go-cue is enough (trial_spikes below)
    # But this is critical for the foraging task, because we need global session-wise times to plot flexibly-aligned PSTHs (in particular, spikes during ITI).
    # However, we CANNOT get this from behavior pybpod .csv files (PC-TIME is inaccurate, whereas BPOD-TIME is trial-based)
    if probe_no == 1 and 'digMarkerPerTrial' in bitcode_raw:  # Only import once for one session
        insert_ephys_events(skey, bitcode_raw, shared_trial_num)

    # trialize the spikes & subtract go cue
    t, trial_spikes, trial_units = 0, [], []

    spike_trial_num = np.full_like(spikes, np.nan)

    while t < len(trial_start) - 1:
        s0, s1 = trial_start[t], trial_start[t + 1]

        trial_idx = np.where((spikes > s0) & (spikes < s1))
        spike_trial_num[trial_idx] = trials[t]  # Assign (behavioral) trial number to each spike

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

    # units
    unit_set = set(units)

    # build spike arrays
    unit_spikes = np.array([spikes[np.where(units == u)] for u in unit_set]) - trial_start[0]

    unit_trial_spikes = np.array(
        [[trial_spikes[t][np.where(trial_units[t] == u)]
          for t in range(len(trials))] for u in unit_set])

    e_config_key = (lab.ElectrodeConfig & (ephys.ProbeInsertion & insertion_key)).fetch1('KEY')
    q_electrodes = lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode & e_config_key
    site2electrode_map = {}
    for recorded_site, (shank, shank_col, shank_row, _) in enumerate(npx_meta.shankmap['data']):
        site2electrode_map[recorded_site + 1] = (q_electrodes
                                                 & {'shank': shank + 1,  # this is a 1-indexed pipeline
                                                    'shank_col': shank_col + 1,
                                                    'shank_row': shank_row + 1}).fetch1('KEY')

    spike_sites = np.array([site2electrode_map[s]['electrode'] for s in spike_sites])
    unit_spike_sites = np.array([spike_sites[np.where(units == u)] for u in unit_set])
    unit_spike_depths = np.array([spike_depths[np.where(units == u)] for u in unit_set])

    # insert Unit
    logger.info('.. ephys.Unit')

    with InsertBuffer(ephys.Unit, 10, skip_duplicates=True,
                      allow_direct_insert=True) as ib:

        for i, u in enumerate(unit_set):
            if method in ['jrclust_v3', 'jrclust_v4']:
                wf_chn_idx = 0
            elif method in ['kilosort2']:
                wf_chn_idx = np.where(data['ks_channel_map'] == vmax_unit_site[i])[0][0]

            ib.insert1({**insertion_key,
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
                logger.debug('.... {}'.format(u))

    # insert Unit.UnitTrial
    logger.info('.. ephys.Unit.UnitTrial')
    dj.conn().ping()
    with InsertBuffer(ephys.Unit.UnitTrial, 10000, skip_duplicates=True,
                      allow_direct_insert=True) as ib:

        for i, u in enumerate(unit_set):
            for t in range(len(trials)):
                if len(unit_trial_spikes[i][t]):
                    ib.insert1({**insertion_key,
                                'clustering_method': method,
                                'unit': u,
                                'trial': trials[t]})
                    if ib.flush():
                        logger.debug('.... (u: {}, t: {})'.format(u, t))

    # insert TrialSpikes
    logger.info('.. ephys.Unit.TrialSpikes')
    dj.conn().ping()
    with InsertBuffer(ephys.Unit.TrialSpikes, 10000, skip_duplicates=True,
                      allow_direct_insert=True) as ib:
        for i, u in enumerate(unit_set):
            for t in range(len(trials)):
                ib.insert1({**insertion_key,
                            'clustering_method': method,
                            'unit': u,
                            'trial': trials[t],
                            'spike_times': unit_trial_spikes[i][t]})
                if ib.flush():
                    logger.debug('.... (u: {}, t: {})'.format(u, t))

    if cluster_noise_label is not None and cluster_noise_label:
        dj.conn().ping()
        logger.info('.. inserting unit noise label')
        ephys.UnitNoiseLabel.insert((
            {**insertion_key, 'clustering_method': method,
             'unit': u, 'noise': note}
            for u, note in zip(cluster_noise_label['cluster_ids'],
                               cluster_noise_label['cluster_notes']) if u in unit_set),
            allow_direct_insert=True)

    dj.conn().ping()
    logger.info('.. inserting clustering timestamp and label')

    ephys.ClusteringLabel.insert([{**insertion_key,
                                   'clustering_method': method, 'unit': u,
                                   'clustering_time': creation_time,
                                   'quality_control': bool('qc' in clustering_label),
                                   'manual_curation': bool('curated' in clustering_label)}
                                  for u in unit_set],
                                 allow_direct_insert=True)

    logger.info('.. inserting file load information')

    EphysIngest.EphysFile.insert1(
        {**insertion_key, 'probe_insertion_number': insertion_key['insertion_number'],
         'ephys_file': data['ef_path'].relative_to(data['rigpath']).as_posix()},
        allow_direct_insert=True, ignore_extra_fields=True)


def ingest_metrics(insertion_key, data):
    method = data['method']
    metrics = data['metrics']  # either None or a pd.DataFrame loaded from 'metrics.csv'

    unit_set = (ephys.Unit & insertion_key & {'clustering_method': method}).fetch('unit', order_by='unit')

    if metrics is not None:
        metrics.columns = [c.lower() for c in metrics.columns]  # lower-case col names
        # -- confirm correct attribute names from the PD
        required_columns = np.setdiff1d(ephys.ClusterMetric.heading.names + ephys.WaveformMetric.heading.names,
                                        ephys.Unit.primary_key)
        missing_columns = np.setdiff1d(required_columns, metrics.columns)

        if len(missing_columns) > 0:
            raise ClusterMetricError('Missing or misnamed column(s) in metrics.csv: {}'.format(missing_columns))

        metrics = dict(metrics.T)

        logger.info('.. inserting cluster metrics and waveform metrics')
        dj.conn().ping()
        ephys.ClusterMetric.insert([{**insertion_key,
                                     'clustering_method': method, 'unit': u, **metrics[u]}
                                    for u in unit_set],
                                   ignore_extra_fields=True, allow_direct_insert=True)
        ephys.WaveformMetric.insert([{**insertion_key,
                                      'clustering_method': method, 'unit': u, **metrics[u]}
                                     for u in unit_set],
                                    ignore_extra_fields=True, allow_direct_insert=True)
        ephys.UnitStat.insert([{**insertion_key,
                                'clustering_method': method, 'unit': u,
                                'isi_violation': metrics[u]['isi_viol'],
                                'avg_firing_rate': metrics[u]['firing_rate']}
                               for u in unit_set],
                              allow_direct_insert=True)


def insert_ephys_events(skey, bf, trial_trunc=None):
    '''
    all times are session-based
    '''

    # --- Events available both from behavior .csv file (trial time) and ephys NIDQ (session time) ---
    # digMarkerPerTrial from bitcode.mat: [STRIG_, GOCUE_, CHOICEL_, CHOICER_, REWARD_, ITI_, BPOD_START_, ZABER_IN_POS_]
    # <--> ephys.TrialEventType: 'bitcodestart', 'go', 'choice', 'choice', 'reward', 'trialend', 'bpodstart', 'zaberinposition'
    logger.info('.... insert_ephys_events() ...')
    logger.info('       loading ephys events from NIDQ ...')
    df = pd.DataFrame()
    headings = bf['headings'][0]
    digMarkerPerTrial = bf['digMarkerPerTrial']

    if trial_trunc is None: trial_trunc = digMarkerPerTrial.shape[0]

    for col, event_type in enumerate(headings):
        times = digMarkerPerTrial[:trial_trunc, col]
        not_nan = np.where(~np.isnan(times))[0]
        trials = not_nan + 1  # Trial all starts from 1
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
                                         'trial': trial + 1,  # Trial all starts from 1
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
        logger.info(f'       loading licks from NIDQ ...')

        for trial, *licks in enumerate(zip(*(bf[lick_wrapper[ltype]][0][:trial_trunc] for ltype in exist_lick))):
            lick_times = {ltype: ltime for ltype, ltime in zip(exist_lick, *licks)}
            all_lick_types = np.concatenate(
                [[ltype] * len(ltimes) for ltype, ltimes in lick_times.items()])
            all_lick_times = np.concatenate(
                [ltimes for ltimes in lick_times.values()]).flatten()
            sorted_licks = sorted(zip(all_lick_types, all_lick_times), key=lambda x: x[-1])  # sort by lick times
            df_action = df_action.append(pd.DataFrame([{**skey,
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
        logger.info('       loading camera frames from NIDQ ...')

        _idx = [_idx for _idx, field in enumerate(bf['chan'].dtype.descr) if 'cameraNameInDJ' in field][0]
        cameras = bf['chan'][0, 0][_idx][0, :]
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
    logger.info('.... insert_ephys_events() Done! ...')


# ====== Methods for ephys ingestion ======

def do_ephys_ingest(session_key, replace=False, probe_insertion_exists=False, into_archive=False):
    """
    Perform ephys-ingestion for a particular session (defined by session_key) to either
        + fresh ingest of new probe insertion and clustering results
        + archive existing clustering results and replace with new one (set 'replace=True')
    """
    # =========== Find Ephys Recording ============
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
        logger.warning(str(e) + '. Skipping...')
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
                logger.info('------ Start loading clustering results for probe: {} ------'.format(probe_no))
                loader = cluster_loader_map[cluster_method]
                dj.conn().ping()
                EphysIngest()._load(loader(sinfo, *f), probe_no, npx_meta, rigpath,
                                    probe_insertion_exists=probe_insertion_exists, into_archive=into_archive)
            except (ProbeInsertionError, ClusterMetricError, FileNotFoundError) as e:
                dj.conn().cancel_transaction()  # either successful ingestion of all probes, or none at all
                if isinstance(e, ProbeInsertionError):
                    logger.warning('Probe Insertion Error: \n{}. \nSkipping...'.format(str(e)))
                else:
                    logger.warning('Error: {}'.format(str(e)))
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


def extend_ephys_ingest(session_key):
    """
    Extend ephys-ingestion for a particular session (defined by session_key) to add clustering results for new probe
    """
    ephys.ProbeInsertion.generate_entries(session_key)
    ephys.Unit.populate(session_key)
    ephys.ClusterMetric.populate(session_key)


def archive_ingested_clustering_results(key, archive_trial_spike=False):
    """
    The input-argument "key" should be at the level of ProbeInsertion or its anscestor.

    1. Copy to ephys.ArchivedUnit
    2. Delete ephys.Unit
    """
    assert dj.__version__ >= "0.13.5", f'Archiving clustering results requires DataJoint 0.13.5 and above - you are using {dj.__version__}'

    insertion_keys = (ephys.ProbeInsertion & key).fetch('KEY')
    logger.info('Archiving {} probe insertion(s): {}'.format(len(insertion_keys), insertion_keys))

    archival_time = datetime.now()

    q_archived_clusterings, q_ephys_files, q_archived_units, \
    q_archived_units_stat, q_archived_cluster_metrics,\
    q_archived_waveform_metrics = [], [], [], [], [], []

    for insert_key in insertion_keys:
        logger.info(f'\tChecking insertion: {insert_key}')
        q_archived_clustering = (ephys.ProbeInsertion.proj() & insert_key).aggr(
            ephys.ClusteringLabel * ephys.ClusteringMethod, ...,
            clustering_method='max(clustering_method)', clustering_time='max(clustering_time)',
            quality_control='max(quality_control)', manual_curation='max(manual_curation)',
            clustering_note='max(clustering_note)', archival_time='cast("{}" as datetime)'.format(archival_time))

        q_files = (EphysIngest.EphysFile.proj(insertion_number='probe_insertion_number') & insert_key)

        q_units = (ephys.Unit & insert_key).join(ephys.UnitCellType, left=True)

        q_units_stat = q_units.proj('unit_amp', 'unit_snr').join(ephys.UnitStat, left=True)
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
        logger.info('This set of clustering results has already been archived, skip archiving...')
    else:
        if archive_trial_spike:
            # preparing spike_times and trial_spike
            tr_no, tr_start = (experiment.SessionTrial & key).fetch(
                'trial', 'start_time', order_by='trial')
            tr_stop = np.append(tr_start[1:], np.inf)

        # units
        archived_units = []
        for q_units in q_archived_units:
            # recompute trial_spike
            logger.info('\tArchiving {} units'.format(len(q_units)))
            units = q_units.fetch(as_dict=True)
            if archive_trial_spike:
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
            logger.info('Archiving {} units from {} probe insertions'.format(len(ephys.Unit & key),
                                                                             len(ephys.ProbeInsertion & key)))
            insert_settings = dict(ignore_extra_fields=True, allow_direct_insert=True)

            logger.info('\tArchivedClustering...')
            [ephys.ArchivedClustering.insert(clustering, **insert_settings)
             for clustering in q_archived_clusterings]
            [ephys.ArchivedClustering.EphysFile.insert(ephys_files, **insert_settings)
             for ephys_files in q_ephys_files]

            logger.info('\tArchivedClustering.Unit...')
            ephys.ArchivedClustering.Unit.insert(archived_units, **insert_settings)

            logger.info('\tArchivedClustering.UnitStat...')
            [ephys.ArchivedClustering.UnitStat.insert(units_stat, **insert_settings)
             for units_stat in q_archived_units_stat]

            logger.info('\tArchivedClustering.ClusterMetric...')
            [ephys.ArchivedClustering.ClusterMetric.insert(cluster_metrics, **insert_settings)
             for cluster_metrics in q_archived_cluster_metrics]

            logger.info('\tArchivedClustering.WaveformMetric...')
            [ephys.ArchivedClustering.WaveformMetric.insert(waveform_metrics, **insert_settings)
             for waveform_metrics in q_archived_waveform_metrics]

        with dj.config(safemode=False):
            logger.info('Delete clustering data and associated analysis results')
            logger.info('\tephys.Unit...')
            (ephys.Unit & key).delete()
            (EphysIngest.EphysFile & key).delete(force=True)
            logger.info('\treport.SessionLevelCDReport...')
            (report.SessionLevelCDReport & key).delete()
            logger.info('\treport.ProbeLevelPhotostimEffectReport...')
            (report.ProbeLevelPhotostimEffectReport & key).delete()
            logger.info('\treport.ProbeLevelReport...')
            (report.ProbeLevelReport & key).delete()
            logger.info('\treport.ProbeLevelDriftMap...')
            (report.ProbeLevelDriftMap & key).delete()

    # the copy, delete part
    if dj.conn().in_transaction:
        copy_and_delete()
    else:
        with dj.conn().transaction:
            copy_and_delete()
