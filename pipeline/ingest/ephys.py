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

from pipeline import lab
from pipeline import experiment
from pipeline import ephys
from pipeline import InsertBuffer, dict_to_hash
from pipeline.ingest import behavior as behavior_ingest
from .. import get_schema_name

schema = dj.schema(get_schema_name('ingest_ephys'))

log = logging.getLogger(__name__)

npx_bit_volts = {'neuropixels 1.0': 2.34375, 'neuropixels 2.0': 0.763}  # uV per bit scaling factor for neuropixels probes


@schema
class EphysDataPath(dj.Lookup):
    # ephys data storage location(s)
    definition = """
    data_path:              varchar(255)                # rig data path
    ---
    search_order:           int                         # rig search order
    """

    @property
    def contents(self):
        if 'ephys_data_paths' in dj.config['custom']:  # for local testing
            return dj.config['custom']['ephys_data_paths']

        return ((r'H:\\data\MAP', 0),)


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

        log.info('EphysIngest().make(): key: {k}'.format(k=key))

        #
        # Find Ephys Recording
        #
        key = (experiment.Session & key).fetch1()
        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session.proj(..., '-session_time')) & key).fetch1()

        rigpath = EphysDataPath().fetch1('data_path')
        h2o = sinfo['water_restriction_number']

        sess_time = (datetime.min + key['session_time']).time()
        sess_datetime = datetime.combine(key['session_date'], sess_time)
        dpath, dglob = self._get_sess_dir(rigpath, h2o, sess_datetime)

        if dpath is not None:
            log.info('Found session folder: {}'.format(dpath))
        else:
            log.warning('Error - No session folder found for {}/{}'.format(h2o, key['session_date']))

        try:
            clustering_files = self._match_probe_to_ephys(h2o, dpath, dglob)
        except FileNotFoundError as e:
            log.warning(str(e) + '. Skipping...')
            return

        for probe_no, (f, loader, npx_meta) in clustering_files.items():
            self._load(loader(sinfo, f), probe_no, npx_meta, rigpath)

    def _load(self, data, probe, npx_meta, rigpath):

        sinfo = data['sinfo']
        ef_path = data['ef_path']
        skey = data['skey']
        method = data['method']
        hz = data['hz'] if data['hz'] else npx_meta.meta['imSampRate']
        spikes = data['spikes']
        spike_sites = data['spike_sites']
        units = data['units']
        unit_wav = data['unit_wav']
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

        log.info('Starting insertions for probe: {} - Clustering method: {}'.format(probe, method))

        # account for the buffer period before trial_start
        buffer_sample_count = np.round(npx_meta.meta['trgTTLMarginS'] * npx_meta.meta['imSampRate']).astype(int)
        trial_start = trial_start - np.round(buffer_sample_count).astype(int)

        # remove noise clusters
        if method in ['jrclust_v3', 'jrclust_v4']:
            units, spikes, spike_sites = (v[i] for v, i in zip(
                (units, spikes, spike_sites), repeat((units > 0))))

        # scale amplitudes by uV/bit scaling factor (for kilosort2)
        if method in ['kilosort2']:
            bit_volts = npx_bit_volts[re.match('neuropixels (\d.0)', npx_meta.probe_model).group()]
            unit_amp = unit_amp * bit_volts

        # Determine trial (re)numbering for ephys:
        #
        # - if ephys & bitcode match, determine ephys-to-behavior trial shift
        #   when needed for different-length recordings
        # - otherwise, use trial number correction array (bf['trialNum'])

        sync_behav_start = np.where(sync_behav == sync_ephys[0])[0][0]
        sync_behav_range = sync_behav[sync_behav_start:][:len(sync_ephys)]

        if not np.all(np.equal(sync_ephys, sync_behav_range)):
            if trial_fix is not None:
                log.info('ephys/bitcode trial mismatch - fix using "trialNum"')
                trials = trial_fix[0]
            else:
                raise Exception('Bitcode Mismatch - Fix with "trialNum" not available')
        else:
            if len(sync_behav) < len(sync_ephys):
                start_behav = np.where(sync_behav[0] == sync_ephys)[0][0]
            elif len(sync_behav) > len(sync_ephys):
                start_behav = - np.where(sync_ephys[0] == sync_behav)[0][0]
            else:
                start_behav = 0
            trial_indices = np.arange(len(sync_behav_range)) - start_behav

            # mapping to the behav-trial numbering
            # "trials" here is just the 0-based indices of the behavioral trials
            behav_trials = (experiment.SessionTrial & skey).fetch('trial', order_by='trial')
            trials = behav_trials[trial_indices]

        # trialize the spikes & subtract go cue
        t, trial_spikes, trial_units = 0, [], []

        while t < len(trial_start) - 1:

            s0, s1 = trial_start[t], trial_start[t+1]

            trial_idx = np.where((spikes > s0) & (spikes < s1))

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
        unit_spikes = np.array([spikes[np.where(units == u)]
                                for u in set(units)]) - trial_start[0]

        unit_trial_spikes = np.array(
            [[trial_spikes[t][np.where(trial_units[t] == u)]
              for t in range(len(trials))] for u in set(units)])

        # create probe insertion records
        insertion_key = self._gen_probe_insert(sinfo, probe, npx_meta)

        electrode_keys = {c['electrode']: c for c in (lab.ElectrodeConfig.Electrode & insertion_key).fetch('KEY')}

        # insert Unit
        log.info('.. ephys.Unit')

        with InsertBuffer(ephys.Unit, 10, skip_duplicates=True,
                          allow_direct_insert=True) as ib:

            for i, u in enumerate(set(units)):

                ib.insert1({**skey, **insertion_key,
                            **electrode_keys[vmax_unit_site[i]],
                            'clustering_method': method,
                            'unit': u,
                            'unit_uid': u,
                            'unit_quality': unit_notes[i],
                            'unit_posx': unit_xpos[i],
                            'unit_posy': unit_ypos[i],
                            'unit_amp': unit_amp[i],
                            'unit_snr': unit_snr[i],
                            'spike_times': unit_spikes[i],
                            'waveform': unit_wav[i][0]})

                if ib.flush():
                    log.debug('.... {}'.format(u))

        # insert Unit.UnitTrial
        log.info('.. ephys.Unit.UnitTrial')

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

        log.info('.. inserting file load information')

        self.insert1(skey, skip_duplicates=True)

        self.EphysFile.insert1(
            {**skey, 'probe_insertion_number': probe,
             'ephys_file': str(ef_path.relative_to(rigpath))})

        log.info('ephys ingest for {} complete'.format(skey))

    def _gen_probe_insert(self, sinfo, probe, npx_meta):
        '''
        generate probe insertion for session / probe - for neuropixels recording

        Arguments:

          - sinfo: lab.WaterRestriction * lab.Subject * experiment.Session
          - probe: probe id

        '''

        part_no = npx_meta.probe_SN

        e_config = self._gen_electrode_config(npx_meta)

        # ------ ProbeInsertion ------
        insertion_key = {'subject_id': sinfo['subject_id'],
                         'session': sinfo['session'],
                         'insertion_number': probe}

        # add probe insertion
        log.info('.. creating probe insertion')

        lab.Probe.insert1({'probe': part_no, 'probe_type': e_config['probe_type']}, skip_duplicates=True)

        ephys.ProbeInsertion.insert1({**insertion_key,  **e_config, 'probe': part_no})

        ephys.ProbeInsertion.RecordingSystemSetup.insert1({**insertion_key, 'sampling_rate': npx_meta.meta['imSampRate']})

        return insertion_key

    def _gen_electrode_config(self, npx_meta):
        """
        Generate and insert (if needed) an ElectrodeConfiguration based on the specified neuropixels meta information
        """

        probe_type = npx_meta.probe_model

        if '1.0' in probe_type:
            eg_members = []
            probe_type = {'probe_type': probe_type}
            q_electrodes = lab.ProbeType.Electrode & probe_type
            for shank, shank_col, shank_row, is_used in npx_meta.shankmap['data']:
                electrode = (q_electrodes & {'shank': shank, 'shank_col': shank_col, 'shank_row': shank_row}).fetch1(
                    'KEY')
                eg_members.append({**electrode, 'is_used': is_used, 'electrode_group': 0})
        else:
            raise NotImplementedError('Processing for neuropixels probe model {} not yet implemented'.format(probe_type))

        # ---- compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode) ----
        ec_hash = dict_to_hash({k['electrode']: k for k in eg_members})

        el_list = sorted([k['electrode'] for k in eg_members])
        el_jumps = [0] + np.where(np.diff(el_list) > 1)[0].tolist() + [len(el_list) - 1]
        ec_name = '; '.join([f'{el_list[s]}-{el_list[e]}' for s, e in zip(el_jumps[:-1], el_jumps[1:])])

        e_config = {**probe_type, 'electrode_config_name': ec_name}

        # ---- make new ElectrodeConfig if needed ----
        if not (lab.ElectrodeConfig & {'electrode_config_hash': ec_hash}):

            log.info('.. creating lab.ElectrodeConfig: {}'.format(ec_name))

            lab.ElectrodeConfig.insert1({**e_config, 'electrode_config_hash': ec_hash})

            lab.ElectrodeConfig.ElectrodeGroup.insert1({**e_config, 'electrode_group': 0})

            lab.ElectrodeConfig.Electrode.insert({**e_config, **m} for m in eg_members)

        return e_config

    @staticmethod
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

    def _load_jrclust_v3(self, sinfo, fpath):
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
        bf_path = pathlib.Path(fpath.parent, '{}_bitcode.mat'.format(h2o))

        log.info('.. jrclust v3 data load:')
        log.info('.... sinfo: {}'.format(sinfo))
        log.info('.... probe: {}'.format(fpath.parent.name))

        log.info('.... loading ef_path: {}'.format(str(ef_path)))
        ef = h5py.File(str(ef_path), mode='r')  # ephys file

        log.info('.... loading bf_path: {}'.format(str(bf_path)))
        bf = spio.loadmat(bf_path)  # bitcode file

        # extract unit data

        hz = ef['P']['sRateHz'][0][0]                   # sampling rate

        spikes = ef['viTime_spk'][0]                    # spike times
        spike_sites = ef['viSite_spk'][0]               # spike electrode

        units = ef['S_clu']['viClu'][0]                 # spike:unit id
        unit_wav = ef['S_clu']['trWav_raw_clu']         # waveform

        unit_notes = ef['S_clu']['csNote_clu'][0]       # curation notes
        unit_notes = self._decode_notes(ef, unit_notes)

        unit_xpos = ef['S_clu']['vrPosX_clu'][0]        # x position
        unit_ypos = ef['S_clu']['vrPosY_clu'][0]        # y position

        unit_amp = ef['S_clu']['vrVpp_uv_clu'][0]       # amplitude
        unit_snr = ef['S_clu']['vrSnr_clu'][0]          # signal to noise

        vmax_unit_site = ef['S_clu']['viSite_clu']      # max amplitude site
        vmax_unit_site = np.array(vmax_unit_site[:].flatten(), dtype=np.int64)

        trial_start = bf['sTrig'].flatten()           # start of trials
        trial_go = bf['goCue'].flatten()                # go cues

        sync_ephys = bf['bitCodeS'].flatten()           # ephys sync codes
        sync_behav = (experiment.TrialNote()            # behavior sync codes
                      & {**skey, 'trial_note_type': 'bitcode'}).fetch(
                          'trial_note', order_by='trial')

        trial_fix = bf['trialNum'] if 'trialNum' in bf else None

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
        }

        return data

    def _load_jrclust_v4(self, sinfo, fpath):
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

        # bitcode path (ex: 'SC022_030319_Imec3_bitcode.mat')
        bf_path = list(fpath.parent.glob(
            '{}_*bitcode.mat'.format(h2o)))[0]
        log.info('.... loading bf_path: {}'.format(str(bf_path)))
        bf = spio.loadmat(str(bf_path))

        # extract unit data
        hz = bf['SR'][0][0] if 'SR' in bf else None    # sampling rate

        spikes = ef['spikeTimes'][0]                    # spikes times
        spike_sites = ef['spikeSites'][0]               # spike electrode

        units = ef['spikeClusters'][0]                  # spike:unit id
        unit_wav = ef['meanWfLocalRaw']                 # waveform

        unit_notes = ef['clusterNotes']                 # curation notes
        unit_notes = self._decode_notes(ef, unit_notes[:].flatten())

        unit_xpos = ef['clusterCentroids'][0]           # x position
        unit_ypos = ef['clusterCentroids'][1]           # y position

        unit_amp = ef['unitVppRaw'][0]                  # amplitude
        unit_snr = ef['unitSNR'][0]                     # signal to noise

        vmax_unit_site = ef['clusterSites']             # max amplitude site
        vmax_unit_site = np.array(vmax_unit_site[:].flatten(), dtype=np.int64)

        trial_start = bf['sTrig'].flatten()             # trial start
        trial_go = bf['goCue'].flatten()                 # trial go cues

        sync_ephys = bf['bitCodeS']                     # ephys sync codes
        sync_behav = (experiment.TrialNote()            # behavior sync codes
                      & {**skey, 'trial_note_type': 'bitcode'}).fetch(
                          'trial_note', order_by='trial')

        trial_fix = bf['trialNum'] if 'trialNum' in bf else None

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
        }

        return data

    def _load_kilosort2(self, sinfo, ks_dir):

        h2o = sinfo['water_restriction_number']
        skey = {k: v for k, v in sinfo.items()
                if k in experiment.Session.primary_key}

        bf_path = pathlib.Path(ks_dir, '{}_bitcode.mat'.format(h2o))

        log.info('.. kilosort v2 data load:')
        log.info('.... sinfo: {}'.format(sinfo))

        log.info('.... loading bf_path: {}'.format(str(bf_path)))
        bf = spio.loadmat(bf_path)  # bitcode file

        # ---- Read Kilosort results ----
        log.info('.... loading kilosort - ks_dir: {}'.format(str(ks_dir)))
        ks = Kilosort(ks_dir)

        spike_times = ks.data['spike_times']

        # ---- Spike-level results ----
        # -- spike_sites --
        # reimplemented from: https://github.com/JaneliaSciComp/JRCLUST/blob/master/%2Bjrclust/%2Bimport/kilosort.m
        spike_sites = np.full(spike_times.shape, np.nan)
        for template_idx, template in enumerate(ks.data['templates']):
            site_idx = np.abs(np.abs(template).max(axis=0)).argmax()
            spike_sites[ks.data['spike_templates'] == template_idx] = ks.data['channel_map'][site_idx]

        # ---- Unit-level results ----
        # -- Remove 0-spike units
        withspike_idx = [i for i, u in enumerate(ks.data['cluster_ids']) if (ks.data['spike_clusters'] == u).any()]

        valid_units = ks.data['cluster_ids'][withspike_idx]
        valid_unit_labels = ks.data['cluster_groups'][withspike_idx]
        valid_unit_labels = np.where(valid_unit_labels == 'mua', 'multi', valid_unit_labels)  # rename 'mua' to 'multi'

        # -- vmax_unit_site --
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

        # -- waveforms --
        log.info('.... extracting waveforms - data dir: {}'.format(str(ks_dir)))
        unit_wfs = extract_ks_waveforms(ks_dir, ks, wf_win=[-int(ks.data['templates'].shape[1]/2),
                                                            int(ks.data['templates'].shape[1]/2)])
        unit_wav = np.dstack([np.nanmean(unit_wfs[u], axis=2) for u in valid_units]).transpose((2, 1, 0))  # unit x channel x sample

        # -- snr --
        unit_snr = [calculate_wf_snr(unit_wfs[u][:, np.where(ks.data['channel_map'] == u_site)[0][0], :])
                    for u, u_site in zip(valid_units, vmax_unit_site)]

        # -- trial-info from bitcode --
        trial_start = bf['sTrig'].flatten()           # start of trials
        trial_go = bf['goCue'].flatten()                # go cues

        sync_ephys = bf['bitCodeS'].flatten()           # ephys sync codes
        sync_behav = (experiment.TrialNote()            # behavior sync codes
                      & {**skey, 'trial_note_type': 'bitcode'}).fetch(
                          'trial_note', order_by='trial')

        trial_fix = bf['trialNum'] if 'trialNum' in bf else None

        # -- Ensuring `spike_times`, `trial_start` and `trial_go` are in `sample` and not `second` --
        hz = ks.data['params']['sample_rate']
        if np.mean(spike_times - np.round(spike_times)) != 0:
            spike_times = np.round(spike_times * hz).astype(int)
        if np.mean(trial_go - np.round(trial_go)) != 0:
            trial_go = np.round(trial_go * hz).astype(int)
        if np.mean(trial_start - np.round(trial_start)) != 0:
            trial_start = np.round(trial_start * hz).astype(int)

        data = {
            'sinfo': sinfo,
            'ef_path': ks_dir,
            'skey': skey,
            'method': 'kilosort2',
            'hz': hz,
            'spikes': spike_times,
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
        }

        return data

    def _get_sess_dir(self, rigpath, h2o, sess_datetime):
        dpath, dglob = None, None
        if pathlib.Path(rigpath, h2o, sess_datetime.date().strftime('%Y%m%d')).exists():
            dpath = pathlib.Path(rigpath, h2o, sess_datetime.date().strftime('%Y%m%d'))
            dglob = '[0-9]/{}'  # probe directory pattern
        else:
            sess_dirs = list(pathlib.Path(rigpath, h2o).glob('*{}_{}_*'.format(
                h2o, sess_datetime.date().strftime('%m%d%y'))))
            for sess_dir in sess_dirs:
                npx_meta = NeuropixelsMeta(next(sess_dir.rglob('{}_*.ap.meta'.format(h2o))))
                # match the recording_time's minute from npx_meta to that of the behavior recording - this is to handle multiple sessions in a day
                if npx_meta.recording_time.minute == sess_datetime.minute:
                    dpath = sess_dir
                    dglob = '{}_{}_*_imec[0-9]'.format(h2o, sess_datetime.date().strftime('%m%d%y')) + '/{}'  # probe directory pattern
                    break
        return dpath, dglob

    def _match_probe_to_ephys(self, h2o, dpath, dglob):
        """
        Based on the identified spike sorted file(s), match the probe number (i.e. 1, 2, 3) to the cluster filepath, loader, and npx_meta
        Return a dict, e.g.:
        {
         1: (cluster_fp, loader, npx_meta),
         2: (cluster_fp, loader, npx_meta),
         3: (cluster_fp, loader, npx_meta),
        }
        """
        # npx ap.meta: '{}_*.imec.ap.meta'.format(h2o)
        npx_meta_files = list(dpath.glob(dglob.format('{}_*.ap.meta'.format(h2o))))
        if not npx_meta_files:
            raise FileNotFoundError('Error - no ap.meta files at {}'.format(dpath))

        jrclustv3spec = '{}_*_jrc.mat'.format(h2o)
        jrclustv4spec = '{}_*.ap_res.mat'.format(h2o)
        ks2spec = 'spike_times.npy'

        clustered_probes = {}
        for meta_file in npx_meta_files:
            probe_dir = meta_file.parent
            probe_number = re.search('(imec)?\d{1}$', probe_dir.name).group()
            probe_number = int(probe_number.replace('imec', '')) + 1 if 'imec' in probe_number else int(probe_number)

            # JRClust v3
            v3files = [(f, self._load_jrclust_v3) for f in probe_dir.glob(jrclustv3spec)]
            # JRClust v4
            v4files = [(f, self._load_jrclust_v4) for f in probe_dir.glob(jrclustv4spec)]
            # Kilosort
            ks2files = [(f.parent, self._load_kilosort2) for f in probe_dir.glob(ks2spec)]

            clustering_results = v4files + v3files + ks2files

            if len(clustering_results) < 1:
                raise FileNotFoundError('Error - No clustering results found at {}'.format(probe_dir))
            elif len(clustering_results) > 1:
                log.warning('Found multiple clustering results at {probe_dir}. Prioritize JRC4 > JRC3 > KS2'.format(probe_dir))

            fp, loader = clustering_results[0]
            clustered_probes[probe_number] = (fp, loader, NeuropixelsMeta(meta_file))

        return clustered_probes


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
                d = np.load(f, mmap_mode='r', allow_pickle=False, fix_imports=False)
                self._data[base] = np.reshape(d, d.shape[0]) if d.ndim == 2 and d.shape[1] == 1 else d

        # Read the Cluster Groups
        if (self._dname / 'cluster_groups.csv').exists():
            df = pd.read_csv(self._dname / 'cluster_groups.csv', delimiter='\t')
            self._data['cluster_groups'] = np.array(df['group'].values)
            self._data['cluster_ids'] = np.array(df['cluster_id'].values)
        elif (self._dname / 'cluster_KSLabel.tsv').exists():
            df = pd.read_csv(self._dname / 'cluster_KSLabel.tsv', sep = "\t", header = 0)
            self._data['cluster_groups'] = np.array(df['KSLabel'].values)
            self._data['cluster_ids'] = np.array(df['cluster_id'].values)
        else:
            raise FileNotFoundError('Neither cluster_groups.csv nor cluster_KSLabel.tsv found!')


def extract_ks_waveforms(npx_dir, ks, n_wf=5, wf_win=(-41, 41), bit_volts=None):
    """
    :param npx_dir: directory to the ap.bin and ap.meta
    :param ks: instance of Kilosort
    :param n_wf: number of spikes per unit to extract the waveforms
    :param wf_win: number of sample pre and post a spike
    :param bit_volts: scalar required to convert int16 values into microvolts
    :return: dictionary of the clusters' waveform (sample x channel x spike)
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
        if len(spikes) > 0:
            # waveform at each spike: (sample x channel x spike)
            spike_wfs = np.dstack([data[int(spk+wf_win[0]):int(spk+wf_win[-1]), chan_map] for spk in spikes])
            unit_wfs[unit] = spike_wfs * bit_volts
        else:  # if no spike found, return NaN of size (sample x channel x 1)
            unit_wfs[unit] = np.full((len(range(*wf_win)), len(chan_map), 1), np.nan)

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

    return snr
