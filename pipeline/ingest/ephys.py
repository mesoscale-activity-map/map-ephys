#! /usr/bin/env python

import os
import logging
import pathlib
from glob import glob

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


@schema
class EphysDataPath(dj.Lookup):
    # ephys data storage location(s)
    definition = """
    data_path:              varchar(255)           # rig data path  
    ---
    search_order:           int                     # rig search order
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
    -> behavior_ingest.BehaviorIngest
    """

    class EphysFile(dj.Part):
        ''' files in rig-specific storage '''
        definition = """
        -> EphysIngest
        probe_insertion_number:        tinyint                 # electrode_group
        ephys_file:             varchar(255)            # rig file subpath
        """

    def make(self, key):
        '''
        Ephys .make() function
        '''

        log.info('EphysIngest().make(): key: {k}'.format(k=key))

        #
        # Find corresponding BehaviorIngest
        #
        # ... we are keying times, sessions, etc from behavior ingest;
        # so lookup behavior ingest for session id, quit with warning otherwise

        try:
            behavior = (behavior_ingest.BehaviorIngest() & key).fetch1()
        except dj.DataJointError:
            log.warning('EphysIngest().make(): skip - behavior ingest error')
            return

        log.info('behavior for ephys: {b}'.format(b=behavior))

        #
        # Find Ephys Recording
        #
        key = (experiment.Session & key).fetch1()

        rigpath = EphysDataPath().fetch1('data_path')
        date = key['session_date'].strftime('%Y-%m-%d')
        subject_id = key['subject_id']
        water = (lab.WaterRestriction() & {'subject_id': subject_id}).fetch1('water_restriction_number')

        for probe in range(1, 3):

            # TODO: should code include actual logic to pick these up still?
            # file = '{h2o}_g0_t0.imec.ap_imec3_opt3_jrc.mat'.format(h2o=water) # some older files
            # subpath = os.path.join('{}-{}'.format(date, probe), file)
            # file = '{h2o}ap_imec3_opt3_jrc.mat'.format(h2o=water) # current file naming format
            epfile = '{h2o}_g0_*.imec.ap_imec3_opt3_jrc.mat'.format(h2o=water)  # current file naming format
            epfullpath = pathlib.Path(rigpath, water, date, str(probe))
            ephys_files = list(epfullpath.glob(epfile))

            if len(ephys_files) != 1:
                log.info('EphysIngest().make(): skipping probe {} - incorrect files found: {}/{}'.format(probe, epfullpath, ephys_files))
                continue

            epfullpath = ephys_files[0]
            epsubpath = epfullpath.relative_to(rigpath)
            log.info('EphysIngest().make(): found probe {} ephys recording in {}'.format(probe, epfullpath))

            #
            # Prepare ProbeInsertion configuration
            #
            # HACK / TODO: assuming single specific ProbeInsertion for all tests;
            # better would be to have this encoded in filename or similar.
            probe_part_no = '15131808323'  # hard-coded here

            ekey = {
                'subject_id': behavior['subject_id'],
                'session': behavior['session'],
                'insertion_number': probe
            }

            # ElectrodeConfig - add electrode group and group member (hard-coded to be the first 384 electrode)
            electrode_group = {'probe': probe_part_no, 'electrode_group': 0}
            electrode_group_member = [{**electrode_group, 'electrode': chn} for chn in range(1, 385)]
            electrode_config_name = 'npx_first384'  # user-friendly name - npx probe config with the first 384 channels
            electrode_config_hash = dict_to_hash(
                {**electrode_group, **{str(idx): k for idx, k in enumerate(electrode_group_member)}})
            # extract ElectrodeConfig, check DB to reference if exists, else create
            if ({'probe': probe_part_no, 'electrode_config_name': electrode_config_name}
                    not in lab.ElectrodeConfig()):
                log.info('create Neuropixels electrode configuration (lab.ElectrodeConfig)')
                lab.ElectrodeConfig.insert1({
                    'probe': probe_part_no,
                    'electrode_config_hash': electrode_config_hash,
                    'electrode_config_name': electrode_config_name})
                lab.ElectrodeConfig.ElectrodeGroup.insert1({'electrode_config_name': electrode_config_name,
                                                            **electrode_group})
                lab.ElectrodeConfig.Electrode.insert(
                    {'electrode_config_name': electrode_config_name, **member} for member in electrode_group_member)

            log.info('inserting probe insertion')
            ephys.ProbeInsertion.insert1(dict(ekey, probe=probe_part_no, electrode_config_name=electrode_config_name))

            #
            # Extract spike data
            #

            log.info('extracting spike data')

            f = h5py.File(epfullpath, 'r')
            cluster_ids = f['S_clu']['viClu'][0]# cluster (unit) number
            trWav_raw_clu = f['S_clu']['trWav_raw_clu'] # spike waveform
    #        trWav_raw_clu1 = np.concatenate((trWav_raw_clu[0:1][:][:],trWav_raw_clu),axis=0) # add a spike waveform to cluster 0, not necessary anymore after the previous step
            csNote_clu = f['S_clu']['csNote_clu'][0] # manual sorting note
            viSite_clu = f['S_clu']['viSite_clu'][:] # site of the unit with the largest amplitude
            vrPosX_clu = f['S_clu']['vrPosX_clu'][0] # x position of the unit
            vrPosY_clu = f['S_clu']['vrPosY_clu'][0] # y position of the unit
            vrVpp_uv_clu = f['S_clu']['vrVpp_uv_clu'][0] # amplitude of the unit
            vrSnr_clu = f['S_clu']['vrSnr_clu'][0] # snr of the unit
            strs = ["all" for x in range(len(csNote_clu))] # all units are "all" by definition
            for iU in range(0, len(csNote_clu)): # read the manual curation of each unit
                log.debug('extracting spike indicators {s}:{u}'.format(s=behavior['session'], u=iU))
                unitQ = f[csNote_clu[iU]]
                str1 = ''.join(chr(i) for i in unitQ[:])
                if str1 == 'single':  # definitions in unit quality
                    strs[iU] = 'good'
                elif str1 =='ok':
                    strs[iU] = 'ok'
                elif str1 =='multi':
                    strs[iU] = 'multi'
            spike_times = f['viTime_spk'][0]  # spike times
            viSite_spk = f['viSite_spk'][0]  # electrode site for the spike
            sRateHz = f['P']['sRateHz'][0]  # sampling rate

            # get rid of the -ve noise clusters
            non_neg_cluster_idx = cluster_ids > 0

            cluster_ids = cluster_ids[non_neg_cluster_idx]
            spike_times = spike_times[non_neg_cluster_idx]
            viSite_spk = viSite_spk[non_neg_cluster_idx]

            file = '{h2o}_bitcode.mat'.format(h2o=water) # fetch the bitcode and realign
            # subpath = os.path.join('{}-{}'.format(date, probe), file)
            bcsubpath = pathlib.Path(water, date, str(probe), file)
            bcfullpath = rigpath / bcsubpath

            log.info('opening bitcode for session {s} probe {p} ({f})'
                     .format(s=behavior['session'], p=probe, f=bcfullpath))

            mat = spio.loadmat(bcfullpath, squeeze_me = True) # load the bitcode file

            log.info('extracting spike information {s} probe {p} ({f})'
                     .format(s=behavior['session'], p=probe, f=bcfullpath))

            bitCodeE = mat['bitCodeS'].flatten()  # bitCodeS is the char variable
            goCue = mat['goCue'].flatten()  # bitCodeS is the char variable
            viT_offset_file = mat['sTrig'].flatten() - 7500 # start of each trial, subtract this number for each trial
            trialNote = experiment.TrialNote()
            bitCodeB = (trialNote & {'subject_id': ekey['subject_id']} & {'session': ekey['session']} & {'trial_note_type': 'bitcode'}).fetch('trial_note', order_by='trial') # fetch the bitcode from the behavior trialNote

            # check ephys/bitcode match to determine trial numbering method
            bitCodeB_0 = np.where(bitCodeB == bitCodeE[0])[0][0]
            bitCodeB_ext = bitCodeB[bitCodeB_0:][:len(bitCodeE)]
            spike_trials_fix = None
            if not np.all(np.equal(bitCodeE, bitCodeB_ext)):
                log.info('ephys/bitcode trial mismatch - attempting fix')
                if 'trialNum' in mat:
                    spike_trials_fix = mat['trialNum']
                else:
                    raise Exception('Bitcode Mismatch')

            spike_trials = np.full_like(spike_times, (len(viT_offset_file) - 1))  # every spike is in the last trial
            spike_times2 = np.copy(spike_times)
            for i in range(len(viT_offset_file) - 1, 0, -1): #find the trials each unit has a spike in
                log.debug('locating trials with spikes {s}:{t}'.format(s=behavior['session'], t=i))
                spike_trials[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] = i-1  # Get the trial number of each spike
                spike_times2[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] = spike_times[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] - goCue[i-1] # subtract the goCue from each trial

            spike_trials[np.where(spike_times2 >= viT_offset_file[-1])] = len(viT_offset_file) - 1
            spike_times2[np.where(spike_times2 >= viT_offset_file[-1])] = spike_times[np.where(spike_times2 >= viT_offset_file[-1])] - goCue[-1] # subtract the goCue from the last trial

            spike_times2 = spike_times2 / sRateHz  # divide the sampling rate, sRateHz

            # at this point, spike-times are aligned to go-cue for that respective trial
            unit_trial_spks = {u: (spike_trials[cluster_ids == u], spike_times2[cluster_ids == u])
                               for u in set(cluster_ids)}
            trial_start_time = viT_offset_file / sRateHz

            log.info('inserting units for session {s}'.format(s=behavior['session']))
            #pdb.set_trace()

            # Unit - with JRclust clustering method
            ekey['clustering_method'] = 'jrclust'
            def build_unit_insert():
                for u_id, (u, (u_spk_trials, u_spk_times)) in enumerate(unit_trial_spks.items()):
                    # unit spike times - realign back to trial-start, relative to 1st trial
                    spk_times = sorted(u_spk_times + (goCue / sRateHz)[u_spk_trials] + trial_start_time[u_spk_trials])
                    yield (dict(ekey, unit=u, unit_uid=u, unit_quality=strs[u_id],
                                      electrode_config_name=electrode_config_name, probe=probe_part_no,
                                      electrode_group=0, electrode=int(viSite_clu[u_id]),
                                      unit_posx=vrPosX_clu[u_id], unit_posy=vrPosY_clu[u_id],
                                      unit_amp=vrVpp_uv_clu[u_id], unit_snr=vrSnr_clu[u_id],
                                      spike_times=spk_times, waveform=trWav_raw_clu[u_id][0]))

            ephys.Unit.insert(build_unit_insert(), allow_direct_insert=True)

            # UnitTrial
            log.info('inserting UnitTrial information')

            if spike_trials_fix is None:
                if len(bitCodeB) < len(bitCodeE):  # behavior file is shorter; e.g. seperate protocols were used; Bpod trials missing due to crash; session restarted
                    startB = np.where(bitCodeE == bitCodeB[0])[0].squeeze()
                elif len(bitCodeB) > len(bitCodeE):  # behavior file is longer; e.g. only some trials are sorted, the bitcode.mat should reflect this; Sometimes SpikeGLX can skip a trial, I need to check the last trial
                    startE = np.where(bitCodeB == bitCodeE[0])[0].squeeze()
                    startB = -startE
                else:
                    startB = 0
                    startE = 0
                spike_trials_fix = np.arange(spike_trials.max() + 1)
            else:  # XXX: under test
                startB = 0
                startE = 0
                spike_trials_fix -= 1

            with InsertBuffer(ephys.Unit.UnitTrial, 10000,
                              skip_duplicates=True,
                              allow_direct_insert=True) as ib:

                for x, (u_spk_trials, u_spk_times) in unit_trial_spks.items():
                    ib.insert(dict(ekey, unit=x,
                                   trial=spike_trials_fix[tr] - startB) for tr in set(spike_trials))
                    if ib.flush():
                        log.debug('... UnitTrial spike')

            # TrialSpike
            with InsertBuffer(ephys.TrialSpikes, 10000, skip_duplicates = True,
                              allow_direct_insert = True) as ib:
                for x, (u_spk_trials, u_spk_times) in unit_trial_spks.items():
                    ib.insert(dict(ekey, unit=x,
                                   spike_times=u_spk_times[u_spk_trials == tr],
                                   trial=spike_trials_fix[tr] - startB) for tr in set(spike_trials))
                    if ib.flush():
                        log.debug('... TrialSpike spike')

            log.info('inserting file load information')

            self.insert1(key, ignore_extra_fields=True, skip_duplicates=True,
                         allow_direct_insert=True)

            EphysIngest.EphysFile().insert1(
                dict(key, probe_insertion_number=probe, ephys_file=epsubpath.as_posix()),
                ignore_extra_fields=True, allow_direct_insert=True)

            log.info('ephys ingest for {} complete'.format(key))
