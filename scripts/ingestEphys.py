#! /usr/bin/env python

import os
import glob
import logging
import datetime
import pdb

from itertools import chain
from collections import namedtuple

import scipy.io as spio
import h5py
import numpy as np

import datajoint as dj

from pipeline import lab
from pipeline import experiment
from pipeline import ephys
from scripts import ingestBehavior


log = logging.getLogger(__name__)
schema = dj.schema(dj.config['ingestEphys.database'])

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
        if 'ephys_data_paths' in dj.config:  # for local testing
            return dj.config['ephys_data_paths']

        return ((r'H:\\data\MAP', 0),)


@schema
class EphysIngest(dj.Imported):
    # subpaths like: \2017-10-21\tw5ap_imec3_opt3_jrc.mat

    definition = """
    -> ingestBehavior.BehaviorIngest
    """

    class EphysFile(dj.Part):
        # TODO: track files
        ''' files in rig-specific storage '''
        definition = """
        -> EphysIngest
        ephys_file:              varchar(255)          # rig file subpath
        """

    def make(self, key):
        log.info('EphysIngest().make(): key: {k}'.format(k=key))

        #
        # Find Ephys Recording
        #
        key = (experiment.Session & key).fetch1()

        rigpath = EphysDataPath().fetch1('data_path')
        date = key['session_date'].strftime('%Y-%m-%d')
        subject_id = key['subject_id']
        water = (lab.WaterRestriction() & {'subject_id': subject_id}).fetch1('water_restriction_number')
        file = '{h2o}ap_imec3_opt3_jrc.mat'.format(h2o=water) # current file naming format
#        file = '{h2o}_g0_t0.imec.ap_imec3_opt3_jrc.mat'.format(h2o=water) # some older files
        # subpath = os.path.join('Spike', date, file)
        fullpath = os.path.join(rigpath, date, file)

        if not os.path.exists(fullpath):
            log.info('EphysIngest().make(): skipping - no file in %s'
                     % fullpath)
            return

        log.info('EphysIngest().make(): found ephys recording in %s'
                 % fullpath)

        #
        # Find corresponding BehaviorIngest
        #
        # ... we are keying times, sessions, etc from behavior ingest;
        # so lookup behavior ingest for session id, quit with warning otherwise

        try:
            behavior = (ingestBehavior.BehaviorIngest() & key).fetch1()
        except dj.DataJointError:
            log.warning('EphysIngest().make(): skip - behavior ingest error')
            return

        log.info('behavior for ephys: {b}'.format(b=behavior))

        #
        # Prepare ElectrodeGroup configuration
        #
        # HACK / TODO: assuming single specific ElectrodeGroup for all tests;
        # better would be to have this encoded in filename or similar.

        ekey = {
            'subject_id': behavior['subject_id'],
            'session': behavior['session'],
            'electrode_group': 1,
        }

        log.debug('inserting electrode group')
        ephys.ElectrodeGroup().insert1(dict(ekey, probe_part_no=15131808323))
        ephys.ElectrodeGroup().make(ekey)  # note: no locks; is dj.Manual

        log.debug('extracting spike data')

        f = h5py.File(fullpath,'r')
        ind = np.argsort(f['S_clu']['viClu'][0]) # index sorted by cluster
        cluster_ids = f['S_clu']['viClu'][0][ind] # cluster (unit) number
        ind = ind[np.where(cluster_ids > 0)[0]] # get rid of the -ve noise clusters
        cluster_ids = cluster_ids[np.where(cluster_ids > 0)[0]] # get rid of the -ve noise clusters
        trWav_raw_clu = f['S_clu']['trWav_raw_clu'] # spike waveform
#        trWav_raw_clu1 = np.concatenate((trWav_raw_clu[0:1][:][:],trWav_raw_clu),axis=0) # add a spike waveform to cluster 0, not necessary anymore after the previous step
        csNote_clu=f['S_clu']['csNote_clu'][0] # manual sorting note
        strs = ["all" for x in range(len(csNote_clu))] # all units are "all" by definition
        for iU in range(0, len(csNote_clu)): # read the manual curation of each unit
            log.debug('extracting spike indicators {s}:{u}'.format(s=behavior['session'], u=iU))
            unitQ = f[csNote_clu[iU]]
            str1 = ''.join(chr(i) for i in unitQ[:])
            if str1 == 'single': # definitions in unit quality
                strs[iU] = 'good'
            elif str1 =='multi':
                strs[iU] = 'multi'
        spike_times = f['viTime_spk'][0][ind] # spike times
        viSite_spk = f['viSite_spk'][0][ind] # electrode site for the spike

        file = '{h2o}_bitcode.mat'.format(h2o=water) # fetch the bitcode and realign
        # subpath = os.path.join('Spike', date, file)
        fullpath = os.path.join(rigpath, date, file)

        log.debug('opening bitcode for {s} ({f})'.format(s=behavior['session'], f=fullpath))

        mat = spio.loadmat(fullpath, squeeze_me = True) # load the bitcode file

        goCue = mat['goCue'][:]
        viT_offset_file = f['viT_offset_file'][:] # start of each trial, subtract this number for each trial
        sRateHz = f['P']['sRateHz'][0] # sampling rate
        spike_trials = np.ones(len(spike_times)) * (len(viT_offset_file) - 1) # every spike is in the last trial
        spike_times2 = np.copy(spike_times)
        for i in range(len(viT_offset_file) - 1, 0, -1): #find the trials each unit has a spike in
            log.debug('locating trials with spikes {s}:{t}'.format(s=behavior['session'], t=i))
            spike_trials[spike_times < viT_offset_file[i]] = i-1 # Get the trial number of each spike
            spike_times2[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] = spike_times[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] - goCue[i - 1] # subtract the viT_offset_file from each trial
        spike_times2[np.where(spike_times2 >= viT_offset_file[-1])] = spike_times[np.where(spike_times2 >= viT_offset_file[-1])] - goCue[-1] # subtract the viT_offset_file from each trial
        spike_times2 = spike_times2 / sRateHz # divide the sampling rate, sRateHz
        clu_ids_diff = np.diff(cluster_ids) # where the units seperate
        clu_ids_diff = np.where(clu_ids_diff != 0)[0] + 1 # separate the spike_times

        units = np.split(spike_times, clu_ids_diff) / sRateHz # sub arrays of spike_times for each unit (for ephys.Unit())
        trialunits = np.split(spike_trials, clu_ids_diff) # sub arrays of spike_trials for each unit
        unit_ids = np.arange(len(clu_ids_diff) + 1) # unit number
        
        trialunits1 = [] # array of unit number (for ephys.Unit.UnitTrial())
        trialunits2 = [] # array of trial number
        for i in range(0,len(trialunits)): # loop through each unit
            log.debug('aggregating trials with units {s}:{t}'.format(s=behavior['session'], t=i))
            trialunits2 = np.append(trialunits2, np.unique(trialunits[i])) # add the trials that a unit is in
            trialunits1 = np.append(trialunits1, np.zeros(len(np.unique(trialunits[i])))+i) # add the unit numbers 

        log.debug('inserting units for session {s}'.format(s=behavior['session']))
        ephys.Unit().insert(list(dict(ekey, unit = x, unit_uid = x, unit_quality = strs[x], spike_times = units[x], waveform = trWav_raw_clu[x][0]) for x in unit_ids)) # batch insert the units

        bitCodeE = mat['bitCodeS'].flatten() # bitCodeS is the char variable
        trialNote = experiment.TrialNote()
        bitCodeB = (trialNote & {'subject_id': ekey['subject_id']} & {'session': ekey['session']} & {'trial_note_type': 'bitcode'}).fetch('trial_note', order_by='trial') # fetch the bitcode from the behavior trialNote
        if len(bitCodeB) < len(bitCodeE): # behavior file is shorter; e.g. seperate protocols were used; Bpod trials missing due to crash; session restarted
            startB = np.where(bitCodeE==bitCodeB[0])[0]
        elif len(bitCodeB) > len(bitCodeE): # behavior file is longer; e.g. only some trials are sorted, the bitcode.mat should reflect this; Sometimes SpikeGLX can skip a trial, I need to check the last trial
            startE = np.where(bitCodeB==bitCodeE[0])[0]
            startB = -startE
            startB = -39 # hack to get tw34 going
        else:
            startB = 0
            startE = 0

        log.debug('extracting trial unit information {s} ({f})'.format(s=behavior['session'], f=fullpath))

        #pdb.set_trace()
        trialunits2 = trialunits2-startB # behavior has less trials if startB is +ve, behavior has more trials if startB is -ve
        indT = np.where(trialunits2 > -1)[0] # get rid of the -ve trials
        trialunits1 = trialunits1[indT]
        trialunits2 = trialunits2[indT]

        spike_trials = spike_trials - startB # behavior has less trials if startB is +ve, behavior has more trials if startB is -ve
        indT = np.where(spike_trials > -1)[0] # get rid of the -ve trials
        cluster_ids = cluster_ids[indT]
        spike_times2 = spike_times2[indT]
        viSite_spk = viSite_spk[indT]
        spike_trials = spike_trials[indT]

        trialunits = np.asarray(trialunits) # convert the list to an array
        trialunits = trialunits - startB

        # split units based on which trial they are in (for ephys.TrialSpikes())
        trialPerUnit = np.copy(units) # list of trial index for each unit
        for i in unit_ids: # loop through each unit, maybe this can be avoid?
            log.debug('.. unit information {u}'.format(u=i))
            indT = np.where(trialunits[i] > -1)[0] # get rid of the -ve trials
            trialunits[i] = trialunits[i][indT]
            units[i] = units[i][indT]
            trialidx = np.argsort(trialunits[i]) # index of the sorted trials
            trialunits[i] = np.sort(trialunits[i]) # sort the trials for a given unit
            trial_ids_diff = np.diff(trialunits[i]) # where the trial index seperate
            trial_ids_diff = np.where(trial_ids_diff != 0)[0] + 1
            units[i] = units[i][trialidx] # sort the spike times based on the trial mapping
            units[i] = np.split(units[i], trial_ids_diff) # separate the spike_times based on trials
            trialPerUnit[i] = np.arange(0, len(trial_ids_diff)+1, dtype = int) # list of trial index


        log.debug('inserting UnitTrial information')
        ephys.Unit.UnitTrial().insert(list(dict(ekey, unit = trialunits1[x], trial = trialunits2[x]) for x in range(0, len(trialunits2)))) # batch insert the TrialUnit (key, unit, trial)
        log.debug('inserting UnitSpike information')
        ephys.Unit.UnitSpike().insert(list(dict(ekey, unit = cluster_ids[x]-1, spike_time = spike_times2[x], electrode = viSite_spk[x], trial = spike_trials[x]) for x in range(0, len(spike_times2))), skip_duplicates=True) # batch insert the Spikes (key, unit, spike_time, electrode, trial)

        # TODO: 2D batch insert
        # pdb.set_trace()
        l = [] # list of trialSpikes to be inserted
        for x in zip(unit_ids, trialPerUnit): # loop through the units
            for i in x[1]: # loop through the trials for each unit
                l.append(dict(ekey, unit=x[0], trial=int(trialunits2[x[1]][i]), spike_times=units[x[0]][x[1][i]])) # create the list
        ephys.TrialSpikes().insert(l, skip_duplicates=True) # batch insert TrialSpikes

        log.debug('inserting file load information')
        self.insert1(key, ignore_extra_fields=True)
        EphysIngest.EphysFile().insert1(dict(key, ephys_file=fullpath), ignore_extra_fields=True)
