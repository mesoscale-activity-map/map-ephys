#! /usr/bin/env python

import os
import glob
import logging
import datetime

from itertools import chain
from collections import namedtuple

import scipy.io as spio
import h5py
import numpy as np

import datajoint as dj

import lab
import experiment
import ephys
import ingestBehavior

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

    contents = [(r'H:\\data\MAP', 1)] # Testing the JRClust output files on my computer

@schema
class EphysIngest(dj.Imported):
    # subpaths like: \Spike\2017-10-21\tw5ap_imec3_opt3_jrc.mat

    definition = """
    -> ingestBehavior.SessionDiscovery
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

        rigpath = EphysDataPath().fetch1('data_path')
        date = key['session_date'].strftime('%Y-%m-%d')
        subject_id = key['subject_id']
        water = (lab.WaterRestriction() & {'subject_id': subject_id}).fetch1('water_restriction_number')
        file = '{h2o}ap_imec3_opt3_jrc.mat'.format(h2o=water)
        subpath = os.path.join('Spike', date, file)
        fullpath = os.path.join(rigpath, subpath)

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
            behavior = (BehaviorIngest() & key).fetch1()
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

        ephys.ElectrodeGroup().insert1(dict(ekey, probe_part_no=15131808323))
        ephys.ElectrodeGroup().make(ekey)  # note: no locks; is dj.Manual
        ephys.Ephys().insert1(ekey, ignore_extra_fields=True) # insert Ephys first

        f = h5py.File(fullpath,'r')
        ind = np.argsort(f['S_clu']['viClu'][0]) # index sorted by cluster
        cluster_ids = f['S_clu']['viClu'][0][ind] # cluster (unit) number
        trWav_raw_clu = f['S_clu']['trWav_raw_clu'] # spike waveform
        trWav_raw_clu1 = np.concatenate((trWav_raw_clu[0:1][:][:],trWav_raw_clu),axis=0) # add a spike waveform of cluster 0
        spike_times = f['viTime_spk'][0][ind] # spike times
        viSite_spk = f['viSite_spk'][0][ind] # electrode site for the spike
        viT_offset_file = f['viT_offset_file'][:] # start of each trial, subtract this number for each trial
        sRateHz = f['P']['sRateHz'][0] # sampling rate
        spike_trials = np.ones(len(spike_times)) * (len(viT_offset_file) - 1) # every spike is in the last trial
        spike_times2 = np.copy(spike_times)
        for i in range(len(viT_offset_file) - 1, 0, -1): #find the trials each unit has a spike in
            spike_trials[spike_times < viT_offset_file[i]] = i-1 # Get the trial number of each spike
            spike_times2[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] = spike_times[(spike_times >= viT_offset_file[i-1]) & (spike_times < viT_offset_file[i])] - viT_offset_file[i - 1] # subtract the viT_offset_file from each trial
        spike_times2[np.where(spike_times2 >= viT_offset_file[-1])] = spike_times[np.where(spike_times2 >= viT_offset_file[-1])] - viT_offset_file[-1] # subtract the viT_offset_file from each trial
        spike_times2 = spike_times2 / sRateHz # divide the sampling rate, sRateHz
        clu_ids_diff = np.diff(cluster_ids) # where the units seperate
        clu_ids_diff = np.where(clu_ids_diff != 0)[0] + 1 # separate the spike_times
        units = np.split(spike_times, clu_ids_diff) / sRateHz # sub arrays of spike_times
        trialunits = np.split(spike_trials, clu_ids_diff) # sub arrays of spike_trials
        unit_ids = np.arange(len(clu_ids_diff) + 1) # unit number
        trialunits1 = [] # array of unit number
        trialunits2 = [] # array of trial number
        for i in range(0,len(trialunits)):
            trialunits2 = np.append(trialunits2, np.unique(trialunits[i]))
            trialunits1 = np.append(trialunits1, np.zeros(len(np.unique(trialunits[i])))+i)
        ephys.Ephys.Unit().insert(list(dict(ekey, unit = x, spike_times = units[x], waveform = trWav_raw_clu1[x][0]) for x in unit_ids)) # batch insert the units
        #experiment.Session.Trial() #TODO: fetch the trial from experiment.Session.Trial and realign?
        ephys.Ephys.TrialUnit().insert(list(dict(ekey, unit = trialunits1[x], trial = trialunits2[x]) for x in range(0, len(trialunits2)))) # batch insert the TrialUnit (key, unit, trial)
        ephys.Ephys.Spike().insert(list(dict(ekey, unit = cluster_ids[x], spike_time = spike_times2[x], electrode = viSite_spk[x], trial = spike_trials[x]) for x in range(0, len(spike_times2))), skip_duplicates=True) # batch insert the Spikes (key, unit, spike_time, electrode, trial)

        self.insert1(key)
        EphysIngest.EphysFile().insert1(dict(key, ephys_file=subpath))
