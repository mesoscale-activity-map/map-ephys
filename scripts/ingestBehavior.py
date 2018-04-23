#! /usr/bin/env python

import os
import glob
import logging
import datetime
import re
import pdb

from itertools import chain
from collections import namedtuple

import scipy.io as spio
import h5py
import numpy as np
import warnings

import datajoint as dj

from pipeline import lab
from pipeline import experiment
from pipeline import ephys

warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)
schema = dj.schema(dj.config['ingestBehavior.database'])


@schema
class RigDataPath(dj.Lookup):
    ''' rig storage locations '''
    # todo: cross platform path mapping needed?
    definition = """
    -> lab.Rig
    ---
    rig_data_path:              varchar(1024)           # rig data path
    rig_search_order:           int                     # rig search order
    """

    @property
    def contents(self):
        if 'rig_data_paths' in dj.config:  # for local testing
            return dj.config['rig_data_paths']

        return (('TRig1', r'\\MOHARB-NUC1\Documents\Arduino\Bpod_Train1\Bpod Local\Data', 0), # Hardcode the rig path
                ('TRig2', r'\\MOHARB-WW2\C\Users\labadmin\Documents\MATLAB\Bpod Local\Data', 1),
                ('TRig3', r'\\WANGT-NUC\Documents\MATLAB\Bpod Local\Data', 2),
                ('RRig', r'\\wangt-ww1\Documents\MATLAB\Bpod Local\Data', 3)
                ) # A table with the data path


@schema
class SessionDiscovery(dj.Manual):
    '''
    Table to populate sessions available in filesystem for discovery

    Note: session date is duplicated w/r/t actual Session table;
    this is somewhat unavoidable since session requires the
    synthetic session ID and we are not quite ready to generate it;
    put another way, this table helps to map one session ID (h2o+date)
    into the 'official' sequential session ID of the main schema
    '''

    definition = """
    -> lab.WaterRestriction
    session_date:               date                    # discovered date
    """

    def populate(self):
        '''
        Scan the RigDataPath records, looking for new unknown sessions.

        Local implementation, since we aren't really a computed table.
        '''

        rigs = [r for r in RigDataPath().fetch(as_dict=True)
                if r['rig'].startswith('RRig')]  # todo?: rig 'type'? Change between TRig and RRig for now

        h2os = {k: v for k, v in
                zip(*lab.WaterRestriction().fetch(
                    'water_restriction_number', 'subject_id'))} # fetch existing water_restriction_number

        initial = SessionDiscovery().fetch(as_dict=True) # sessions already discovered
        log.debug('initial: %s' % rigs)
        found = []
        
        rexp = '^[a-zA-Z]{2}.*_.*_[0-9]{8}_[0-9]{6}.mat$'

        for r in rigs:
            data_path = r['rig_data_path']
            for root, dirs, files in os.walk(data_path):

                log.info('RigDataFile.make(): traversing %s' % root)
                log.debug('RigDataFile.make(): dirs %s' % dirs)
                log.debug('RigDataFile.make(): files %s' % files)
                subpaths = list(os.path.join(root, f)
                                .split(data_path)[1].lstrip(os.path.sep)
                                for f in files if re.match(rexp,f)) # find files with tw2

                for filename in subpaths:
                    log.debug('found file %s' % filename)

                    # split files like 'dl7_TW_autoTrain_20171114_140357.mat'
                    filename = os.path.basename(filename)
                    fsplit = filename.split('.')[0].split('_')
                    h2o, date = (fsplit[0], fsplit[-2:-1][0],) # first is the water restriction, second last is the date

                    if h2o not in h2os:
                        log.warning('{f} skipped - no animal for {h2o}'.format(
                            f=filename, h2o=h2o))
                        continue
                    else:
                        animal = h2os[h2o]

                    log.debug('animal is {animal}'.format(animal=animal))

                    key = {
                        'subject_id': animal,
                        'session_date': datetime.date(
                            int(date[0:4]), int(date[4:6]), int(date[6:8]))
                    }

                    if key not in found and key not in initial:
                        log.info('found session: %s' % key)
                        found.append(key) # finding new sessions

        # add the new sessions
        self.insert(found)


@schema
class BehaviorIngest(dj.Imported):
    definition = """
    -> SessionDiscovery
    ---
    -> experiment.Session
    """

    class BehaviorFile(dj.Part):
        # TODO: track files
        ''' files in rig-specific storage '''
        definition = """
        -> BehaviorIngest
        behavior_file:              varchar(255)          # rig file subpath
        """

    def make(self, key):
        log.info('BehaviorIngest.make(): key: {key}'.format(key=key))
        rigpaths = [p for p in RigDataPath().fetch(order_by='rig_data_path')
                    if 'RRig' in p['rig']] # change between TRig and RRig

        subject_id = key['subject_id']
        h2o = (lab.WaterRestriction() & {'subject_id': subject_id}).fetch1('water_restriction_number')
        date = key['session_date']
        datestr = date.strftime('%Y%m%d')
        log.debug('h2o: {h2o}, date: {d}'.format(h2o=h2o, d=datestr))

        # session record key
        skey = {}
        skey['subject_id'] = subject_id
        skey['session_date'] = date
        skey['username'] = 'daveliu' # username has to be changed

        # e.g: dl7/TW_autoTrain/Session Data/dl7_TW_autoTrain_20180104_132813.mat
        #         # p.split('/foo/bar')[1]
        for rp in rigpaths:
            root = rp['rig_data_path']
            path = root
            path = os.path.join(path, h2o)
#            path = os.path.join(path, 'TW_autoTrain')
            path = os.path.join(path, 'tw2')
            path = os.path.join(path, 'Session Data')
            path = os.path.join(
#                path, '{h2o}_TW_autoTrain_{d}*.mat'.format(h2o=h2o, d=datestr)) # earlier program protocol
                path, '{h2o}_tw2_{d}*.mat'.format(h2o=h2o, d=datestr)) # later program protocol

            log.debug('rigpath {p}'.format(p=path))

            matches = glob.glob(path)
            if len(matches):
                log.debug('found files, this is the rig')
                skey['rig'] = rp['rig']
                break
            else:
                log.info('no file matches found in {p}'.format(p=path))

        if not len(matches):
            log.warning('no file matches found for {h2o} / {d}'.format(
                h2o=h2o, d=datestr))
            return

        #
        # Find files & Check for split files
        # XXX: not checking rig.. 2+ sessions on 2+ rigs possible for date?
        #

        if len(matches) > 1:
            log.warning('split session case detected for {h2o} on {date}'
                        .format(h2o=h2o, date=date))

        # session:date relationship is 1:1; skip if we have a session
        if experiment.Session() & skey:
            log.warning("Warning! session exists for {h2o} on {d}".format(
                h2o=h2o, d=date))
            return

        #
        # Extract trial data from file(s) & prepare trial loop
        #

        trials = zip()

        trial = namedtuple(  # simple structure to track per-trial vars
            'trial', ('ttype', 'settings', 'state_times', 'state_names',
                      'state_data', 'event_data', 'event_times'))

        for f in matches:

            if os.stat(f).st_size/1024 < 100:
                log.info('skipping file {f} - too small'.format(f=f))
                continue

            mat = spio.loadmat(f, squeeze_me=True)
            SessionData = mat['SessionData'].flatten()

            AllTrialTypes = SessionData['TrialTypes'][0]
            AllTrialSettings = SessionData['TrialSettings'][0]

            RawData = SessionData['RawData'][0].flatten()
            AllStateNames = RawData['OriginalStateNamesByNumber'][0]
            AllStateData = RawData['OriginalStateData'][0]
            AllEventData = RawData['OriginalEventData'][0]
            AllStateTimestamps = RawData['OriginalStateTimestamps'][0]
            AllEventTimestamps = RawData['OriginalEventTimestamps'][0]

            # verify trial-related data arrays are all same length
            assert(all((x.shape[0] == AllStateTimestamps.shape[0] for x in
                        (AllTrialTypes, AllTrialSettings, AllStateNames,
                         AllStateData, AllEventData, AllEventTimestamps))))

            z = zip(AllTrialTypes, AllTrialSettings, AllStateTimestamps,
                    AllStateNames, AllStateData, AllEventData,
                    AllEventTimestamps)

            trials = chain(trials, z)  # concatenate the files

        trials = list(trials)

        # all files were internally invalid or size < 100k
        if not trials:
            log.warning('skipping date {d}, no valid files'.format(d=date))

        #
        # Trial data seems valid; synthesize session id & add session record
        # XXX: note - later breaks can result in Sessions without valid trials
        #

        log.debug('synthesizing session ID')
        session = (dj.U().aggr(experiment.Session() & {'subject_id': subject_id},
                               n='max(session)').fetch1('n') or 0) + 1
        log.info('generated session id: {session}'.format(session=session))
        skey['session'] = session
        key = dict(key, **skey)

        log.debug('BehaviorIngest.make(): adding session record')
        experiment.Session().insert1(skey)

        #
        # Actually load the per-trial data
        #
        log.info('BehaviorIngest.make(): trial parsing phase')

        # lists of various records for batch-insert
        rows = {k: list() for k in ('trial', 'behavior_trial', 'trial_note',
                                    'trial_event', 'action_event')}

        i = -1
        for t in trials:

            #
            # Misc
            #

            t = trial(*t)  # convert list of items to a 'trial' structure
            i += 1  # increment trial counter

            log.info('BehaviorIngest.make(): parsing trial {i}'.format(i=i))

            # covert state data names into a lookup dictionary
            #
            # names (seem to be? are?):
            #
            # Trigtrialstart
            # PreSamplePeriod
            # SamplePeriod
            # DelayPeriod
            # EarlyLickDelay
            # EarlyLickSample
            # ResponseCue
            # GiveLeftDrop
            # GiveRightDrop
            # GiveLeftDropShort
            # GiveRightDropShort
            # AnswerPeriod
            # Reward
            # RewardConsumption
            # NoResponse
            # TimeOut
            # StopLicking
            # StopLickingReturn
            # TrialEnd

            states = {k: (v+1) for v, k in enumerate(t.state_names)}
            required_states = ('PreSamplePeriod', 'SamplePeriod',
                               'DelayPeriod', 'ResponseCue', 'StopLicking', 'TrialEnd')

            missing = list(k for k in required_states if k not in states)

            if len(missing):
                log.info('skipping trial {i}; missing {m}'
                         .format(i=i, m=missing))
                continue

            gui = t.settings['GUI'].flatten()

            # ProtocolType - only ingest protocol >= 3
            #
            # 1 Water-Valve-Calibration 2 Licking 3 Autoassist
            # 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed
            #

            if 'ProtocolType' not in gui.dtype.names:
                log.info('skipping trial {i}; protocol undefined'
                         .format(i=i))
                continue

            protocol_type = gui['ProtocolType'][0]
            if gui['ProtocolType'][0] < 3:
                log.info('skipping trial {i}; protocol {n} < 3'
                         .format(i=i, n=gui['ProtocolType'][0]))
                continue

            #
            # Top-level 'Trial' record
            #

            tkey = dict(skey)
            startindex = np.where(t.state_data == states['PreSamplePeriod'])[0]

            # should be only end of 1st StopLicking;
            # rest of data is irrelevant w/r/t separately ingested ephys
            endindex = np.where(t.state_data == states['StopLicking'])[0]

            log.debug('states\n' + str(states))
            log.debug('state_data\n' + str(t.state_data))
            log.debug('startindex\n' + str(startindex))
            log.debug('endendex\n' + str(endindex))
             
            if not(len(startindex) and len(endindex)):
                log.info('skipping trial {i}: start/end index error: {s}/{e}'.format(i=i,s=str(startindex), e=str(endindex)))
                continue
                   
            try:    
                tkey['trial'] = i
                tkey['trial_uid'] = i # Arseny has unique id to identify some trials
                tkey['start_time'] = t.state_times[startindex][0]
            except IndexError:
                log.info('skipping trial {i}: error indexing {s}/{e} into {t}'.format(i=i,s=str(startindex), e=str(endindex), t=str(t.state_times)))
                continue
               
            log.debug('BehaviorIngest.make(): Trial().insert1')  # TODO msg
            log.debug('tkey' + str(tkey))
            rows['trial'].append(tkey)

            #
            # Specific BehaviorTrial information for this trial
            #

            bkey = dict(tkey)
            bkey['task'] = 'audio delay'
            bkey['task_protocol'] = 1

            # determine trial instruction
            trial_instruction = 'left'

            if gui['Reversal'][0] == 1:
                if t.ttype == 1:
                    trial_instruction = 'left'
                elif t.ttype == 0:
                    trial_instruction = 'right'
            elif gui['Reversal'][0] == 2:
                if t.ttype == 1:
                    trial_instruction = 'right'
                elif t.ttype == 0:
                    trial_instruction = 'left'

            bkey['trial_instruction'] = trial_instruction

            # determine early lick
            early_lick = 'no early'

            if (protocol_type >= 5
                and 'EarlyLickDelay' in states
                    and np.any(t.state_data == states['EarlyLickDelay'])):
                    early_lick = 'early'
            if (protocol_type > 5
                and ('EarlyLickSample' in states
                     and np.any(t.state_data == states['EarlyLickSample']))):
                    early_lick = 'early'

            bkey['early_lick'] = early_lick

            # determine outcome
            outcome = 'ignore'

            if ('Reward' in states
                    and np.any(t.state_data == states['Reward'])):
                outcome = 'hit'
            elif ('TimeOut' in states
                    and np.any(t.state_data == states['TimeOut'])):
                outcome = 'miss'
            elif ('NoResponse' in states
                    and np.any(t.state_data == states['NoResponse'])):
                outcome = 'ignore'

            bkey['outcome'] = outcome

            # add behavior record
            log.debug('BehaviorIngest.make(): BehaviorTrial()')
            rows['behavior_trial'].append(bkey)

            #
            # Add 'protocol' note
            #

            nkey = dict(tkey)
            nkey['trial_note_type'] = 'protocol #'
            nkey['trial_note'] = str(protocol_type)

            log.debug('BehaviorIngest.make(): TrialNote().insert1')
            rows['trial_note'].append(nkey)

            #
            # Add 'autolearn' note
            #

            nkey = dict(tkey)
            nkey['trial_note_type'] = 'autolearn'
            nkey['trial_note'] = str(gui['Autolearn'][0])
            rows['trial_note'].append(nkey)
            
            #pdb.set_trace()
            #
            # Add 'bitcode' note
            #
            if 'randomID' in gui.dtype.names:
                nkey = dict(tkey)
                nkey['trial_note_type'] = 'bitcode'
                nkey['trial_note'] = str(gui['randomID'][0])
                rows['trial_note'].append(nkey)

            #
            # Add presample event
            #

            ekey = dict(tkey)
            sampleindex = np.where(t.state_data == states['SamplePeriod'])[0]

            ekey['trial_event_type'] = 'presample'
            ekey['trial_event_time'] = t.state_times[startindex][0]
            ekey['duration'] = (t.state_times[sampleindex[0]]
                                - t.state_times[startindex])[0]

            log.debug('BehaviorIngest.make(): presample')
            rows['trial_event'].append(ekey)

            #
            # Add 'go' event
            #

            ekey = dict(tkey)
            responseindex = np.where(t.state_data == states['ResponseCue'])[0]

            ekey['trial_event_type'] = 'go'
            ekey['trial_event_time'] = t.state_times[responseindex][0]
            ekey['duration'] = gui['AnswerPeriod'][0]

            log.debug('BehaviorIngest.make(): go')
            rows['trial_event'].append(ekey)

            #
            # Add other 'sample' events
            #

            log.debug('BehaviorIngest.make(): sample events')
            for s in sampleindex:  # in protocol > 6 ~-> n>1
                # todo: batch events
                ekey = dict(tkey)
                ekey['trial_event_type'] = 'sample'
                ekey['trial_event_time'] = t.state_times[s]
                ekey['duration'] = gui['SamplePeriod'][0]
                rows['trial_event'].append(ekey)

            #
            # Add 'delay' events
            #

            delayindex = np.where(t.state_data == states['DelayPeriod'])[0]

            log.debug('BehaviorIngest.make(): delay events')
            for d in delayindex:  # protocol > 6 ~-> n>1
                # todo: batch events
                ekey = dict(tkey)
                ekey['trial_event_type'] = 'delay'
                ekey['trial_event_time'] = t.state_times[d]
                ekey['duration'] = gui['DelayPeriod'][0]
                rows['trial_event'].append(ekey)
                
            #
            # Add lick events
            #

            lickleft = np.where(t.event_data == 69)[0]
            log.debug('... lickleft: {r}'.format(r=str(lickleft)))

            if len(lickleft):
                [rows['action_event'].append(
                    dict(**tkey, action_event_type='left lick',
                         action_event_time=t.event_times[l]))
                 for l in lickleft]

            lickright = np.where(t.event_data == 70)[0]
            log.debug('... lickright: {r}'.format(r=str(lickright)))

            if len(lickright):
                [rows['action_event'].append(
                    dict(**tkey, action_event_type='right lick',
                         action_event_time=t.event_times[r]))
                    for r in lickright]

            # end of trial loop.

        log.info('BehaviorIngest.make(): bulk insert phase')

        log.info('BehaviorIngest.make(): ... experiment.Session.Trial')
        experiment.SessionTrial().insert(
            rows['trial'], ignore_extra_fields=True)

        log.info('BehaviorIngest.make(): ... experiment.BehaviorTrial')
        experiment.BehaviorTrial().insert(
            rows['behavior_trial'], ignore_extra_fields=True)

        log.info('BehaviorIngest.make(): ... experiment.TrialNote')
        experiment.TrialNote().insert(
            rows['trial_note'], ignore_extra_fields=True)

        log.info('BehaviorIngest.make(): ... experiment.TrialEvent')
        experiment.TrialEvent().insert(
            rows['trial_event'], ignore_extra_fields=True)

        log.info('BehaviorIngest.make(): ... experiment.ActionEvent')
        experiment.ActionEvent().insert(
            rows['action_event'], ignore_extra_fields=True)

        log.info('BehaviorIngest.make(): saving ingest {d}'.format(d=key))
        self.insert1(key, ignore_extra_fields=True)

        BehaviorIngest.BehaviorFile().insert(
            (dict(key, behavior_file=f.split(root)[1]) for f in matches),
            ignore_extra_fields=True)
