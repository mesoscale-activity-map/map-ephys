#! /usr/bin/env python

import os
import logging

from itertools import chain
from collections import namedtuple

from code import interact

import scipy.io as spio
import numpy as np

import datajoint as dj

import lab
import experiment


log = logging.getLogger(__name__)
schema = dj.schema(dj.config['ingest.database'], locals())


@schema
class RigDataPath(dj.Lookup):
    ''' rig storage locations '''
    # todo: cross platform path mapping needed?
    definition = """
    -> lab.Rig
    ---
    rig_data_path:             varchar(1024)           # rig data path
    """

    @property
    def contents(self):
        if 'rig_data_paths' in dj.config:  # for local testing
            return dj.config['rig_data_paths']

        return (('RRig', r'Z:\\MATLAB\Bpod Local\Data'), ('TRig1', r'R:\\Arduino\Bpod_Train1\Bpod Local\Data'),
                ('TRig2', r'Q:\\Users\labadmin\Documents\MATLAB\Bpod Local\Data'), ('TRig3', r'S:\\MATLAB\Bpod Local\Data'))


@schema
class RigDataFile(dj.Imported):
    ''' files in rig-specific storage '''
    definition = """
    -> RigDataPath
    rig_data_file:              varchar(255)          # rig file subpath
    """

    @property
    def key_source(self):
        return RigDataPath()

    def make(self, key):
        log.info('RigDataFile.make(): key:', key)
        rig, data_path = (RigDataPath() & {'rig': key['rig']}).fetch1().values()
        log.info('RigDataFile.make(): searching %s' % rig)

        initial = list(k['rig_data_file'] for k in
                       (self & key).fetch(as_dict=True))

        for root, dirs, files in os.walk(data_path):
            log.debug('RigDataFile.make(): traversing %s' % root)
            subpaths = list(os.path.join(root, f)
                            .split(data_path)[1].lstrip(os.path.sep)
                            for f in files if f.endswith('.mat')
                            and 'TW_autoTrain' in f)

            subpaths.sort()  # ascending dates help sequential session id

            self.insert(list((rig, f,) for f in subpaths if f not in initial))

    def populate(self):
        '''
        Overriding populate since presence of any rig_data_file
        will prevent that key to be given to Make.
        '''
        for k in self.key_source.fetch(as_dict=True):
            self.make(k)


@schema
class RigDataFileIngest(dj.Imported):
    definition = """
    -> RigDataFile
    ---
    -> experiment.Session
    """

    def make(self, key):

        #
        # Handle filename & Construct Session
        #

        rigdir = (RigDataPath() & key).fetch1('rig_data_path')
        rigfile = key['rig_data_file']
        filename = os.path.basename(rigfile)
        fullpath = os.path.join(rigdir, rigfile)

        log.debug('RigDataFileIngest.make(): {f}'
                  .format(f=dict(rigdir=rigdir, rigfile=rigfile,
                                 filename=filename, fullpath=fullpath)))

        # split files like 'dl7_TW_autoTrain_20171114_140357.mat'
        h2o, t1, t2, date, time = filename.split('.')[0].split('_')

        if not (lab.AnimalWaterRestriction() & {'water_restriction': h2o}): # hack to only ingest files with a water restriction number
            log.info('skipping file {f} - no water restriction #'.format(f=fullpath))
            return

        # lookup animal
        log.debug('looking up animal for {h2o}'.format(h2o=h2o))
        animal = (lab.AnimalWaterRestriction()
                  & {'water_restriction': h2o}).fetch1('animal')
        log.info('animal is {animal}'.format(animal=animal))

        # session record key
        skey = {}
        skey['animal'] = animal
        skey['session_date'] = date[0:4] + '-' + date[4:6] + '-' + date[6:8]
        skey['username'] = 'daveliu'
        skey['rig'] = key['rig']

        # session:date relationship is 1:1; skip if we have a session
        if experiment.Session() & skey:
            log.warning("Warning! session exists for {f}".format(f=rigfile))
            return

        #
        # Check for split files and prepare filelists
        # XXX: not querying by rig.. 2+ sessions on 2+ rigs possible?
        #

        # '%%' vs '%' due to datajoint-python/issues/376
        daily = (RigDataFile() & "rig_data_file like '%%{h2o}%%{date}%%'"
                 .format(h2o=h2o, date=date)).fetch('rig_data_file')

        daily = list(os.path.join(rigdir, x) for x in daily)

        if len(daily) > 1:
            log.warning('split session case detected for {h2o} on {date}'
                        .format(h2o=h2o, date=date))

        #
        # Extract trial data from file(s) & prepare trial loop
        #

        trials = zip()

        trial = namedtuple(  # simple structure to track per-trial vars
            'trial', ('ttype', 'settings', 'state_times', 'state_names',
                      'state_data', 'event_data', 'event_times'))

        for f in daily:
            # interact('dailyloop', local=locals())

            if os.stat(f).st_size/1024 < 500:
                log.info('skipping file {f} - too small'.format(f=fullpath))
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

            trials = chain(trials, z)

        trials = list(trials)

        # all files were invalid / size < 500k
        if not trials:
            log.warning('skipping date {d}, no valid files'.format(d=date))

        #
        # Trial data seems valid; synthesize session id & add session record
        # XXX: note - later breaks can result in Sessions without valid trials
        #

        log.debug('synthesizing session ID')
        session = (dj.U().aggr(experiment.Session() & {'animal': animal},
                               n='max(session)').fetch1('n') or 0) + 1
        log.info('generated session id: {session}'.format(session=session))
        skey['session'] = session
        key = dict(key, **skey)

        log.debug('ImportedSessionFileIngest.make(): adding session record')
        experiment.Session().insert1(skey)

        #
        # Actually load the per-trial data
        #

        i = -1
        for t in trials:

            #
            # Misc
            #

            t = trial(*t)  # convert list of items to a 'trial' structure
            i += 1  # increment trial counter

            log.info('ImportedSessionFileIngest.make(): trial {i}'.format(i=i))

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
                               'DelayPeriod', 'ResponseCue', 'TrialEnd')

            missing = list(k for k in required_states if k not in states)

            if len(missing):
                log.info('skipping trial {i}; missing {m}'
                         .format(i=i, m=missing))
                continue

            gui = t.settings['GUI'].flatten()

            # ProtocolType
            #
            # 1 Water-Valve-Calibration 2 Licking 3 Autoassist
            # 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed
            # only ingest protocol >= 5

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

            tkey['trial'] = i
            tkey['start_time'] = t.state_times[startindex][0]
            tkey['end_time'] = t.state_times[endindex][0]

            log.debug('ImportedSessionFileIngest.make(): Trial().insert1')
            log.debug('tkey' + str(tkey))
            experiment.Session.Trial().insert1(tkey, ignore_extra_fields=True)

            #
            # Specific BehaviorTrial information for this trial
            #

            bkey = dict(tkey)
            bkey['task'] = 'audio delay'

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
            log.debug('ImportedSessionFileIngest.make(): BehaviorTrial()')
            experiment.BehaviorTrial().insert1(bkey, ignore_extra_fields=True)

            #
            # Add 'protocol' note
            #

            nkey = dict(tkey)
            nkey['trial_note_type'] = 'protocol #'
            nkey['trial_note'] = str(protocol_type)

            log.debug('ImportedSessionFileIngest.make(): TrialNote().insert1')
            experiment.TrialNote().insert1(nkey, ignore_extra_fields=True)

            #
            # Add presample event
            #

            ekey = dict(tkey)
            sampleindex = np.where(t.state_data == states['SamplePeriod'])[0]

            ekey['trial_event_type'] = 'presample'
            ekey['trial_event_time'] = t.state_times[startindex][0]
            ekey['duration'] = (t.state_times[sampleindex[0]]
                                - t.state_times[startindex])[0]

            log.debug('ImportedSessionFileIngest.make(): presample')
            experiment.TrialEvent().insert1(ekey, ignore_extra_fields=True)

            #
            # Add 'go' event
            #

            ekey = dict(tkey)
            responseindex = np.where(t.state_data == states['ResponseCue'])[0]

            ekey['trial_event_type'] = 'go'
            ekey['trial_event_time'] = t.state_times[responseindex][0]
            ekey['duration'] = gui['AnswerPeriod'][0]

            log.debug('ImportedSessionFileIngest.make(): go')
            experiment.TrialEvent().insert1(ekey, ignore_extra_fields=True)

            #
            # Add other 'sample' events
            #

            log.debug('ImportedSessionFileIngest.make(): sample events')
            for s in sampleindex:  # in protocol > 6 ~-> n>1
                # todo: batch events
                ekey = dict(tkey)
                ekey['trial_event_type'] = 'sample'
                ekey['trial_event_time'] = t.state_times[s]
                ekey['duration'] = gui['SamplePeriod'][0]
                experiment.TrialEvent().insert1(ekey, ignore_extra_fields=True)

            #
            # Add 'delay' events
            #

            delayindex = np.where(t.state_data == states['DelayPeriod'])[0]

            log.debug('ImportedSessionFileIngest.make(): delay events')
            for d in delayindex:  # protocol > 6 ~-> n>1
                # todo: batch events
                ekey = dict(tkey)
                ekey['trial_event_type'] = 'delay'
                ekey['trial_event_time'] = t.state_times[d]
                ekey['duration'] = gui['DelayPeriod'][0]
                experiment.TrialEvent().insert1(ekey, ignore_extra_fields=True)

            #
            # Add lick events
            #

            lickleft = np.where(t.event_data == 69)[0]
            log.debug('... lickleft: {r}'.format(r=str(lickleft)))
            if len(lickleft):
                # TODO: is 'sample' the type?
                leftlicks = list(
                    (dict(**tkey,
                          action_event_type='left lick',
                          action_event_time=t.event_times[l])
                     for l in lickleft))

                experiment.ActionEvent().insert(
                    leftlicks, ignore_extra_fields=True)

            lickright = np.where(t.event_data == 70)[0]
            log.debug('... lickright: {r}'.format(r=str(lickright)))
            if len(lickright):
                rightlicks = list(
                    (dict(**tkey,
                          action_event_type='right lick',
                          action_event_time=t.event_times[r])
                     for r in lickright))

                experiment.ActionEvent().insert(
                    rightlicks, ignore_extra_fields=True)

            # end of trial loop.

        log.debug('RigDataFileIngest.make(): saving ingest {d}'.format(d=key))
        self.insert1(key, ignore_extra_fields=True)
