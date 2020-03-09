#! /usr/bin/env python

import os
import logging
import re
import math
import pathlib

from datetime import date
from datetime import datetime
from collections import namedtuple

import scipy.io as spio
import numpy as np
import warnings

import datajoint as dj

from pipeline import lab
from pipeline import experiment
from .. import get_schema_name

schema = dj.schema(get_schema_name('ingest_behavior'))

warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)


# ================ PHOTOSTIM PROTOCOL ===============
photostim_duration = 0.5  # (s)
skull_ref = 'Bregma'
photostims = {4: {'photo_stim': 4, 'photostim_device': 'OBIS470', 'duration': photostim_duration,
                  'locations': [{'skull_reference': skull_ref, 'brain_area': 'ALM',
                                 'ap_location': 2500, 'ml_location': -1500, 'depth': 0,
                                 'theta': 15, 'phi': 15}]},
              5: {'photo_stim': 5, 'photostim_device': 'OBIS470', 'duration': photostim_duration,
                  'locations': [{'skull_reference': skull_ref, 'brain_area': 'ALM',
                                 'ap_location': 2500, 'ml_location': 1500, 'depth': 0,
                                 'theta': 15, 'phi': 15}]},
              6: {'photo_stim': 6, 'photostim_device': 'OBIS470', 'duration': photostim_duration,
                  'locations': [{'skull_reference': skull_ref, 'brain_area': 'ALM',
                                 'ap_location': 2500, 'ml_location': -1500, 'depth': 0,
                                 'theta': 15, 'phi': 15},
                                {'skull_reference': skull_ref, 'brain_area': 'ALM',
                                 'ap_location': 2500, 'ml_location': 1500, 'depth': 0,
                                 'theta': 15, 'phi': 15}
                                ]}}


def get_behavior_paths():
    '''
    retrieve behavior rig paths from dj.config
    config should be in dj.config of the format:

      dj.config = {
        ...,
        'custom': {
          'behavior_data_paths':
            [
                ["RRig", "/path/string", 0],
                ["RRig2", "/path2/string2", 1]
            ],
        }
        ...
      }

    where 'behavior_data_paths' is a list of multiple possible path for behavior data, each in format:
    [rig name, rig full path, search order]
    '''

    paths = dj.config.get('custom', {}).get('behavior_data_paths', None)
    if paths is None:
        raise ValueError("Missing 'behavior_data_paths' in dj.config['custom']")

    return sorted(paths, key=lambda x: x[-1])


@schema
class BehaviorIngest(dj.Imported):
    definition = """
    -> experiment.Session
    """

    class BehaviorFile(dj.Part):
        ''' files in rig-specific storage '''
        definition = """
        -> BehaviorIngest
        behavior_file:              varchar(255)          # behavior file name
        """

    class CorrectedTrialEvents(dj.Part):
        ''' TrialEvents containing auto-corrected data '''
        definition = """
        -> BehaviorIngest
        -> experiment.TrialEvent
        """

    @staticmethod
    def get_session_user():
        '''
        Determine desired 'session user' for a session.

        - 1st, try dj.config['custom']['session.user']
        - 2nd, try dj.config['database.user']
        - else, use 'unknown'

        TODO: multi-user / bulk ingest support
        '''
        session_user = dj.config.get('custom', {}).get('session.user', None)

        session_user = (dj.config.get('database.user')
                        if not session_user else session_user)

        if len(lab.Person() & {'username': session_user}):
            return session_user
        else:
            return 'unknown'

    @property
    def key_source(self):

        # 2 letters, anything, _, anything, 8 digits, _, 6 digits, .mat
        # where:
        # (2 letters, anything): water restriction
        # (anything): task name
        # (8 digits): date YYYYMMDD
        # (6 digits): time HHMMSS

        rexp = '^[a-zA-Z]{2}.*_.*_[0-9]{8}_[0-9]{6}.mat$'

        # water_restriction_number -> subject
        h2os = {k: v for k, v in zip(*lab.WaterRestriction().fetch(
            'water_restriction_number', 'subject_id'))}

        def buildrec(rig, rigpath, root, f):

            if not re.match(rexp, f):
                log.debug("{f} skipped - didn't match rexp".format(f=f))
                return

            log.debug('found file {f}'.format(f=f))

            fullpath = pathlib.Path(root, f)
            subpath = fullpath.relative_to(rigpath)

            fsplit = subpath.stem.split('_')
            h2o = fsplit[0]
            ymd = fsplit[-2:-1][0]

            if h2o not in h2os:
                log.warning('{f} skipped - no animal for {h2o}'.format(
                    f=f, h2o=h2o))
                return

            animal = h2os[h2o]

            log.debug('animal is {animal}'.format(animal=animal))

            return {
                'subject_id': animal,
                'session_date': date(
                    int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])),
                'rig': rig,
                'rig_data_path': rigpath.as_posix(),
                'subpath': subpath.as_posix()
            }

        recs = []
        found = set()
        known = set(BehaviorIngest.BehaviorFile().fetch('behavior_file'))
        rigs = get_behavior_paths()

        for (rig, rigpath, _) in rigs:
            rigpath = pathlib.Path(rigpath)

            log.info('RigDataFile.make(): traversing {}'.format(rigpath))
            for root, dirs, files in os.walk(rigpath):
                log.debug('RigDataFile.make(): entering {}'.format(root))
                for f in files:
                    log.debug('RigDataFile.make(): visiting {}'.format(f))
                    r = buildrec(rig, rigpath, root, f)
                    if not r:
                        continue
                    if f in set.union(known, found):
                        log.info('skipping already ingested file {}'.format(
                            r['subpath']))
                    else:
                        found.add(f)  # block duplicate path conf
                        recs.append(r)

        return recs

    def populate(self, *args, **kwargs):
        for k in self.key_source:
            with dj.conn().transaction:
                self.make(k)

    def make(self, key):
        log.info('BehaviorIngest.make(): key: {key}'.format(key=key))

        subject_id = key['subject_id']
        h2o = (lab.WaterRestriction() & {'subject_id': subject_id}).fetch1(
            'water_restriction_number')

        ymd = key['session_date']
        datestr = ymd.strftime('%Y%m%d')
        log.info('h2o: {h2o}, date: {d}'.format(h2o=h2o, d=datestr))

        # session record key
        skey = {}
        skey['subject_id'] = subject_id
        skey['session_date'] = ymd
        skey['username'] = self.get_session_user()
        skey['rig'] = key['rig']

        # File paths conform to the pattern:
        # dl7/TW_autoTrain/Session Data/dl7_TW_autoTrain_20180104_132813.mat
        # which is, more generally:
        # {h2o}/{training_protocol}/Session Data/{h2o}_{training protocol}_{YYYYMMDD}_{HHMMSS}.mat

        path = pathlib.Path(key['rig_data_path'], key['subpath'])

        if experiment.Session() & skey:
            log.info("note: session exists for {h2o} on {d}".format(
                h2o=h2o, d=ymd))

        if os.stat(path).st_size/1024 < 1000:
            log.info('skipping file {} - too small'.format(path))
            return

        log.debug('loading file {}'.format(path))

        mat = spio.loadmat(path, squeeze_me=True, struct_as_record=False)
        SessionData = mat['SessionData']

        # parse session datetime
        session_datetime_str = str('').join((str(SessionData.Info.SessionDate), ' ',
                                             str(SessionData.Info.SessionStartTime_UTC)))
        session_datetime = datetime.strptime(
            session_datetime_str, '%d-%b-%Y %H:%M:%S')

        AllTrialTypes = SessionData.TrialTypes
        AllTrialSettings = SessionData.TrialSettings
        AllTrialStarts = SessionData.TrialStartTimestamp
        AllTrialStarts = AllTrialStarts - AllTrialStarts[0]  # real 1st trial

        RawData = SessionData.RawData
        AllStateNames = RawData.OriginalStateNamesByNumber
        AllStateData = RawData.OriginalStateData
        AllEventData = RawData.OriginalEventData
        AllStateTimestamps = RawData.OriginalStateTimestamps
        AllEventTimestamps = RawData.OriginalEventTimestamps

        AllRawEvents = SessionData.RawEvents.Trial

        # verify trial-related data arrays are all same length
        assert(all((x.shape[0] == AllStateTimestamps.shape[0] for x in
                    (AllTrialTypes, AllTrialSettings,
                     AllStateNames, AllStateData, AllEventData,
                     AllEventTimestamps, AllTrialStarts, AllTrialStarts, AllRawEvents))))

        # AllStimTrials optional special case
        if 'StimTrials' in SessionData._fieldnames:
            log.debug('StimTrials detected in session - will include')
            AllStimTrials = SessionData.StimTrials
            assert(AllStimTrials.shape[0] == AllStateTimestamps.shape[0])
        else:
            log.debug('StimTrials not detected in session - will skip')
            AllStimTrials = np.array([
                None for _ in enumerate(range(AllStateTimestamps.shape[0]))])

        # AllFreeTrials optional special case
        if 'FreeTrials' in SessionData._fieldnames:
            log.debug('FreeTrials detected in session - will include')
            AllFreeTrials = SessionData.FreeTrials
            assert(AllFreeTrials.shape[0] == AllStateTimestamps.shape[0])
        else:
            log.debug('FreeTrials not detected in session - synthesizing')
            AllFreeTrials = np.zeros(AllStateTimestamps.shape[0],
                                     dtype=np.uint8)

        # Photostim Period: early-delay, late-delay (default is early-delay)
        # Infer from filename for now, only applicable to Susu's sessions (i.e. "SC" in h2o)
        # If RecordingRig3, then 'late-delay'
        photostim_period = 'early-delay'
        rig_name = re.search('Recording(Rig\d)_', path.as_posix())
        if re.match('SC', h2o) and rig_name:
            rig_name = rig_name.groups()[0]
            if rig_name == "Rig3":
                photostim_period = 'late-delay'
        log.info('Photostim Period: {}'.format(photostim_period))

        trials = list(zip(AllTrialTypes, AllStimTrials, AllFreeTrials,
                          AllTrialSettings, AllStateTimestamps, AllStateNames,
                          AllStateData, AllEventData, AllEventTimestamps,
                          AllTrialStarts, AllRawEvents))

        if not trials:
            log.warning('skipping date {d}, no valid files'.format(d=date))
            return

        #
        # Trial data seems valid; synthesize session id & add session record
        # XXX: note - later breaks can result in Sessions without valid trials
        #

        assert skey['session_date'] == session_datetime.date()

        skey['session_date'] = session_datetime.date()
        skey['session_time'] = session_datetime.time()

        log.debug('synthesizing session ID')
        session = (dj.U().aggr(experiment.Session()
                               & {'subject_id': subject_id},
                               n='max(session)').fetch1('n') or 0) + 1

        log.info('generated session id: {session}'.format(session=session))
        skey['session'] = session
        key = dict(key, **skey)

        #
        # Actually load the per-trial data
        #
        log.info('BehaviorIngest.make(): trial parsing phase')

        # lists of various records for batch-insert
        rows = {k: list() for k in ('trial', 'behavior_trial', 'trial_note',
                                    'trial_event', 'corrected_trial_event',
                                    'action_event', 'photostim',
                                    'photostim_location', 'photostim_trial',
                                    'photostim_trial_event')}

        trial = namedtuple(  # simple structure to track per-trial vars
            'trial', ('ttype', 'stim', 'free', 'settings', 'state_times',
                      'state_names', 'state_data', 'event_data',
                      'event_times', 'trial_start', 'trial_raw_events'))

        i = 0  # trial numbering starts at 1
        for t in trials:

            #
            # Misc
            #

            t = trial(*t)  # convert list of items to a 'trial' structure
            i += 1  # increment trial counter

            log.debug('BehaviorIngest.make(): parsing trial {i}'.format(i=i))

            # covert state data names into a lookup dictionary
            #
            # names (seem to be? are?):
            #
            # Trigtrialstart, PreSamplePeriod, SamplePeriod, DelayPeriod
            # EarlyLickDelay, EarlyLickSample, ResponseCue, GiveLeftDrop
            # GiveRightDrop, GiveLeftDropShort, GiveRightDropShort
            # AnswerPeriod, Reward, RewardConsumption, NoResponse
            # TimeOut, StopLicking, StopLickingReturn, TrialEnd
            #

            states = {k: (v+1) for v, k in enumerate(t.state_names)}
            required_states = ('PreSamplePeriod', 'SamplePeriod',
                               'DelayPeriod', 'ResponseCue', 'StopLicking',
                               'TrialEnd')

            missing = list(k for k in required_states if k not in states)

            if len(missing):
                log.warning('skipping trial {i}; missing {m}'
                            .format(i=i, m=missing))
                continue

            gui = t.settings.GUI

            # ProtocolType - only ingest protocol >= 3
            #
            # 1 Water-Valve-Calibration 2 Licking 3 Autoassist
            # 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed
            #

            if 'ProtocolType' not in gui._fieldnames:
                log.warning('skipping trial {i}; protocol undefined'
                            .format(i=i))
                continue

            protocol_type = gui.ProtocolType
            if gui.ProtocolType < 3:
                log.warning('skipping trial {i}; protocol {n} < 3'
                            .format(i=i, n=gui.ProtocolType))
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
            log.debug('endindex\n' + str(endindex))

            if not(len(startindex) and len(endindex)):
                log.warning('skipping {}: start/end mismatch: {}/{}'.format(
                    i, str(startindex), str(endindex)))
                continue

            try:
                tkey['trial'] = i
                tkey['trial_uid'] = i
                tkey['start_time'] = t.trial_start
                tkey['stop_time'] = t.trial_start + t.state_times[endindex][0]
            except IndexError:
                log.warning('skipping {}: IndexError: {}/{} -> {}'.format(
                    i, str(startindex), str(endindex), str(t.state_times)))
                continue

            log.debug('tkey' + str(tkey))
            rows['trial'].append(tkey)

            #
            # Specific BehaviorTrial information for this trial
            #

            bkey = dict(tkey)
            bkey['task'] = 'audio delay'  # hard-coded here
            bkey['task_protocol'] = 1     # hard-coded here

            # determine trial instruction
            trial_instruction = 'left'    # hard-coded here

            if gui.Reversal == 1:
                if t.ttype == 1:
                    trial_instruction = 'left'
                elif t.ttype == 0:
                    trial_instruction = 'right'
            elif gui.Reversal == 2:
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
            if (protocol_type >= 5
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

            # Determine free/autowater (Autowater 1 == enabled, 2 == disabled)
            bkey['auto_water'] = True if gui.Autowater == 1 else False
            bkey['free_water'] = t.free

            rows['behavior_trial'].append(bkey)

            #
            # Add 'protocol' note
            #
            nkey = dict(tkey)
            nkey['trial_note_type'] = 'protocol #'
            nkey['trial_note'] = str(protocol_type)
            rows['trial_note'].append(nkey)

            #
            # Add 'autolearn' note
            #
            nkey = dict(tkey)
            nkey['trial_note_type'] = 'autolearn'
            nkey['trial_note'] = str(gui.Autolearn)
            rows['trial_note'].append(nkey)

            #
            # Add 'bitcode' note
            #
            if 'randomID' in gui._fieldnames:
                nkey = dict(tkey)
                nkey['trial_note_type'] = 'bitcode'
                nkey['trial_note'] = str(gui.randomID)
                rows['trial_note'].append(nkey)

            # ==== TrialEvents ====
            trial_event_types = [('PreSamplePeriod', 'presample'),
                                 ('SamplePeriod', 'sample'),
                                 ('DelayPeriod', 'delay'),
                                 ('ResponseCue', 'go'),
                                 ('TrialEnd', 'trialend')]

            for tr_state, trial_event_type in trial_event_types:
                tr_events = getattr(t.trial_raw_events.States, tr_state)
                tr_events = np.array([tr_events]) if tr_events.ndim < 2 else tr_events
                for (s_start, s_end) in tr_events:
                    ekey = dict(tkey)
                    ekey['trial_event_id'] = len(rows['trial_event'])
                    ekey['trial_event_type'] = trial_event_type
                    ekey['trial_event_time'] = s_start
                    ekey['duration'] = s_end - s_start
                    rows['trial_event'].append(ekey)

            # ==== ActionEvents ====

            #
            # Add lick events
            #

            lickleft = np.where(t.event_data == 69)[0]
            log.debug('... lickleft: {r}'.format(r=str(lickleft)))

            action_event_count = len(rows['action_event'])
            if len(lickleft):
                [rows['action_event'].append(
                    dict(tkey, action_event_id=action_event_count+idx,
                         action_event_type='left lick',
                         action_event_time=t.event_times[l]))
                 for idx, l in enumerate(lickleft)]

            lickright = np.where(t.event_data == 71)[0]
            log.debug('... lickright: {r}'.format(r=str(lickright)))

            action_event_count = len(rows['action_event'])
            if len(lickright):
                [rows['action_event'].append(
                    dict(tkey, action_event_id=action_event_count+idx,
                         action_event_type='right lick',
                         action_event_time=t.event_times[r]))
                    for idx, r in enumerate(lickright)]

            # ==== PhotostimEvents ====

            #
            # Photostim Events
            #

            if t.stim:
                log.debug('BehaviorIngest.make(): t.stim == {}'.format(t.stim))
                rows['photostim_trial'].append(tkey)
                if photostim_period == 'early-delay':  # same as the delay-onset
                    delay_periods = t.trial_raw_events.States.DelayPeriod
                    delay_periods = np.array([delay_periods]) if delay_periods.ndim < 2 else delay_periods
                    stim_onset = delay_periods[-1][0]
                elif photostim_period == 'late-delay':  # 0.5 sec prior to the go-cue
                    stim_onset = t.trial_raw_events.States.ResponseCue[0] - 0.5

                rows['photostim_trial_event'].append(
                    dict(tkey,
                         photo_stim=t.stim,
                         photostim_event_id=len(
                             rows['photostim_trial_event']),
                         photostim_event_time=stim_onset,
                         power=5.5))

            # end of trial loop.

        # Session Insertion

        log.info('BehaviorIngest.make(): adding session record')
        experiment.Session().insert1(skey)

        # Behavior Insertion

        log.info('BehaviorIngest.make(): bulk insert phase')

        log.info('BehaviorIngest.make(): saving ingest {d}'.format(d=key))
        self.insert1(key, ignore_extra_fields=True, allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.Session.Trial')
        experiment.SessionTrial().insert(
            rows['trial'], ignore_extra_fields=True, allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.BehaviorTrial')
        experiment.BehaviorTrial().insert(
            rows['behavior_trial'], ignore_extra_fields=True,
            allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.TrialNote')
        experiment.TrialNote().insert(
            rows['trial_note'], ignore_extra_fields=True,
            allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.TrialEvent')
        experiment.TrialEvent().insert(
            rows['trial_event'], ignore_extra_fields=True,
            allow_direct_insert=True, skip_duplicates=True)

        log.info('BehaviorIngest.make(): ... experiment.ActionEvent')
        experiment.ActionEvent().insert(
            rows['action_event'], ignore_extra_fields=True,
            allow_direct_insert=True)

        # Photostim Insertion

        photostim_ids = np.unique(
            [r['photo_stim'] for r in rows['photostim_trial_event']])

        unknown_photostims = np.setdiff1d(
            photostim_ids, list(photostims.keys()))

        if unknown_photostims:
            raise ValueError(
                'Unknown photostim protocol: {}'.format(unknown_photostims))

        if photostim_ids.size > 0:
            log.info('BehaviorIngest.make(): ... experiment.Photostim')
            for stim in photostim_ids:
                experiment.Photostim.insert1(
                    dict(skey, **photostims[stim]), ignore_extra_fields=True)

                experiment.Photostim.PhotostimLocation.insert(
                    (dict(skey, **loc,
                          photo_stim=photostims[stim]['photo_stim'])
                     for loc in photostims[stim]['locations']),
                    ignore_extra_fields=True)

        log.info('BehaviorIngest.make(): ... experiment.PhotostimTrial')
        experiment.PhotostimTrial.insert(rows['photostim_trial'],
                                         ignore_extra_fields=True,
                                         allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.PhotostimTrialEvent')
        experiment.PhotostimEvent.insert(rows['photostim_trial_event'],
                                         ignore_extra_fields=True,
                                         allow_direct_insert=True)

        # Behavior Ingest Insertion

        log.info('BehaviorIngest.make(): ... BehaviorIngest.BehaviorFile')
        BehaviorIngest.BehaviorFile().insert1(
            dict(key, behavior_file=os.path.basename(key['subpath'])),
            ignore_extra_fields=True, allow_direct_insert=True)
