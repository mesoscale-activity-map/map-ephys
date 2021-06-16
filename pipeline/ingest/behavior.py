#! /usr/bin/env python

import os
import logging
import re
import pathlib

from datetime import date, datetime
from collections import namedtuple
import time as timer

import scipy.io as spio
import numpy as np
import pandas as pd
import decimal
import warnings
import datajoint as dj

from pybpodgui_api.models.project import Project as BPodProject
from . import util

from pipeline import lab, experiment
from pipeline import get_schema_name, dict_to_hash


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


@schema
class BehaviorIngest(dj.Imported):
    definition = """
    -> experiment.Session
    """

    class BehaviorFile(dj.Part):
        ''' files in rig-specific storage '''
        definition = """
        -> master
        behavior_file:              varchar(255)          # behavior file name
        """

    class CorrectedTrialEvents(dj.Part):
        ''' TrialEvents containing auto-corrected data '''
        definition = """
        -> BehaviorIngest
        -> experiment.TrialEvent
        """

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
        # 'populate' which won't require upstream tables
        # 'reserve_jobs' not parallel, overloaded to mean "don't exit on error"
        for k in self.key_source:
            try:
                with dj.conn().transaction:
                    self.make(k)
            except Exception as e:
                log.warning('session key {} error: {}'.format(k, repr(e)))
                if not kwargs.get('reserve_jobs', False):
                    raise

    def make(self, key):
        log.info('BehaviorIngest.make(): key: {key}'.format(key=key))

        # File paths conform to the pattern:
        # dl7/TW_autoTrain/Session Data/dl7_TW_autoTrain_20180104_132813.mat
        # which is, more generally:
        # {h2o}/{training_protocol}/Session Data/{h2o}_{training protocol}_{YYYYMMDD}_{HHMMSS}.mat

        path = pathlib.Path(key['rig_data_path'], key['subpath'])

        if os.stat(path).st_size/1024 < 1000:
            log.info('skipping file {} - too small'.format(path))
            return

        log.debug('loading file {}'.format(path))

        # Read from behavior file and parse all trial info (the heavy lifting here)
        skey, rows = BehaviorIngest._load(key, path)

        # Session Insertion

        log.info('BehaviorIngest.make(): adding session record')
        experiment.Session.insert1(skey)

        # Behavior Insertion

        log.info('BehaviorIngest.make(): bulk insert phase')

        log.info('BehaviorIngest.make(): saving ingest {d}'.format(d=key))
        self.insert1(key, ignore_extra_fields=True, allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.Session.Trial')
        experiment.SessionTrial.insert(
            rows['trial'], ignore_extra_fields=True, allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.BehaviorTrial')
        experiment.BehaviorTrial.insert(
            rows['behavior_trial'], ignore_extra_fields=True,
            allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.TrialNote')
        experiment.TrialNote.insert(
            rows['trial_note'], ignore_extra_fields=True,
            allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.TrialEvent')
        experiment.TrialEvent.insert(
            rows['trial_event'], ignore_extra_fields=True,
            allow_direct_insert=True, skip_duplicates=True)

        log.info('BehaviorIngest.make(): ... experiment.ActionEvent')
        experiment.ActionEvent.insert(
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
        BehaviorIngest.BehaviorFile.insert1(
            dict(key, behavior_file=os.path.basename(key['subpath'])),
            ignore_extra_fields=True, allow_direct_insert=True)

    @classmethod
    def _load(cls, key, path):
        """
        Method to load the behavior file (.mat), parse trial info and prepare for insertion
        (no table insertion is done here)

        :param key: session_key
        :param path: (str) filepath of the behavior file (.mat)
        :return: skey, rows
            + skey: session_key
            + rows: a dictionary containing all per-trial information to be inserted
        """
        path = pathlib.Path(path)

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
        skey['username'] = get_session_user()
        skey['rig'] = key['rig']

        mat = spio.loadmat(path.as_posix(), squeeze_me=True, struct_as_record=False)
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
            bkey['auto_water'] = gui.Autowater == 1 or np.any(t.settings.GaveFreeReward[:2])
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

                    if trial_event_type == 'delay':
                        this_trial_delay_duration = s_end - s_start

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

            if photostim_period == 'early-delay':
                valid_protocol = protocol_type == 5
            elif photostim_period == 'late-delay':
                valid_protocol = protocol_type > 4

            if t.stim and valid_protocol and gui.Autolearn == 4 and this_trial_delay_duration == 1.2:
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

        return skey, rows


@schema
class BehaviorBpodIngest(dj.Imported):
    definition = """
    -> experiment.Session
    """

    class BehaviorFile(dj.Part):
        ''' files in rig-specific storage '''
        definition = """
        -> master
        behavior_file:              varchar(255)          # behavior file name
        """
    
    water_port_name_mapper = {'left': 'L', 'right': 'R', 'middle': 'M'}
    
    @staticmethod
    def get_bpod_projects():
        projectdirs = dj.config.get('custom', {}).get('behavior_bpod', []).get('project_paths')
        # construct a list of BPod Projects
        projects = []
        for projectdir in projectdirs:
            projects.append(BPodProject())
            projects[-1].load(projectdir)
        return projects
    
    @property
    def key_source(self):
        key_source = []

        IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch(
            'water_restriction_number', 'subject_id'))}

        for subject_now, subject_id_now in IDs.items():
            meta_dir = dj.config.get('custom', {}).get('behavior_bpod', []).get('meta_dir')

            subject_csv = pathlib.Path(meta_dir) / '{}.csv'.format(subject_now)
            if subject_csv.exists():
                df_wr = pd.read_csv(subject_csv)
            else:
                log.info('No metadata csv found for {}'.format(subject_now))
                continue

            for r_idx, df_wr_row in df_wr.iterrows():
                # we use it when both start and end times are filled in and Water during training > 0; restriction, freewater and handling is skipped
                if (df_wr_row['Time'] and isinstance(df_wr_row['Time'], str)
                        and df_wr_row['Time-end'] and isinstance(df_wr_row['Time-end'], str)
                        and df_wr_row['Training type'] != 'restriction'
                        and df_wr_row['Training type'] != 'handling'
                        and df_wr_row['Training type'] != 'freewater'
                        and df_wr_row['Water during training'] > 0):

                    try:
                        date_now = datetime.strptime(df_wr_row.Date, '%Y-%m-%d').date()
                    except:
                        try:
                            date_now = datetime.strptime(df_wr_row.Date, '%Y/%m/%d').date()
                        except:
                            log.info('Unable to parse session date: {}. Skipping...'.format(df_wr_row.Date))
                            continue

                    if not (experiment.Session & {'subject_id': subject_id_now, 'session_date': date_now}):
                        key_source.append({'subject_id': subject_id_now,
                                           'session_date': date_now,
                                           'session_comment': str(df_wr_row['Notes']),
                                           'session_weight': df_wr_row['Weight'],
                                           'session_water_earned': df_wr_row['Water during training'],
                                           'session_water_extra': df_wr_row['Extra water']})

        return key_source

    def populate(self, *args, **kwargs):
        # Load project info (just once)
        log.info('------ Loading pybpod project -------')
        self.projects = self.get_bpod_projects()
        log.info('------------   Done! ----------------')

        # 'populate' which won't require upstream tables
        # 'reserve_jobs' not parallel, overloaded to mean "don't exit on error"                          
        for k in self.key_source:
            try:
                with dj.conn().transaction:
                    self.make(k)
            except Exception as e:
                log.warning('session key {} error: {}'.format(k, repr(e)))
                if not kwargs.get('reserve_jobs', False):
                    raise

    def make(self, key):
        log.info('----------------------\nBehaviorBpodIngest.make(): key: {key}'.format(key=key))

        subject_id_now = key['subject_id']
        subject_now = (lab.WaterRestriction() & {'subject_id': subject_id_now}).fetch1('water_restriction_number')
        date_now_str = key['session_date'].strftime('%Y%m%d')
        log.info('h2o: {h2o}, date: {d}'.format(h2o=subject_now, d=date_now_str))

        # ---- Ingest information for BPod projects ----
        sessions_now, session_start_times_now, experimentnames_now = [], [], []
        for proj in self.projects:  #
            exps = proj.experiments
            for exp in exps:
                stps = exp.setups
                for stp in stps:
                    for session in stp.sessions:
                        if (session.subjects and session.subjects[0].find(subject_now) > -1
                                and session.name.startswith(date_now_str)):
                            sessions_now.append(session)
                            session_start_times_now.append(session.started)
                            experimentnames_now.append(exp.name)
                            
        bpodsess_order = np.argsort(session_start_times_now)
        
        # --- Handle missing BPod session ---
        if len(bpodsess_order) == 0:
            log.error('BPod session not found!')
            return

        # ---- Concatenate bpod sessions (and corresponding trials) into one datajoint session ----
        tbls_2_insert = ('sess_trial', 'behavior_trial', 'trial_note',
                         'sess_block', 'sess_block_trial',
                         'trial_choice', 'trial_event', 'action_event',
                         'photostim', 'photostim_location', 'photostim_trial', 'photostim_trial_event',
                         'valve_setting', 'valve_open_dur', 'available_reward')

        # getting started
        concat_rows = {k: list() for k in tbls_2_insert}
        sess_key = None
        trial_num = 0  # trial numbering starts at 1

        for s_idx, session_idx in enumerate(bpodsess_order):
            session = sessions_now[session_idx]
            experiment_name = experimentnames_now[session_idx]
            csvfilename = (pathlib.Path(session.path) / (pathlib.Path(session.path).name + '.csv'))
            
            # ---- Special parsing for csv file ----
            log.info('Load session file(s) ({}/{}): {}'.format(s_idx + 1, len(bpodsess_order), csvfilename))
            df_behavior_session = util.load_and_parse_a_csv_file(csvfilename)
            
            # ---- Integrity check of the current bpodsess file ---
            # It must have at least one 'trial start' and 'trial end'
            trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRIAL') & (df_behavior_session['MSG'] == 'New trial')].index
            if not len(trial_start_idxs):
                log.info('No "trial start" for {}. Skipping...'.format(csvfilename))
                continue   # Make sure 'start' exists, otherwise move on to try the next bpodsess file if exists     
                   
            trial_end_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRANSITION') & (df_behavior_session['MSG'] == 'End')].index
            if not len(trial_end_idxs):
                log.info('No "trial end" for {}. Skipping...'.format(csvfilename))
                continue   # Make sure 'end' exists, otherwise move on to try the next bpodsess file if exists     
            
            # It must be a foraging session
            # extracting task protocol - hard-code implementation
            if 'foraging' in experiment_name.lower() or ('bari' in experiment_name.lower() and 'cohen' in experiment_name.lower()):
                if 'var:lickport_number' in df_behavior_session and df_behavior_session['var:lickport_number'][0] == 3:
                    task = 'foraging 3lp'
                    task_protocol = 101
                    lick_ports = ['left', 'right', 'middle']
                else:
                    task = 'foraging'
                    task_protocol = 100
                    lick_ports = ['left', 'right']
            else:
                log.info('ERROR: unhandled task name {}. Skipping...'.format(experiment_name))
                continue   # Make sure this is a foraging bpodsess, otherwise move on to try the next bpodsess file if exists 
                
            # ---- New session - construct a session key (from the first bpodsess that passes the integrity check) ----
            if sess_key is None:
                session_time = df_behavior_session['PC-TIME'][trial_start_idxs[0]]
                if session.setup_name.lower() in ['day1', 'tower-2', 'day2-7', 'day_1', 'real foraging']:
                    setupname = 'Training-Tower-2'
                elif session.setup_name.lower() in ['tower-3', 'tower-3beh', ' tower-3', '+', 'tower 3']:
                    setupname = 'Training-Tower-3'
                elif session.setup_name.lower() in ['tower-1']:
                    setupname = 'Training-Tower-1'
                elif session.setup_name.lower() in ['ephys_han']:
                    setupname = 'Ephys-Han'
                else:
                    log.info('ERROR: unhandled setup name {} (from {}). Skipping...'.format(session.setup_name, session.path))
                    continue   # Another integrity check here

                log.debug('synthesizing session ID')
                key['session'] = (dj.U().aggr(experiment.Session()
                                              & {'subject_id': subject_id_now},
                                              n='max(session)').fetch1('n') or 0) + 1
                sess_key = {**key,
                            'session_time': session_time.time(),
                            'username': df_behavior_session['experimenter'][0],
                            'rig': setupname}

            # ---- channel for water ports ----
            water_port_channels = {}
            for lick_port in lick_ports:
                chn_varname = 'var:WaterPort_{}_ch_in'.format(self.water_port_name_mapper[lick_port])
                if chn_varname not in df_behavior_session:
                    log.error('Bpod CSV KeyError: {} - Available columns: {}'.format(chn_varname, df_behavior_session.columns))
                    return
                water_port_channels[lick_port] = df_behavior_session[chn_varname][0]

            # ---- Ingestion of trials ----

            # extracting trial data
            session_start_time = datetime.combine(sess_key['session_date'], sess_key['session_time'])
            trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRIAL') & (df_behavior_session['MSG'] == 'New trial')].index
            trial_start_idxs -= 2 # To reflect the change that bitcode is moved before the "New trial" line
            trial_start_idxs = pd.Index([0]).append(trial_start_idxs[1:])  # so the random seed will be present
            trial_end_idxs = trial_start_idxs[1:].append(pd.Index([(max(df_behavior_session.index))]))
            # trial_end_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'END-TRIAL')].index
            prevtrialstarttime = np.nan
            blocknum_local_prev = np.nan

            # getting ready
            rows = {k: list() for k in tbls_2_insert}  # lists of various records for batch-insert

            for trial_start_idx, trial_end_idx in zip(trial_start_idxs, trial_end_idxs):
                df_behavior_trial = df_behavior_session[trial_start_idx:trial_end_idx + 1]

                # Trials without GoCue are skipped
                if not len(df_behavior_trial['PC-TIME'][(df_behavior_trial['MSG'] == 'GoCue') & (
                        df_behavior_trial['TYPE'] == 'TRANSITION')]):
                    continue
                
                # ---- session trial ----
                trial_num += 1  # increment trial number
                trial_uid = len(experiment.SessionTrial & {'subject_id': subject_id_now}) + 1
                trial_start_time = df_behavior_session['PC-TIME'][trial_start_idx].to_pydatetime() - session_start_time
                trial_stop_time = df_behavior_session['PC-TIME'][trial_end_idx].to_pydatetime() - session_start_time

                sess_trial_key = {**sess_key,
                                  'trial': trial_num,
                                  'trial_uid': trial_uid,
                                  'start_time': trial_start_time.total_seconds(),
                                  'stop_time': trial_stop_time.total_seconds()}
                rows['sess_trial'].append(sess_trial_key)

                # ---- session block ----
                if 'Block_number' in df_behavior_session:
                    if np.isnan(df_behavior_trial['Block_number'].to_list()[0]):
                        blocknum_local = 0 if np.isnan(blocknum_local_prev) else blocknum_local_prev
                    else:
                        blocknum_local = int(df_behavior_trial['Block_number'].to_list()[0]) - 1
                        blocknum_local_prev = blocknum_local

                    reward_probability = {}
                    for lick_port in lick_ports:
                        p_reward_varname = 'var:reward_probabilities_{}'.format(self.water_port_name_mapper[lick_port])
                        reward_probability[lick_port] = decimal.Decimal(
                            df_behavior_session[p_reward_varname][0][blocknum_local]).quantize(
                            decimal.Decimal('.001'))    # Note: Reward probabilities never changes during a **bpod** session

                    # determine if this is a new block: compare reward probability with the previous block
                    if rows['sess_block']:
                        itsanewblock = dict_to_hash(reward_probability) != dict_to_hash(rows['sess_block'][-1]['reward_probability'])
                    else:
                        itsanewblock = True

                    if itsanewblock:
                        all_blocks = [b['block'] for b in rows['sess_block'] + concat_rows['sess_block']]
                        block_num = (np.max(all_blocks) + 1 if all_blocks else 1)
                        rows['sess_block'].append({**sess_key,
                                                   'block': block_num,
                                                   'block_start_time': trial_start_time.total_seconds(),
                                                   'reward_probability': reward_probability})
                    else:
                        block_num = rows['sess_block'][-1]['block']

                    rows['sess_block_trial'].append({**sess_trial_key, 'block': block_num})

                # ---- WaterPort Choice ----
                trial_choice = {'water_port': None}
                for lick_port in lick_ports:
                    if any((df_behavior_trial['MSG'] == 'Choice_{}'.format(self.water_port_name_mapper[lick_port]))
                           & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                        trial_choice['water_port'] = lick_port
                        break

                rows['trial_choice'].append({**sess_trial_key, **trial_choice})

                # ---- Trial events ----
                time_TrialStart = df_behavior_session['PC-TIME'][trial_start_idx].to_numpy()
                time_GoCue = df_behavior_trial['PC-TIME'][(df_behavior_trial['MSG'] == 'GoCue') & (df_behavior_trial['TYPE'] == 'TRANSITION')].to_numpy()

                lick_times = {}
                for lick_port in lick_ports:
                    lick_times[lick_port] = df_behavior_trial['PC-TIME'][(df_behavior_trial['+INFO'] == water_port_channels[lick_port])].to_numpy()

                # early lick
                early_lick = 'no early'
                for lick_port in lick_ports:
                    if any(lick_times[lick_port] - time_GoCue < np.timedelta64(0)):
                        early_lick = 'early'
                        break
                # outcome
                outcome = 'miss' if trial_choice['water_port'] else 'ignore'
                for lick_port in lick_ports:
                    if any((df_behavior_trial['MSG'] == 'Reward_{}'.format(self.water_port_name_mapper[lick_port]))
                           & (df_behavior_trial['TYPE'] == 'TRANSITION')):
                        outcome = 'hit'
                        break

                # ---- accumulated reward ----
                for lick_port in lick_ports:
                    reward_var_name = 'reward_{}_accumulated'.format(self.water_port_name_mapper[lick_port])
                    if reward_var_name not in df_behavior_trial:
                        log.error('Bpod CSV KeyError: {} - Available columns: {}'.format(reward_var_name, df_behavior_trial.columns))
                        return

                    reward = df_behavior_trial[reward_var_name].values[0]
                    rows['available_reward'].append({
                        **sess_trial_key, 'water_port': lick_port,
                        'reward_available': False if np.isnan(reward) else reward})

                # ---- auto water and notes ----
                auto_water = False
                auto_water_times = {}
                for lick_port in lick_ports:
                    auto_water_varname = 'Auto_Water_{}'.format(self.water_port_name_mapper[lick_port])
                    auto_water_ind = (df_behavior_trial['TYPE'] == 'STATE') & (df_behavior_trial['MSG'] == auto_water_varname)
                    if any(auto_water_ind):
                        auto_water = True
                        auto_water_times[lick_port] = float(df_behavior_trial['+INFO'][auto_water_ind.idxmax()])

                if auto_water_times:
                    auto_water_ports = [k for k, v in auto_water_times.items() if v > 0.001]
                    rows['trial_note'].append({**sess_trial_key,
                                               'trial_note_type': 'autowater',
                                               'trial_note': 'and '.join(auto_water_ports)})

                # add random seed start note
                if any(df_behavior_trial['MSG'] == 'Random seed:'):
                    seedidx = (df_behavior_trial['MSG'] == 'Random seed:').idxmax() + 1
                    rows['trial_note'].append({**sess_trial_key,
                                               'trial_note_type': 'random_seed_start',
                                               'trial_note': str(df_behavior_trial['MSG'][seedidx])})
                    
                # add randomID (TrialBitCode)
                if any(df_behavior_trial['MSG'] == 'TrialBitCode: '):
                    bitcode_ind = (df_behavior_trial['MSG'] == 'TrialBitCode: ').idxmax() + 1
                    rows['trial_note'].append({**sess_trial_key,
                                               'trial_note_type': 'bitcode',
                                               'trial_note': str(df_behavior_trial['MSG'][bitcode_ind])})

                # ---- Behavior Trial ----
                rows['behavior_trial'].append({**sess_trial_key,
                                               'task': task,
                                               'task_protocol': task_protocol,
                                               'trial_instruction': 'none',
                                               'early_lick': early_lick,
                                               'outcome': outcome,
                                               'auto_water': auto_water,
                                               'free_water': False})  # TODO: verify this

                # ---- Water Valve Setting ----
                valve_setting = {**sess_trial_key}

                if 'var_motor:LickPort_Lateral_pos' in df_behavior_trial.keys():
                    valve_setting['water_port_lateral_pos'] = \
                    df_behavior_trial['var_motor:LickPort_Lateral_pos'].values[0]
                if 'var_motor:LickPort_RostroCaudal_pos' in df_behavior_trial.keys():
                    valve_setting['water_port_rostrocaudal_pos'] = \
                    df_behavior_trial['var_motor:LickPort_RostroCaudal_pos'].values[0]
                if 'var_motor:LickPort_DorsoVentral_pos' in df_behavior_trial.keys():
                    valve_setting['water_port_dorsoventral_pos'] = \
                    df_behavior_trial['var_motor:LickPort_DorsoVentral_pos'].values[0]

                rows['valve_setting'].append(valve_setting)

                for lick_port in lick_ports:
                    valve_open_varname = 'var:ValveOpenTime_{}'.format(self.water_port_name_mapper[lick_port])
                    if valve_open_varname in df_behavior_trial:
                        rows['valve_open_dur'].append({
                            **sess_trial_key, 'water_port': lick_port,
                            'open_duration': df_behavior_trial[valve_open_varname].values[0]})

                # ---- Trial Event and Action Event ----

                # -- add Go Cue
                GoCueTimes = (time_GoCue - time_TrialStart) / np.timedelta64(1, 's')
                GoCueTimes[GoCueTimes > 9999] = 9999  # Wordaround for bug #9: BPod protocol was paused and then 
                                                      # resumed after an impossible long period of time (> decimal(8, 4)).
                
                rows['trial_event'].extend([{**sess_trial_key, 'trial_event_id': idx, 'trial_event_type': 'go',
                                             'trial_event_time': t, 'duration': 0} for idx, t in enumerate(GoCueTimes)])

                # -- add licks
                all_lick_types = np.concatenate([[ltype]*len(ltimes) for ltype, ltimes in lick_times.items()])
                all_lick_times = np.concatenate([(ltimes - time_TrialStart) / np.timedelta64(1, 's') for ltimes in lick_times.values()])

                # sort by lick times
                sorted_licks = sorted(zip(all_lick_types, all_lick_times), key=lambda x: x[-1])

                rows['action_event'].extend([{**sess_trial_key, 'action_event_id': idx,
                                              'action_event_type': '{} lick'.format(ltype),
                                              'action_event_time': ltime} for idx, (ltype, ltime)
                                            in enumerate(sorted_licks)])

            # add to the session-concat
            for tbl in tbls_2_insert:
                concat_rows[tbl].extend(rows[tbl])
        # ---- The insertions to relevant tables ----
        # Session, SessionComment, SessionDetails insert
        log.info('BehaviorIngest.make(): adding session record')
        experiment.Session.insert1(sess_key, ignore_extra_fields=True)
        experiment.SessionComment.insert1(sess_key, ignore_extra_fields=True)
        experiment.SessionDetails.insert1(sess_key, ignore_extra_fields=True)

        # Behavior Insertion
        insert_settings = {'ignore_extra_fields': True, 'allow_direct_insert': True}

        log.info('BehaviorIngest.make(): bulk insert phase')

        log.info('BehaviorIngest.make(): ... experiment.Session.Trial')
        experiment.SessionTrial.insert(concat_rows['sess_trial'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.BehaviorTrial')
        experiment.BehaviorTrial.insert(concat_rows['behavior_trial'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.WaterPortChoice')
        experiment.WaterPortChoice.insert(concat_rows['trial_choice'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.TrialNote')
        experiment.TrialNote.insert(concat_rows['trial_note'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.TrialEvent')
        experiment.TrialEvent.insert(concat_rows['trial_event'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.ActionEvent')
        experiment.ActionEvent.insert(concat_rows['action_event'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.SessionBlock')
        experiment.SessionBlock.insert(concat_rows['sess_block'], **insert_settings)
        experiment.SessionBlock.BlockTrial.insert(concat_rows['sess_block_trial'], **insert_settings)
        block_reward_prob = []
        for block in concat_rows['sess_block']:
            block_reward_prob.extend([{**block, 'water_port': water_port, 'reward_probability': reward_p}
                                      for water_port, reward_p in block.pop('reward_probability').items()])
        experiment.SessionBlock.WaterPortRewardProbability.insert(block_reward_prob, **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.TrialAvailableReward')
        experiment.TrialAvailableReward.insert(concat_rows['available_reward'], **insert_settings)

        log.info('BehaviorIngest.make(): ... experiment.WaterValveSetting')
        experiment.WaterPortSetting.insert(concat_rows['valve_setting'], **insert_settings)
        experiment.WaterPortSetting.OpenDuration.insert(concat_rows['valve_open_dur'], **insert_settings)

        # Behavior Ingest Insertion
        log.info('BehaviorBpodIngest.make(): saving ingest {}'.format(sess_key))
        self.insert1(sess_key, **insert_settings)
        self.BehaviorFile.insert([{**sess_key, 'behavior_file': pathlib.Path(s.path).as_posix()}
                                  for s in sessions_now], **insert_settings)
