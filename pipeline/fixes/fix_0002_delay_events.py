#! /usr/bin/env python

import os
import logging
import math

from pathlib import Path
from itertools import chain
from collections import namedtuple

import datajoint as dj
import numpy as np
import scipy.io as spio

from pipeline import shell
from pipeline import experiment
from pipeline.ingest import behavior as behavior_ingest

# from pipeline.fixes import fix_0002_delay_events
# from pipeline.fixes.fix_0002_delay_events import fix_session
log = logging.getLogger(__name__)
loglevel = 'INFO'
shell.logsetup(loglevel)
log.setLevel(loglevel)


def find_path(path, fname):
    log.info('finding {} in {}'.format(fname, path))

    for root, dirs, files in os.walk(path):
        log.debug('RigDataFile.make(): entering {r}'.format(r=root))
        for f in files:
            if f == fname:
                log.info('found: {}/{}'.format(root, f))
                return Path(root) / Path(f)

    return None


def fix_session(session_key):
    paths = behavior_ingest.RigDataPath.fetch(as_dict=True)
    files = (behavior_ingest.BehaviorIngest
             * behavior_ingest.BehaviorIngest.BehaviorFile
             & session_key).fetch(as_dict=True, order_by='behavior_file asc')

    filelist = []
    for pf in [(p, f) for f in files for p in paths]:
        p, f = pf
        found = find_path(p['rig_data_path'], f['behavior_file'])
        if found:
            filelist.append(found)

    if len(filelist) != len(files):
        log.warning("behavior files missing in {} ({}/{}). skipping".format(
            session_key, len(filelist), len(files)))
        return

    log.info('filelist: {}'.format(filelist))

    #
    # Prepare PhotoStim
    #
    photosti_duration = 0.5  # (s) Hard-coded here
    photostims = {4: {'photo_stim': 4, 'photostim_device': 'OBIS470',
                      'brain_location_name': 'left_alm', 'duration': photosti_duration},
                  5: {'photo_stim': 5, 'photostim_device': 'OBIS470',
                      'brain_location_name': 'right_alm', 'duration': photosti_duration},
                  6: {'photo_stim': 6, 'photostim_device': 'OBIS470',
                      'brain_location_name': 'both_alm', 'duration': photosti_duration}}


    #
    # Extract trial data from file(s) & prepare trial loop
    #

    trials = zip()

    trial = namedtuple(  # simple structure to track per-trial vars
        'trial', ('ttype', 'stim', 'settings', 'state_times', 'state_names',
                  'state_data', 'event_data', 'event_times'))

    for f in filelist:

        if os.stat(f).st_size/1024 < 1000:
            log.info('skipping file {f} - too small'.format(f=f))
            continue

        log.debug('loading file {}'.format(f))

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
                    (AllTrialTypes, AllTrialSettings,
                     AllStateNames, AllStateData, AllEventData,
                     AllEventTimestamps))))

        if 'StimTrials' in SessionData.dtype.fields:
            log.debug('StimTrials detected in session - will include')
            AllStimTrials = SessionData['StimTrials'][0]
            assert(AllStimTrials.shape[0] == AllStateTimestamps.shape[0])
        else:
            log.debug('StimTrials not detected in session - will skip')
            AllStimTrials = np.array([
                None for i in enumerate(range(AllStateTimestamps.shape[0]))])

        z = zip(AllTrialTypes, AllStimTrials, AllTrialSettings,
                AllStateTimestamps, AllStateNames, AllStateData,
                AllEventData, AllEventTimestamps)

        trials = chain(trials, z)  # concatenate the files

    trials = list(trials)

    # all files were internally invalid or size < 100k
    if not trials:
        log.warning('skipping ., no valid files')
        return

    key = session_key
    skey = (experiment.Session & key).fetch1()

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

    i = -1
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
                           'DelayPeriod', 'ResponseCue', 'StopLicking',
                           'TrialEnd')

        missing = list(k for k in required_states if k not in states)

        if len(missing):
            log.warning('skipping trial {i}; missing {m}'
                        .format(i=i, m=missing))
            continue

        gui = t.settings['GUI'].flatten()

        # ProtocolType - only ingest protocol >= 3
        #
        # 1 Water-Valve-Calibration 2 Licking 3 Autoassist
        # 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed
        #

        if 'ProtocolType' not in gui.dtype.names:
            log.warning('skipping trial {i}; protocol undefined'
                        .format(i=i))
            continue

        protocol_type = gui['ProtocolType'][0]
        if gui['ProtocolType'][0] < 3:
            log.warning('skipping trial {i}; protocol {n} < 3'
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
        log.debug('endindex\n' + str(endindex))

        if not(len(startindex) and len(endindex)):
            log.warning('skipping trial {i}: start/end index error: {s}/{e}'.format(i=i,s=str(startindex), e=str(endindex)))
            continue

        try:    
            tkey['trial'] = i
            tkey['trial_uid'] = i  # Arseny has unique id to identify some trials
            tkey['start_time'] = t.state_times[startindex][0]
            tkey['stop_time'] = t.state_times[endindex][0]
        except IndexError:
            log.warning('skipping trial {i}: error indexing {s}/{e} into {t}'.format(i=i, s=str(startindex), e=str(endindex), t=str(t.state_times)))
            continue

        log.debug('BehaviorIngest.make(): Trial().insert1')  # TODO msg
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
        nkey['trial_note'] = str(gui['Autolearn'][0])
        rows['trial_note'].append(nkey)

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
        log.debug('BehaviorIngest.make(): presample')

        ekey = dict(tkey)
        sampleindex = np.where(t.state_data == states['SamplePeriod'])[0]

        ekey['trial_event_id'] = len(rows['trial_event'])
        ekey['trial_event_type'] = 'presample'
        ekey['trial_event_time'] = t.state_times[startindex][0]
        ekey['duration'] = (t.state_times[sampleindex[0]]
                            - t.state_times[startindex])[0]

        if math.isnan(ekey['duration']):
            log.debug('BehaviorIngest.make(): fixing presample duration')
            ekey['duration'] = 0.0  # FIXDUR: lookup from previous trial

        rows['trial_event'].append(ekey)

        #
        # Add other 'sample' events
        #

        log.debug('BehaviorIngest.make(): sample events')

        last_dur = None

        for s in sampleindex:  # in protocol > 6 ~-> n>1
            # todo: batch events
            ekey = dict(tkey)
            ekey['trial_event_id'] = len(rows['trial_event'])
            ekey['trial_event_type'] = 'sample'
            ekey['trial_event_time'] = t.state_times[s]
            ekey['duration'] = gui['SamplePeriod'][0]

            if math.isnan(ekey['duration']) and last_dur is None:
                log.warning('... trial {} bad duration, no last_edur'
                            .format(i, last_dur))
                ekey['duration'] = 0.0  # FIXDUR: cross-trial check
                rows['corrected_trial_event'].append(ekey)

            elif math.isnan(ekey['duration']) and last_dur is not None:
                log.warning('... trial {} duration using last_edur {}'
                            .format(i, last_dur))
                ekey['duration'] = last_dur
                rows['corrected_trial_event'].append(ekey)

            else:
                last_dur = ekey['duration']  # only track 'good' values.

            rows['trial_event'].append(ekey)

        #
        # Add 'delay' events
        #

        log.debug('BehaviorIngest.make(): delay events')

        last_dur = None
        delayindex = np.where(t.state_data == states['DelayPeriod'])[0]

        for d in delayindex:  # protocol > 6 ~-> n>1
            ekey = dict(tkey)
            ekey['trial_event_id'] = len(rows['trial_event'])
            ekey['trial_event_type'] = 'delay'
            ekey['trial_event_time'] = t.state_times[d]
            ekey['duration'] = gui['DelayPeriod'][0]

            if math.isnan(ekey['duration']) and last_dur is None:
                log.warning('... {} bad duration, no last_edur'
                            .format(i, last_dur))
                ekey['duration'] = 0.0  # FIXDUR: cross-trial check
                rows['corrected_trial_event'].append(ekey)

            elif math.isnan(ekey['duration']) and last_dur is not None:
                log.warning('... {} duration using last_edur {}'
                            .format(i, last_dur))
                ekey['duration'] = last_dur
                rows['corrected_trial_event'].append(ekey)

            else:
                last_dur = ekey['duration']  # only track 'good' values.

            log.debug('delay event duration: {}'.format(ekey['duration']))
            rows['trial_event'].append(ekey)

        #
        # Add 'go' event
        #
        log.debug('BehaviorIngest.make(): go')

        ekey = dict(tkey)
        responseindex = np.where(t.state_data == states['ResponseCue'])[0]

        ekey['trial_event_id'] = len(rows['trial_event'])
        ekey['trial_event_type'] = 'go'
        ekey['trial_event_time'] = t.state_times[responseindex][0]
        ekey['duration'] = gui['AnswerPeriod'][0]

        if math.isnan(ekey['duration']):
            log.debug('BehaviorIngest.make(): fixing go duration')
            ekey['duration'] = 0.0  # FIXDUR: lookup from previous trials
            rows['corrected_trial_event'].append(ekey)

        rows['trial_event'].append(ekey)

        #
        # Add 'trialEnd' events
        #

        log.debug('BehaviorIngest.make(): trialend events')

        last_dur = None
        trialendindex = np.where(t.state_data == states['TrialEnd'])[0]

        ekey = dict(tkey)
        ekey['trial_event_id'] = len(rows['trial_event'])
        ekey['trial_event_type'] = 'trialend'
        ekey['trial_event_time'] = t.state_times[trialendindex][0]
        ekey['duration'] = 0.0

        rows['trial_event'].append(ekey)

        #
        # Add lick events
        #

        lickleft = np.where(t.event_data == 69)[0]
        log.debug('... lickleft: {r}'.format(r=str(lickleft)))

        action_event_count = len(rows['action_event'])
        if len(lickleft):
            [rows['action_event'].append(
                dict(tkey, action_event_id=action_event_count+idx, action_event_type='left lick',
                     action_event_time=t.event_times[l]))
             for idx, l in enumerate(lickleft)]

        lickright = np.where(t.event_data == 71)[0]
        log.debug('... lickright: {r}'.format(r=str(lickright)))

        action_event_count = len(rows['action_event'])
        if len(lickright):
            [rows['action_event'].append(
                dict(tkey, action_event_id=action_event_count+idx, action_event_type='right lick',
                     action_event_time=t.event_times[r]))
                for idx, r in enumerate(lickright)]

        # Photostim Events
        #
        # TODO:
        #
        # - base stimulation parameters:
        #
        #   - should be loaded elsewhere - where
        #   - actual ccf locations - cannot be known apriori apparently?
        #   - Photostim.Profile: what is? fix/add
        #
        # - stim data
        #
        #   - how retrieve power from file (didn't see) or should
        #     be statically coded here?
        #   - how encode stim type 6?
        #     - we have hemisphere as boolean or
        #     - but adding an event 4 and event 5 means querying
        #       is less straightforwrard (e.g. sessions with 5 & 6)

        if t.stim:
            log.debug('BehaviorIngest.make(): t.stim == {}'.format(t.stim))
            rows['photostim_trial'].append(tkey)
            delay_period_idx = np.where(t.state_data == states['DelayPeriod'])[0][0]
            rows['photostim_trial_event'].append(dict(
                tkey, **photostims[t.stim], photostim_event_id=len(rows['photostim_trial_event']),
                photostim_event_time=t.state_times[delay_period_idx],
                power=5.5))

        # end of trial loop.

    log.info('BehaviorIngest.make(): ... experiment.TrialEvent')

    fix_events = rows['trial_event']

    ref_events = (experiment.TrialEvent() & skey).fetch(
        order_by='trial, trial_event_id', as_dict=True)

    if False:
        for e in ref_events:
            log.debug('ref_events: t: {}, e: {}, event_type: {}'.format(
                e['trial'], e['trial_event_id'], e['trial_event_type']))
        for e in fix_events:
            log.debug('fix_events: t: {}, e: {}, type: {}'.format(
                e['trial'], e['trial_event_id'], e['trial_event_type']))

    log.info('deleting old events')

    with dj.config(safemode=False):

        log.info('... TrialEvent')
        (experiment.TrialEvent() & session_key).delete()

        log.info('... CorrectedTrialEvents')
        (behavior_ingest.BehaviorIngest.CorrectedTrialEvents()
         & session_key).delete_quick()

    log.info('adding new records')

    log.info('... experiment.TrialEvent')
    experiment.TrialEvent().insert(
        rows['trial_event'], ignore_extra_fields=True,
        allow_direct_insert=True, skip_duplicates=True)

    log.info('... CorrectedTrialEvents')
    behavior_ingest.BehaviorIngest.CorrectedTrialEvents().insert(
        rows['corrected_trial_event'], ignore_extra_fields=True,
        allow_direct_insert=True)


def verify_session(s):
    log.info('verifying_session {}'.format(s))
    evts = (experiment.TrialEvent & s).fetch(order_by='trial, trial_event_id')

    def note_prob(s, e, msg):
        log.warning('{} {} {} {}: {}'.format(
            s, e['trial'], e['trial_event_id'], e['trial_event_type'], msg))

    eid, state, nerr = None, None, 0

    for e in evts:

        neweid = e['trial_event_id']
        newstate = e['trial_event_type']

        if eid is not None and neweid != eid + 1:
            note_prob(s, e, 'laste: {} newe: {}'.format(eid, neweid))

        if newstate == 'presample':
            if state and state not in {'presample', 'trialend'}:
                note_prob(s, e)
                nerr += 1
        if newstate == 'sample':
            if state and state not in {'presample', 'sample'}:
                note_prob(s, e)
                nerr += 1
        if newstate == 'delay':
            if state and state not in {'sample', 'delay'}:
                note_prob(s, e)
                nerr += 1
        if newstate == 'go':
            if state and state not in {'delay', 'go'}:
                note_prob(s, e)
                nerr += 1
        if newstate == 'trialend':
            if state and state not in {'go', 'trialend'}:
                note_prob(s, e)
                nerr += 1

        eid, state = neweid, newstate

    if not nerr:
        log.info('session {} verifies OK'.format(s))
    else:
        log.warning('session {} had {} verification errors.'.format(s, nerr))

def fix_0002_delay_events():
    with dj.conn().transaction:

        # fh = {'fix_name': 'fix_0002_delay_events',
        #       'fix_timestamp': datetime.now()}

        # fix_history.FixHistory.insert1(fh)

        q = (experiment.Session & behavior_ingest.BehaviorIngest)

        for s in q.fetch('KEY'):
            fix_session(s)
            verify_session(s)


if __name__ == '__main__':
    # shell.logsetup('DEBUG')
    fix_0002_delay_events()
