#! /usr/bin/env python

'''
reload of photostim trial event times from file
'''

import os
import logging
import math

from pathlib import Path
from itertools import chain
from collections import namedtuple
from datetime import datetime

import numpy as np
import scipy.io as spio

from pipeline import lab
from pipeline import experiment
from pipeline.fixes import fix_history

from pipeline.ingest import behavior as behavior_ingest

log = logging.getLogger(__name__)


# TODO: were we inserting full rig-subpaths paths before pathlib?
#       or just names.. this would not be necessary with rig-subpath stored
#       also, could be separate issue from which is better.

def find_path(path, fname):
    print('finding', fname, 'in', path)

    for root, dirs, files in os.walk(path):
        print('RigDataFile.make(): entering {r}'.format(r=root))
        for f in files:
            if f == fname:
                print('found', root, f)
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
        print("session {} has behavior files missing. skipping")
        return

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
    # Load all files & create combined list of per-trial data
    #

    trials = zip()

    trial = namedtuple(  # simple structure to track per-trial vars
        'trial', ('ttype', 'stim', 'settings', 'state_times', 'state_names',
                  'state_data', 'event_data', 'event_times'))

    for f in filelist:
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

        trials = chain(trials, z)

    trials = list(trials)

    # all files were internally invalid or size < 100k
    if not trials:
        log.warning('skipping date {d}, no valid files'.format(d=date))

    #
    # Trial data seems valid; synthesize session id & add session record
    # XXX: note - later breaks can result in Sessions without valid trials
    #

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
        # Add 'trialEnd' events
        #

        log.debug('BehaviorIngest.make(): trialend events')

        last_dur = None
        trialendindex = np.where(t.state_data == states['TrialEnd'])[0]

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

        lickright = np.where(t.event_data == 70)[0]
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
            log.info('BehaviorIngest.make(): t.stim == {}'.format(t.stim))
            rows['photostim_trial'].append(tkey)
            delay_period_idx = np.where(t.state_data == states['DelayPeriod'])[0][0]
            rows['photostim_trial_event'].append(dict(
                tkey, **photostims[t.stim], photostim_event_id=len(rows['photostim_trial_event']),
                photostim_event_time=t.state_times[delay_period_idx],
                power=5.5))

        # end of trial loop.

    pe = experiment.PhotostimEvent
    for pev in rows['photostim_trial_event']:

        log.debug('updating photostim_event {}'.format(pev))

        pe_k = {k: v for k, v in pev.items() if k in pe.primary_key}

        (pe & pe_k)._update('photostim_event_time',
                            pev['photostim_event_time'])

        (pe & pe_k)._update('power', pev['power'])


def verify_session(session_key):
    if len(experiment.PhotostimEvent & session_key & {'power': 0.0}) != 0:
        log.warning('NOK: session {} session_key still has power==0.0 events')
    else:
        log.info('OK: session {} has no power==0.0 events')


def fix_0001_photostim():

    fix_history.schema.connection.start_transaction()

    fh = {'fix_name': 'fix_0001_photostim', 'fix_timestamp': datetime.now() }

    fix_history.FixHistory.insert1(fh)

    q = (experiment.Session & behavior_ingest.BehaviorIngest)

    for s in q.fetch('KEY'):
        fix_session(s)
        verify_session(s)
        fix_history.FixHistory.FixHistorySession.insert1({**fh, **s})

    fix_history.schema.connection.commit_transaction()


if __name__ == '__main__':
    fix_0001_photostim()

