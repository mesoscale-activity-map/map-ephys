#! /usr/bin/env python

import os
import sys
import logging

from collections import namedtuple
from code import interact
import yaml

import scipy.io as spio
import numpy as np

import datajoint as dj

import lab
import experiment


if 'imported_session_path' not in dj.config:
    dj.config['imported_session_path'] = 'R:\\Arduino\\Bpod_Train1\\Bpod Local\\Data\\dl7\\TW_autoTrain\\Session Data\\'

log = logging.getLogger(__name__)
schema = dj.schema(dj.config['ingest.database'], locals())


def _listfiles():
    return (f for f in os.listdir(dj.config['imported_session_path'])
            if f.endswith('.mat'))


@schema
class ImportedSessionFile(dj.Lookup):
    # TODO: more representative class name
    definition = """
    imported_session_file:         varchar(255)    # imported session file
    """

    contents = ((f,) for f in (_listfiles()))

    def populate(self):
        for f in _listfiles():
            if not self & {'imported_session_file': f}:
                self.insert1((f,))


@schema
class ImportedSessionFileIngest(dj.Imported):
    definition = """
    -> ImportedSessionFile
    ---
    -> experiment.Session
    """

    def make(self, key):

        #
        # Handle filename & Construct Session
        #

        fname = key['imported_session_file']
        fpath = os.path.join(dj.config['imported_session_path'], fname)

        log.info('ImportedSessionFileIngest.make(): Loading {f}'
                 .format(f=fname))

        # split files like 'dl7_TW_autoTrain_20171114_140357.mat'
        h2o, t1, t2, date, time = fname.split('.')[0].split('_')

        if os.stat(fpath).st_size/1024 < 500:
            log.info('skipping file {f} - too small'.format(f=fname))
            return

        # '%%' vs '%' due to datajoint-python/issues/376
        dups = (ImportedSessionFile()
                & "imported_session_file like '%%{h2o}%%{date}%%'"
                .format(h2o=h2o, date=date))

        if len(dups) > 1:
            # TODO: handle split file
            log.warning('split session case detected for {h2o} on {date}'
                        .format(h2o=h2o, date=date))
            return

        skey = {}
        # lookup animal
        log.debug('looking up animal for {h2o}'.format(h2o=h2o))
        animal = (lab.AnimalWaterRestriction()
                  & {'water_restriction': h2o}).fetch1('animal')
        log.info('animal is {animal}'.format(animal=animal))

        # synthesize session id
        log.debug('synthesizing session ID')
        session = (dj.U().aggr(experiment.Session() & {'animal': animal},
                               n='max(session)').fetch1('n') or 0) + 1
        log.info('generated session id: {session}'.format(session=session))

        skey['animal'] = animal
        skey['session'] = session
        skey['session_date'] = date[0:4] + '-' + date[4:6] + '-' + date[6:8]
        skey['username'] = 'daveliu'
        skey['rig'] = 'TRig1'

        if experiment.Session() & skey:
            log.warning("Warning! session exists for {f}".format(fname))

        log.debug('ImportedSessionFileIngest.make(): adding session record')
        # XXX: note - later breaks can result in Sessions without valid trials
        experiment.Session().insert1(skey)

        #
        # Extract trial data from file & prepare trial loop
        #

        mat = spio.loadmat(fpath, squeeze_me=True)
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

        trial = namedtuple(  # simple structure to track per-trial vars
            'trial', ('ttype', 'settings', 'state_times', 'state_names',
                      'state_data', 'event_data', 'event_times'))

        AllTrials = zip(AllTrialTypes, AllTrialSettings, AllStateTimestamps,
                        AllStateNames, AllStateData, AllEventData,
                        AllEventTimestamps)

        #
        # Actually load per-trial data
        #

        i = -1
        for t in AllTrials:

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
            if gui['ProtocolType'][0] < 5:
                log.info('skipping trial {i}; protocol {n} < 5'
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
                leftlicks = list((dict(**tkey,
                                       trial_event_type='sample',
                                       trial_event_time=t.event_times[l],
                                       action_event_type='left lick',
                                       action_event_time=t.event_times[l],
                                       duration=1)
                                  for l in lickleft))

                experiment.ActionEvent().insert(
                    leftlicks, ignore_extra_fields=True)

            lickright = np.where(t.event_data == 70)[0]
            log.debug('... lickright: {r}'.format(r=str(lickright)))
            if len(lickright):
                # TODO: is 'sample' the type?
                rightlicks = list((dict(**tkey,
                                        trial_event_type='sample',
                                        trial_event_time=t.event_times[r],
                                        action_event_type='right lick',
                                        action_event_time=t.event_times[r],
                                        duration=1)
                                   for r in lickright))

                experiment.ActionEvent().insert(
                    rightlicks, ignore_extra_fields=True)

            # end of trial loop.

        # save a record here to prevent future loading
        self.insert1(dict(**key, **skey), ignore_extra_fields=True)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != 'populate':
        print("usage: {p} [populate]"
              .format(p=os.path.basename(sys.argv[0])))
        sys.exit(0)

    try:
        # TODO: these should be loaded in a more 'official' way
        lab.Animal().insert1({
            'animal': 399752,
            'dob':  '2017-08-01'
        })
        lab.AnimalWaterRestriction().insert1({
            'animal': 399752,
            'water_restriction': 'dl7'
        })
        # dl8's 1000001 id is bogus...
        lab.Animal().insert1({
            'animal': 100001,  # bogus id
            'dob':  '2017-08-01'
        })
        lab.AnimalWaterRestriction().insert1({
            'animal': 100001,  # bogus id
            'water_restriction': 'dl8'
        })
        lab.Person().insert1({
            'username': 'daveliu',
            'fullname': 'Dave Liu'
        })
        lab.Rig().insert1({
            'rig': 'TRig1',
            'rig_description': 'TRig1'
        })
    except:
        print("note: data existed", file=sys.stderr)

    logging.basicConfig(level=logging.ERROR)  # quiet other modules
    log.setLevel(logging.INFO)  # show ours (default INFO; DEBUG for more)
    ImportedSessionFile().populate()
    ImportedSessionFileIngest().populate()
    if log.level == logging.DEBUG:
        interact('debug results', local=locals())
