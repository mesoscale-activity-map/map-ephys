#! /usr/bin/env python

import os
import sys
import logging
from collections import namedtuple


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

        # '%%' due to datajoint-python/issues/376
        dups = (self & "imported_session_file like '%%{h2o}%%{date}%%'"
                .format(h2o=h2o, date=date))

        if len(dups) > 1:
            # TODO: handle split file
            log.warning('split session case detected')
            return

        skey = {}
        # lookup animal
        log.info('looking up animal for {h2o}'.format(h2o=h2o))
        skey['animal'] = (lab.Animal()
                          & (lab.AnimalWaterRestriction
                             and {'water_restriction': h2o})).fetch1('animal')
        log.info('got {animal}'.format(animal=skey['animal']))

        # synthesize session id
        log.info('synthesizing session ID')
        skey['session'] = (dj.U().aggr(experiment.Session(),
                                       n='max(session)').fetch1('n') or 0)+1

        log.info('generated session id: {session}'.format(
            session=skey['session']))

        skey['session_date'] = date[0:4] + '-' + date[4:6] + '-' + date[6:8]
        skey['username'] = 'daveliu'
        skey['rig'] = 'TRig1'

        if experiment.Session() & skey:
            # XXX: raise DataJointError?
            log.warning("Warning! session exists for {f}".format(fname))

        log.info('ImportedSessionFileIngest.make(): adding session record')
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
            # values (seem to be? are?):
            # 'StopLicking', 'Reward', 'TimeOut', 'NoResponse',
            # 'EarlyLickDelay', 'EarlyLickSample', 'PreSamplePeriod'
            # 'SamplePeriod', 'DelayPeriod', 'ResponseCue'

            states = {k: (v+1) for v, k in enumerate(t.state_names)}

            # GUI Settings used in several places
            gui = t.settings['GUI'].flatten()

            #
            # Top-level 'Trial' record
            #

            tkey = dict(skey)

            startindex = (np.where(t.state_data == states['PreSamplePeriod'])
                          if 'PreSamplePeriod' in states else 0)

            endindex = (np.where(t.state_data == states['StopLicking'])
                        if 'StopLicking' in states else 0)

            # print(str(startindex))
            # print(str(endindex))
            tkey['trial'] = i
            tkey['start_time'] = t.state_times[startindex]
            tkey['end_time'] = t.state_times[endindex]

            log.info('ImportedSessionFileIngest.make(): Trial().insert1')
            experiment.Session.Trial().insert1(tkey, ignore_extra_fields=True)

            #
            # Add 'protocol' note
            #

            nkey = dict(tkey)
            nkey['trial_note_type'] = 'protocol #'
            nkey['trial_note'] = str(gui['ProtocolType'][0])
            log.info('ImportedSessionFileIngest.make(): TrialNote().insert1')
            experiment.TrialNote().insert1(nkey, ignore_extra_fields=True)

            #
            # Add presample event
            #

            ekey = dict(tkey)

            sampleindex = (np.where(t.state_data == states['SamplePeriod'])
                           if 'SamplePeriod' in states else 0)

            ekey['trial_event_type'] = 'presample'
            ekey['trial_event_time'] = t.state_times[startindex]
            ekey['duration'] = (t.state_times[sampleindex]
                                - t.state_times[startindex])
            log.info('ImportedSessionFileIngest.make(): presample')
            experiment.TrialEvent().insert1(ekey, ignore_extra_fields=True)

            #
            # Add 'go' event
            #

            ekey = dict(tkey)

            responseindex = (np.where(t.state_data == states['ResponseCue'])
                             if 'ResponseCue' in states else 0)

            ekey['trial_event_type'] = 'go'
            ekey['trial_event_time'] = t.state_times[responseindex]
            ekey['duration'] = gui['AnswerPeriod'][0]
            log.info('ImportedSessionFileIngest.make(): go')
            experiment.TrialEvent().insert1(ekey, ignore_extra_fields=True)

            #
            # Add other 'sample' events
            #

            # TODO: not 100% here - can we index on contents of sampleindex?
            log.info('ImportedSessionFileIngest.make(): sample events')
            for s in sampleindex:
                # todo: batch events
                ekey = dict(tkey)
                ekey['trial_event_type'] = 'sample'
                ekey['trial_event_time'] = t.state_times[s]
                ekey['duration'] = gui['SamplePeriod'][0]
                experiment.TrialEvent().insert1(ekey, ignore_extra_fields=True)

            #
            # Add 'delay' events
            #

            # TODO: not 100% here - can we index on contents of delayindex?
            delayindex = (np.where(t.state_data == states['DelayPeriod'])
                          if 'DelayPeriod' in states else 0)

            log.info('ImportedSessionFileIngest.make(): delay events')
            for d in delayindex:
                # todo: batch events
                ekey = dict(tkey)
                ekey['trial_event_type'] = 'delay'
                ekey['trial_event_time'] = t.state_times[d]
                ekey['duration'] = gui['DelayPeriod'][0]
                experiment.TrialEvent().insert1(ekey, ignore_extra_fields=True)

            #
            # Specific BehaviorTrial information for this trial
            #

            bkey = dict(tkey)
            bkey['task'] = 'audio_delay'

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
            # ProtocolType (seems to be? is?)
            # 1 Water-Valve-Calibration 2 Licking 3 Autoassist
            # 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed

            if (gui['ProtocolType'][0] >= 5
                and 'EarlyLickDelay' in states
                    and np.any(t.state_data == states['EarlyLickDelay'])):
                    early_lick = 'early'

            if (gui['ProtocolType'][0] > 5
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
            log.info('ImportedSessionFileIngest.make(): BehaviorTrial()')
            experiment.BehaviorTrial().insert1(bkey, ignore_extra_fields=True)

            #
            # Add lick events
            #

            lickleft = np.where(t.event_data == 69)

            if lickleft:
                # XXX: needs testing.
                log.info('ImportedSessionFileIngest.make(): left licks')
                experiment.ActionEvent().insert(
                    list((dict(**tkey,
                               trial_event_type='left lick',
                               trial_event_time=t.event_times[l],
                               duration=1)
                          for l in lickleft)))

            lickright = np.where(t.event_data == 70)

            if lickright:
                # XXX: needs testing.
                log.info('ImportedSessionFileIngest.make(): right licks')
                experiment.ActionEvent().insert(
                    list((dict(**tkey,
                               trial_event_type='right lick',
                               trial_event_time=t.event_times[r],
                               duration=1)
                          for r in lickright)))

            # end of trial loop.

        # save a record here to prevent future loading
        self.insert1(key, ignore_extra_fields=True)


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
    log.setLevel(logging.INFO)  # but show ours
    ImportedSessionFile().populate()
    ImportedSessionFileIngest().populate()
