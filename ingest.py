#! /usr/bin/env python

import os
import sys
import logging

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

        fname = key['imported_session_file']
        fpath = os.path.join(dj.config['imported_session_path'], fname)

        log.info('ImportedSessionFileIngest.make(): Loading {f}'
                 .format(f=fname))

        # split files like 'dl7_TW_autoTrain_20171114_140357.mat'
        h2o, t1, t2, date, time = fname.split('.')[0].split('_')
        
        if os.stat(fpath).st_size/1024 > 500: #False:  # TODO: pre-populate lab.Animal and AnimalWaterRestriction

            # '%%' due to datajoint-python/issues/376
            #dups = (self & "imported_session_file like '%%{h2o}%%{date}%%"
                    #.format(h2o=h2o, date=date))

            #if len(dups) > 1:
                #log.warning('split session case detected')
                # TODO: handle split file
                # TODO: self.insert( all split files )
                #return

            # lookup animal
            log.info('looking up animal for {h2o}'.format(h2o=h2o))
            key['animal'] = (lab.Animal()
                             & (lab.AnimalWaterRestriction
                                and {'water_restriction': h2o})).fetch1('animal')
            log.info('got {animal}'.format(animal=key['animal']))

            # synthesize session
            log.info('synthesizing session ID')
            key['session'] = (dj.U().aggr(experiment.Session(),
                                          n='max(session)').fetch1('n') or 0)+1

            log.info('generated session id: {session}'.format(
                session=key['session']))
            experiment.Session().insert1((key['animal'], key['session'], date[0:4]+'-'+date[4:6]+'-'+date[6:8],'daveliu', 'TRig1', key['imported_session_file']))
            #if experiment.Session() & key:
                # XXX: raise DataJointError?
                #log.warning("Warning! session exists for {f}".format(fname))

            mat = spio.loadmat(fpath, squeeze_me=True)  # NOQA
            SessionData=mat['SessionData']
            TrialTypes=SessionData.flatten()[0][0]
            RawData=SessionData.flatten()[0][7]
            TrialSettings=SessionData.flatten()[0][10]
            OriginalStateNamesByNumber=RawData.flatten()[0][0]
            OriginalStateData=RawData.flatten()[0][1]
            OriginalEventData=RawData.flatten()[0][2]
            OriginalStateTimestamps=RawData.flatten()[0][3]
            OriginalEventTimestamps=RawData.flatten()[0][4]
    
            for i in range(0, len(OriginalStateTimestamps)):
                trial_instruction = 'left'
                early_lick = 'no early'
                outcome = 'ignore'
                GUI = TrialSettings[i][0]
                SampleDur = GUI.flatten()[0][1]
                DelayDur = GUI.flatten()[0][2]
                AnswerPeriod = GUI.flatten()[0][3]
                ProtocolType = GUI.flatten()[0][10] # 1 Water-Valve-Calibration 2 Licking 3 Autoassist 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed
                Reversal = GUI.flatten()[0][13]
                StopLicking=np.where(OriginalStateNamesByNumber[i]=='StopLicking')[0]+1
                Reward=np.where(OriginalStateNamesByNumber[i]=='Reward')[0]+1
                TimeOut=np.where(OriginalStateNamesByNumber[i]=='TimeOut')[0]+1
                NoResponse=np.where(OriginalStateNamesByNumber[i]=='NoResponse')[0]+1
                EarlyLickDelay=np.where(OriginalStateNamesByNumber[i]=='EarlyLickDelay')[0]+1
                EarlyLickSample=np.where(OriginalStateNamesByNumber[i]=='EarlyLickSample')[0]+1
                PreSamplePeriod=np.where(OriginalStateNamesByNumber[i]=='PreSamplePeriod')[0]+1
                SamplePeriod=np.where(OriginalStateNamesByNumber[i]=='SamplePeriod')[0]+1
                DelayPeriod=np.where(OriginalStateNamesByNumber[i]=='DelayPeriod')[0]+1
                ResponseCue=np.where(OriginalStateNamesByNumber[i]=='ResponseCue')[0]+1
                startindex = np.where(OriginalStateData[i]==PreSamplePeriod)[0]
                sampleindex = np.where(OriginalStateData[i]==SamplePeriod)[0]
                delayindex = np.where(OriginalStateData[i]==DelayPeriod)[0]
                responseindex = np.where(OriginalStateData[i]==ResponseCue)[0]
                endindex = np.where(OriginalStateData[i]==StopLicking)[0]
                lickleft = np.where(OriginalEventData[i]==69)[0]
                lickright = np.where(OriginalEventData[i]==70)[0]
                if np.any(OriginalStateData[i]==Reward):
                    outcome = 'hit'
                elif np.any(OriginalStateData[i]==TimeOut):
                    outcome = 'miss'
                elif np.any(OriginalStateData[i]==NoResponse):
                    outcome = 'ignore'
                if ProtocolType==5:
                    if np.any(OriginalStateData[i]==EarlyLickDelay):
                        early_lick = 'early'
                if ProtocolType>5:
                    if np.any(OriginalStateData[i]==EarlyLickDelay) or np.any(OriginalStateData[i]==EarlyLickSample):
                        early_lick = 'early'
    
                experiment.Session.Trial().insert1((key['animal'], key['session'], i, OriginalStateTimestamps[i][startindex][0], OriginalStateTimestamps[i][endindex[0]]))
                
                if Reversal==1:
                    if TrialTypes[i]==1:
                        trial_instruction = 'left'
                    elif TrialTypes[i]==0:
                        trial_instruction = 'right'
                elif Reversal==2:
                    if TrialTypes[i]==1:
                        trial_instruction = 'right'
                    elif TrialTypes[i]==0:
                        trial_instruction = 'left'
    
                experiment.BehaviorTrial().insert1((key['animal'], key['session'], i, 'audio delay', trial_instruction, early_lick, outcome))
                experiment.TrialNote().insert1((key['animal'], key['session'], i, 'protocol #', str(ProtocolType)))
                experiment.TrialEvent().insert([(key['animal'], key['session'], i, 'presample', OriginalStateTimestamps[i][startindex][0], OriginalStateTimestamps[i][sampleindex[0]]-OriginalStateTimestamps[i][startindex][0]),
                (key['animal'], key['session'], i, 'go', OriginalStateTimestamps[i][responseindex][0], AnswerPeriod)])
                for j in range(0, len(sampleindex)):
                    experiment.TrialEvent().insert1((key['animal'], key['session'], i, 'sample', OriginalStateTimestamps[i][sampleindex[j]], SampleDur))
                for j in range(0, len(delayindex)):
                    experiment.TrialEvent().insert1((key['animal'], key['session'], i, 'delay', OriginalStateTimestamps[i][delayindex[j]], DelayDur))
                if len(lickleft)>0:
                    lickleftAll=np.tile((key['animal'], key['session'], i, 'left lick'),(len(lickleft),1))
                    lickleftAll=np.insert(lickleftAll,[4],np.split(OriginalEventTimestamps[i][lickleft],len(lickleft)),axis=1)
                    experiment.ActionEvent().insert(lickleftAll)
                if len(lickright)>0:
                    lickrightAll=np.tile((key['animal'], key['session'], i, 'right lick'),(len(lickright),1))
                    lickrightAll=np.insert(lickrightAll,[4],np.split(OriginalEventTimestamps[i][lickright],len(lickright)),axis=1)
                    experiment.ActionEvent().insert(lickrightAll)
                #for j in range(0, len(lickleft)):
                    #experiment.ActionEvent().insert1((key['animal'], key['session'], i, 'left lick', OriginalEventTimestamps[i][lickleft[j]]))
                #for j in range(0, len(lickright)):
                    #experiment.ActionEvent().insert1((key['animal'], key['session'], i, 'right lick', OriginalEventTimestamps[i][lickright[j]]))
            # ... and save a record here to prevent future loading
            self.insert1(key, ignore_extra_fields=True)


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != 'populate':
        print("usage: {p} [populate]"
              .format(p=os.path.basename(sys.argv[0])))
        sys.exit(0)

    try:
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

    except:
        print("note: data existed", file=sys.stderr)

    logging.basicConfig(level=logging.ERROR)  # quiet other modules
    log.setLevel(logging.INFO)  # but show ours
    ImportedSessionFile().populate()
    ImportedSessionFileIngest().populate()
