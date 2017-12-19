import datajoint as dj
import lab, ccf

schema = dj.schema(dj.config['experiment.database'], locals())

@schema
class Task(dj.Lookup):
    definition = """
    # Type of tasks
    task            : varchar(12)                  # task type
    ----
    task_description : varchar(4000)
    """
    contents = [
         ('audio delay', 'auditory delayed response task (2AFC)'),
         ('audio mem', 'auditory working memory task'),
         ('s1 stim', 'S1 photostimulation task (2AFC)')
         ]


@schema
class Session(dj.Manual):
    definition = """
    -> lab.Animal
    #session : smallint 		# session number (change to int for now)
    session : int 		# session number
    ---
    session_date  : date
    -> lab.Person
    -> lab.Rig
    """

    class Trial(dj.Part):
        definition = """
        -> Session
        trial   : smallint
        ---
        start_time : decimal(8,4)  # (s)
        end_time : decimal(8,4)  # (s)
        """
		
		
    def _make_tuples(self, key, key1):
        import scipy.io as spio
        import numpy as np
        mat = spio.loadmat(key1, squeeze_me=True)
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

            Session.Trial().insert1((int(key[0]), int(key[1]), i, OriginalStateTimestamps[i][startindex][0], OriginalStateTimestamps[i][endindex[0]]))
            
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

            BehaviorTrial().insert1((int(key[0]), int(key[1]), i, 'audio delay', trial_instruction, early_lick, outcome))
            TrialNote().insert1((int(key[0]), int(key[1]), i, 'protocol #', str(ProtocolType)))
            TrialEvent().insert([(int(key[0]), int(key[1]), i, 'presample', OriginalStateTimestamps[i][startindex][0], OriginalStateTimestamps[i][sampleindex[0]]-OriginalStateTimestamps[i][startindex][0]),
            (int(key[0]), int(key[1]), i, 'go', OriginalStateTimestamps[i][responseindex][0], AnswerPeriod)])
            for j in range(0, len(sampleindex)):
                TrialEvent().insert1((int(key[0]), int(key[1]), i, 'sample', OriginalStateTimestamps[i][sampleindex[j]], SampleDur))
            for j in range(0, len(delayindex)):
                TrialEvent().insert1((int(key[0]), int(key[1]), i, 'delay', OriginalStateTimestamps[i][delayindex[j]], DelayDur))
            for j in range(0, len(lickleft)):
                ActionEvent().insert1((int(key[0]), int(key[1]), i, 'left lick', OriginalEventTimestamps[i][lickleft[j]]))
            for j in range(0, len(lickright)):
                ActionEvent().insert1((int(key[0]), int(key[1]), i, 'right lick', OriginalEventTimestamps[i][lickright[j]]))

@schema 
class TrialNoteType(dj.Lookup):
    definition = """
    trial_note_type : varchar(12)
    """
    contents = zip(('autolearn', 'protocol #', 'bad'))

@schema
class TrialNote(dj.Manual):
    definition = """
    -> Session.Trial
    -> TrialNoteType
    ---
    trial_note  : varchar(255) 
    """

@schema
class TaskTraining(dj.Manual):
    definition = """
    -> Session
    -> Task
    ----
    training_status :  enum('naive', 'expert', 'overtrained')   
    """

@schema
class TrialEventType(dj.Lookup):
    definition = """
    trial_event_type  : varchar(12)  
    """
    contents = zip(('delay', 'go', 'sample', 'presample', 'bitcode'))


@schema
class Outcome(dj.Lookup):
    definition = """
    outcome : varchar(8)
    """
    contents = zip(('hit', 'miss', 'ignore'))


@schema 
class EarlyLick(dj.Lookup):
    definition = """
    early_lick  :  varchar(12)
    """ 
    contents = zip(('early', 'no early'))

     
@schema 
class TrialInstruction(dj.Lookup):
    definition = """
    # Instruction to mouse 
    trial_instruction  : varchar(8) 
    """
    contents = zip(('left', 'right'))

@schema
class BehaviorTrial(dj.Manual):
    definition = """
    -> Session.Trial
    ----
    -> Task
    -> TrialInstruction
    -> EarlyLick
    -> Outcome
    """

@schema
class TrialEvent(dj.Manual):
    definition = """
    -> BehaviorTrial 
    -> TrialEventType
    trial_event_time : decimal(8, 4)   # (s) from trial start, not session start (depends)
    ---
    duration : decimal(8,4)  #  (s)  
    """

@schema
class ActionEventType(dj.Lookup):
    definition = """
    action_event_type : varchar(12)
    ----
    action_event_description : varchar(1000)
    """
    contents =[  
       ('left lick', ''), 
       ('right lick', '')]

@schema
class ActionEvent(dj.Manual):
    definition = """
    -> BehaviorTrial
    -> ActionEventType
    action_event_time : decimal(8,4)  # (s) from trial or session (it depends but please figure it out)
    """

@schema
class TrackingDevice(dj.Lookup):
    definition = """
    tracking_device  : varchar(8)  # e.g. camera, microphone
    ---
    sampling_rate  :  decimal(8, 4)   # Hz
    """

@schema
class Tracking(dj.Imported):
    definition = """
    -> Session.Trial 
    -> TrackingDevice
    ---
    tracking_data_path  : varchar(255)
    """

@schema
class PhotostimDevice(dj.Lookup):
    definition = """
    photostim_device  : varchar(20)
    ---
    excitation_wavelength :  decimal(5,1)  # (nm) 
    photostim_device_description : varchar(255)
    """

@schema 
class Photostim(dj.Lookup):
    definition = """
    -> PhotostimDevice
    photo_stim :  smallint 
    ---
    -> ccf.CCF
    duration  :  decimal(8,4)   # (s)
    waveform  :  longblob       # (mW)
    """

    class Profile(dj.Part):
        definition = """
        -> master
        (profile_x, profile_y, profile_z) -> ccf.CCF(x, y, z)
        ---
        intensity_timecourse   :  longblob  # (mW/mm^2)
        """


@schema
class PhotostimTrial(dj.Imported):
    definition = """
    -> Session.Trial
    """

    class Event(dj.Part):
        definition = """
        -> master
        -> Photostim
        photostim_event_time : decimal(8,3)   # (s) from trial or session start or whatever 
        """

@schema
class PassivePhotostimTrial(dj.Computed):
    definition = """
    -> PhotostimTrial
    """
    key_source = PhotostimTrial() - BehaviorTrial()

    def make(self, key):
        self.insert1(key)
