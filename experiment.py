import datajoint as dj
import lab, ccf

schema = dj.schema('daveliu_map_experi', locals())

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
    session : smallint 		# session number
    ---
    session_date  : date
    -> lab.Person
    -> lab.Rig
    behavior_file : varchar(255) # the behavior file name
    """
    
    
    class Trial(dj.Part):
        definition = """
        -> Session
        trial   : smallint
        ---
        start_time : decimal(6,4)  # (s)
        end_time : decimal(6,4)  # (s)
        """
		
		
    def _make_tuples(self, key, key1):
        import scipy.io as spio
        import numpy as np
        mat = spio.loadmat(key1, squeeze_me=True)
        SessionData=mat['SessionData']
        TrialTypes=SessionData.flatten()[0][0]
        RawData=SessionData.flatten()[0][7]
        TrialSettings=SessionData.flatten()[0][10]
        OriginalStateData=RawData.flatten()[0][1]
        OriginalStateTimestamps=RawData.flatten()[0][3]
        OriginalEventTimestamps=RawData.flatten()[0][4]
        
<<<<<<< HEAD
        
		
        for i in range(0, len(OriginalEventTimestamps)):
            trial_instruction = 'left'
            early_lick = 'no early'
            outcome = 'ignore'
            GUI = TrialSettings[i][0]
            ProtocolType = GUI.flatten()[0][10] # 1 Water-Valve-Calibration 2 Licking 3 Autoassist 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed

            if ProtocolType==4:
                itemindex = np.where(OriginalStateData[i]==15)
                if np.any(OriginalStateData[i]==11):
                    outcome = 'hit'
                elif np.any(OriginalStateData[i]==14):
                    outcome = 'miss'
                elif np.any(OriginalStateData[i]==13):
                    outcome = 'ignore'

            if ProtocolType==5:
                itemindex = np.where(OriginalStateData[i]==16)
                if np.any(OriginalStateData[i]==5):
                    early_lick = 'early'

=======
        trial_instruction = 'left'
        early_lick = 'no early'
        outcome = 'ignore'
		
        for i in range(0, len(OriginalEventTimestamps)):
            GUI = TrialSettings[i][0]

            ProtocolType = GUI.flatten()[0][10] # 1 Water-Valve-Calibration 2 Licking 3 Autoassist 4 No autoassist 5 DelayEnforce 6 SampleEnforce 7 Fixed

            if ProtocolType>5:
                itemindex = np.where(OriginalStateData[i]==17)
                if np.any(OriginalStateData[i]==4) or np.any(OriginalStateData[i]==6):
                    early_lick = 'early'
                if np.any(OriginalStateData[i]==13):
                    outcome = 'hit'
                elif np.any(OriginalStateData[i]==16):
                    outcome = 'miss'
                elif np.any(OriginalStateData[i]==15):
                    outcome = 'ignore'
            elif ProtocolType>4:
                itemindex = np.where(OriginalStateData[i]==16)
                if np.any(OriginalStateData[i]==5):
                    early_lick = 'early'
>>>>>>> map-ephys
                if np.any(OriginalStateData[i]==12):
                    outcome = 'hit'
                elif np.any(OriginalStateData[i]==15):
                    outcome = 'miss'
                elif np.any(OriginalStateData[i]==14):
                    outcome = 'ignore'
<<<<<<< HEAD

            if ProtocolType>5:
                itemindex = np.where(OriginalStateData[i]==17)
                if np.any(OriginalStateData[i]==4) or np.any(OriginalStateData[i]==6):
                    early_lick = 'early'
                if np.any(OriginalStateData[i]==13):
                    outcome = 'hit'
                elif np.any(OriginalStateData[i]==16):
                    outcome = 'miss'
                elif np.any(OriginalStateData[i]==15):
=======
            elif ProtocolType>3:
                itemindex = np.where(OriginalStateData[i]==15)
                if np.any(OriginalStateData[i]==11):
                    outcome = 'hit'
                elif np.any(OriginalStateData[i]==14):
                    outcome = 'miss'
                elif np.any(OriginalStateData[i]==13):
>>>>>>> map-ephys
                    outcome = 'ignore'
            else:
                itemindex = np.where(OriginalStateData[i]==1)

            Session.Trial().insert1((int(key[0]), int(key[1]), i, OriginalEventTimestamps[i][1], OriginalStateTimestamps[i][itemindex][0]))
            
            if TrialTypes[i]==0:
                trial_instruction = 'left'
            elif TrialTypes[i]==1:
                trial_instruction = 'right'
            

            BehaviorTrial().insert1((int(key[0]), int(key[1]), i, 'audio delay', trial_instruction, early_lick, outcome))
            TrialNote().insert1((int(key[0]), int(key[1]), i, 'protocol #', str(ProtocolType)))

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
    trial_event_time : decimal(8, 3)   # (s) from session and trial start (depends)
    ---
    duration : decimal(8,3)  #  (s)  
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
    action_event_time : decimal(8,3)  # (s) from trial or session (it depends but please figure it out)
    """

@schema
class TrackingDevice(dj.Lookup):
    definition = """
    tracking_device  : varchar(8)  # e.g. camera, microphone
    ---
    sampling_rate  :  decimal(9, 3)   # Hz
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
    duration  :  decimal(5,3)   # (s)
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
