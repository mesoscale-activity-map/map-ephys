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
    session : smallint 		# session number
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
