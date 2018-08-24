import datajoint as dj

from . import lab
from . import ccf

schema = dj.schema(dj.config.get('experiment.database', 'map_experiment'))


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
    -> lab.Subject
    session : smallint 		# session number
    ---
    session_date  : date
    -> lab.Person
    -> lab.Rig
    """

@schema
class SessionTrial(dj.Imported):
    definition = """
    -> Session
    trial : smallint 		# trial number
    ---
    trial_uid : int  # unique across sessions/animals
    start_time : decimal(8,4)  # (s) relative to session beginning
    """
    
@schema 
class TrialNoteType(dj.Lookup):
    definition = """
    trial_note_type : varchar(12)
    """
    contents = zip(('autolearn', 'protocol #', 'bad', 'bitcode'))

@schema
class TrialNote(dj.Imported):
    definition = """
    -> SessionTrial
    -> TrialNoteType
    ---
    trial_note  : varchar(255) 
    """

@schema
class TrainingType(dj.Lookup):
    definition = """
    # Mouse training
    training_type : varchar(100) # mouse training
    ---
    training_type_description : varchar(2000) # description
    """
    contents = [
         ('regular', ''),
         ('regular + distractor', 'mice were first trained on the regular S1 photostimulation task  without distractors, then the training continued in the presence of distractors'),
         ('regular or regular + distractor', 'includes both training options')
         ]
    
@schema
class SessionTraining(dj.Manual):
    definition = """
    -> Session
    -> TrainingType
    """

@schema
class TrialEventType(dj.Lookup):
    definition = """
    trial_event_type  : varchar(12)  
    """
    contents = zip(('delay', 'go', 'sample', 'presample'))

@schema
class Outcome(dj.Lookup):
    definition = """
    outcome : varchar(32)
    """
    contents = zip(('hit', 'miss', 'ignore'))

@schema 
class EarlyLick(dj.Lookup):
    definition = """
    early_lick  :  varchar(32)
    ---
    early_lick_description : varchar(4000)
    """ 
    contents = [
        ('early', 'early lick during sample and/or delay'),
        ('early, presample only', 'early lick in the presample period, after the onset of the scheduled wave but before the sample period'),
        ('no early', '')]

     
@schema 
class TrialInstruction(dj.Lookup):
    definition = """
    # Instruction to mouse 
    trial_instruction  : varchar(8) 
    """
    contents = zip(('left', 'right'))

@schema
class TaskProtocol(dj.Lookup):
    definition = """
    # SessionType
    -> Task
    task_protocol : tinyint # task protocol
    ---
    task_protocol_description : varchar(4000)
    """
    contents = [
         ('audio delay', 1, 'high tone vs. low tone'),
         ('s1 stim', 2, 'mini-distractors'),
         ('s1 stim', 3, 'full distractors, with 2 distractors (at different times) on some of the left trials'),
         ('s1 stim', 4, 'full distractors'),
         ('s1 stim', 5, 'mini-distractors, with different levels of the mini-stim during sample period'),
         ('s1 stim', 6, 'full distractors; same as protocol 4 but with a no-chirp trial-type'),
         ('s1 stim', 7, 'mini-distractors and full distractors (only at late delay)'),
         ('s1 stim', 8, 'mini-distractors and full distractors (only at late delay), with different levels of the mini-stim and the full-stim during sample                 period'),
         ('s1 stim', 9, 'mini-distractors and full distractors (only at late delay), with different levels of the mini-stim and the full-stim during sample period')
         ]

@schema
class BehaviorTrial(dj.Imported):
    definition = """
    -> SessionTrial
    ----
    -> TaskProtocol
    -> TrialInstruction
    -> EarlyLick
    -> Outcome
    """

@schema
class TrialEvent(dj.Manual):
    definition = """
    -> BehaviorTrial 
    -> TrialEventType
    trial_event_time : decimal(8, 4)   # (s) from trial start, not session start
    ---
    duration : decimal(8,4)  #  (s)  
    """

@schema
class ActionEventType(dj.Lookup):
    definition = """
    action_event_type : varchar(32)
    ----
    action_event_description : varchar(1000)
    """
    contents =[  
       ('left lick', ''), 
       ('right lick', '')]

@schema
class ActionEvent(dj.Imported):
    definition = """
    -> BehaviorTrial
    -> ActionEventType
    action_event_time : decimal(8,4)  # (s) from trial start
    """

@schema
class TrackingDevice(dj.Lookup):
    definition = """
    tracking_device  : varchar(20)  # e.g. camera, microphone
    ---
    sampling_rate  :  decimal(8, 4)   # Hz
    tracking_device_description: varchar(100) # 
    """
    contents =[  
       ('Camera 0, side', 400, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
       ('Camera 1, bottom', 400, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)')]

@schema
class Tracking(dj.Imported):
    definition = """
    -> SessionTrial 
    -> TrackingDevice
    ---
    tracking_data_path  : varchar(1000)
    start_time = null : decimal(8,4) # (s) from trial start (which should coincide with the beginning of the ephys recordings)
    duration = null : decimal(8,4)                   # (s)
    """

@schema
class PhotostimDevice(dj.Lookup):
    definition = """
    photostim_device  : varchar(20)
    ---
    excitation_wavelength :  decimal(5,1)  # (nm) 
    photostim_device_description : varchar(255)
    """
    contents =[  
       ('LaserGem473', 473, 'Laser (Laser Quantum, Gem 473)'), 
       ('LED470', 470, 'LED (Thor Labs, M470F3 - 470 nm, 17.2 mW (Min) Fiber-Coupled LED)'),
       ('OBIS470', 473, 'OBIS 473nm LX 50mW Laser System: Fiber Pigtail (Coherent Inc)')]

@schema 
class Photostim(dj.Imported):
    definition = """
    -> PhotostimDevice
    photo_stim :  smallint 
    ---
    -> ccf.CCF
    duration  :  decimal(8,4)   # (s)
    waveform  :  longblob       # normalized to maximal power. The value of the maximal power is specified for each PhotostimTrialEvent individually
    """

    class Profile(dj.Part):
        definition = """
        -> master
        (profile_x, profile_y, profile_z) -> ccf.CCF(x, y, z)
        ---
        intensity_timecourse   :  longblob  # (mW/mm^2)
        """

@schema
class PhotostimLocation(dj.Manual):
    definition = """
    -> Photostim
    ---
    -> lab.Hemisphere
    -> lab.BrainArea
    -> lab.SkullReference
    photostim_ml_location = null           : decimal(8,3)       # um from ref ; right is positive; 
    photostim_ap_location = null           : decimal(8,3)       # um from ref; anterior is positive; 
    photostim_dv_location = null           : decimal(8,3)       # um from dura; ventral is positive; 
    photostim_ml_angle = null              : decimal(8,3)       # Angle between the photostim path and the Medio-Lateral axis. A tilt towards the right hemishpere is positive. 
    photostim_ap_angle = null              : decimal(8,3)       # Angle between the photostim path and the Anterior-Posterior axis. An anterior tilt is positive. 
    """

@schema
class PhotostimTrial(dj.Imported):
    definition = """
    -> SessionTrial
    """

    class Event(dj.Part):
        definition = """
        -> master
        -> Photostim
        photostim_event_time : decimal(8,3)   # (s) from trial start
        power : decimal(8,3)   # Maximal power (mW)
        """

@schema
class PassivePhotostimTrial(dj.Computed):
    definition = """
    -> PhotostimTrial
    """
    key_source = PhotostimTrial() - BehaviorTrial()

    def make(self, key):
        self.insert1(key)

@schema
class SessionTask(dj.Manual):
    definition = """
    -> Session
    -> TaskProtocol
    """

@schema
class SessionComment(dj.Manual):
    definition = """
    -> Session
    session_comment : varchar(767)
    """
