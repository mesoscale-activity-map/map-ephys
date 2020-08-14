
import datajoint as dj
import numpy as np

from . import lab, ccf
from . import get_schema_name

schema = dj.schema(get_schema_name('experiment'))


@schema
class Session(dj.Manual):
    definition = """
    -> lab.Subject
    session: smallint 		# session number
    ---
    session_date: date
    session_time: time
    unique index (subject_id, session_date, session_time)
    -> lab.Person
    -> lab.Rig
    """


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
        ('s1 stim', 'S1 photostimulation task (2AFC)'),
        ('foraging', 'foraging task based on Bari-Cohen 2019'),
        ('foraging 3lp', 'foraging task based on Bari-Cohen 2019 with variable delay period')
    ]


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
        ('s1 stim', 8, 'mini-distractors and full distractors (only at late delay), with different levels of the mini-stim and the full-stim during sample period'),
        ('s1 stim', 9, 'mini-distractors and full distractors (only at late delay), with different levels of the mini-stim and the full-stim during sample period'),
        ('foraging', 100, 'moving lickports, delay period, early lick punishment, sound GO cue then free choice'),
        ('foraging 3lp', 101, 'moving lickports, delay period, early lick punishment, sound GO cue then free choice from three lickports')
    ]


@schema
class WaterPort(dj.Lookup):
    definition = """
    water_port: varchar(16)  # e.g. left, right, middle, top-left, purple
    """

    contents = zip(['left', 'right', 'middle'])


@schema
class Photostim(dj.Manual):
    definition = """  # Photostim protocol
    -> Session
    photo_stim :  smallint  # photostim protocol number
    ---
    -> lab.PhotostimDevice
    duration=null:  decimal(8,4)   # (s)
    waveform=null:  longblob       # normalized to maximal power. The value of the maximal power is specified for each PhotostimTrialEvent individually
    """

    class PhotostimLocation(dj.Part):
        definition = """
        -> master
        -> lab.SkullReference
        ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
        ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
        depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
        theta:       decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
        phi:         decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
        ---
        -> lab.BrainArea           # target brain area for photostim 
        """

    class Profile(dj.Part):
        # NOT USED CURRENT
        definition = """
        -> master
        (profile_x, profile_y, profile_z) -> ccf.CCF(ccf_x, ccf_y, ccf_z)
        ---
        intensity_timecourse   :  longblob  # (mW/mm^2)
        """


@schema
class PhotostimBrainRegion(dj.Computed):
    definition = """
    -> Photostim
    ---
    -> lab.BrainArea.proj(stim_brain_area='brain_area')
    stim_laterality: enum('left', 'right', 'both')
    """

    def make(self, key):
        brain_areas, ml_locations = (Photostim.PhotostimLocation & key).fetch('brain_area', 'ml_location')
        ml_locations = ml_locations.astype(float)
        if len(set(brain_areas)) > 1:
            raise ValueError('Multiple different brain areas for one photostim protocol is unsupported')
        if (ml_locations > 0).any() and (ml_locations < 0).any():
            lat = 'both'
        elif (ml_locations > 0).all():
            lat = 'right'
        elif (ml_locations < 0).all():
            lat = 'left'
        else:
            assert (ml_locations == 0).all()  # sanity check
            raise ValueError('Ambiguous hemisphere: ML locations are all 0...')

        self.insert1(dict(key, stim_brain_area=brain_areas[0], stim_laterality=lat))


@schema
class SessionTrial(dj.Imported):
    definition = """
    -> Session
    trial : smallint 		# trial number (1-based indexing)
    ---
    trial_uid : int  # unique across sessions/animals
    start_time : decimal(9, 4)  # (s) relative to session beginning 
    stop_time : decimal(9, 4)  # (s) relative to session beginning 
    """


@schema 
class TrialNoteType(dj.Lookup):
    definition = """
    trial_note_type : varchar(24)
    """
    contents = zip(('autolearn', 'protocol #', 'bad', 'bitcode', 'autowater', 'random_seed_start'))


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


@schema
class SessionDetails(dj.Manual):
    definition = """
    -> Session
    ---
    session_weight : decimal(8,4) # weight of the mouse at the beginning of the session
    session_water_earned : decimal(8,4) # water earned by the mouse during the session
    session_water_extra : decimal(8,4) # extra water provided after the session
    """


# ---- behavioral trials ----

@schema
class TrialInstruction(dj.Lookup):
    definition = """
    # Instruction to mouse 
    trial_instruction  : varchar(8) 
    """
    contents = zip(('left', 'right', 'middle', 'none'))


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
class BehaviorTrial(dj.Imported):
    definition = """
    -> SessionTrial
    ----
    -> TaskProtocol
    -> TrialInstruction
    -> EarlyLick
    -> Outcome
    auto_water=0: bool  # water given after response-time, regardless of correct/incorrect
    free_water=0: bool  # "empty" trial with water given
    """


@schema
class WaterPortChoice(dj.Imported):
    definition = """  # The water port selected by the animal for each trial
    -> BehaviorTrial
    ---
    [nullable] -> WaterPort
    """


@schema
class TrialEventType(dj.Lookup):
    definition = """
    trial_event_type  : varchar(12)  
    """
    contents = zip(('delay', 'go', 'sample', 'presample', 'trialend'))


@schema
class TrialEvent(dj.Imported):
    definition = """
    -> BehaviorTrial 
    trial_event_id: smallint
    ---
    -> TrialEventType
    trial_event_time : decimal(8, 4)   # (s) from trial start, not session start
    duration : decimal(8,4)  #  (s)  
    """


@schema
class ActionEventType(dj.Lookup):
    definition = """
    action_event_type : varchar(32)
    ----
    action_event_description : varchar(1000)
    """

    contents = [('left lick', ''),
                ('right lick', ''),
                ('middle lick', '')]


@schema
class ActionEvent(dj.Imported):
    definition = """
    -> BehaviorTrial
    action_event_id: smallint
    ---
    -> ActionEventType
    action_event_time : decimal(8,4)  # (s) from trial start
    """


# ---- Foraging paradigm specifics ----

@schema
class SessionBlock(dj.Imported):
    definition = """
    -> Session
    block : smallint 		# block number
    ---
    block_start_time : decimal(10, 4)  # (s) relative to session beginning
    """

    class WaterPortRewardProbability(dj.Part):
        definition = """
        -> master
        -> WaterPort
        ---
        reward_probability: decimal(8, 4)
        """

    class BlockTrial(dj.Part):
        definition = """
        -> master
        -> BehaviorTrial
        """


@schema
class WaterPortSetting(dj.Imported):
    definition = """  
    -> BehaviorTrial
    ----
    water_port_lateral_pos=null: int # position value of the motor
    water_port_rostrocaudal_pos=null: int # position value of the motor
    water_port_dorsoventral_pos=null: int # position value of the motor
    """

    class OpenDuration(dj.Part):
        definition = """
        -> master
        -> WaterPort
        ---
        open_duration: decimal(5, 4)  # (s) duration of port open time
        """


@schema
class TrialAvailableReward(dj.Imported):
    definition = """ # available reward (bool) for each water port per trial
    -> BehaviorTrial
    -> WaterPort
    ---
    reward_available: bool
    """


# ---- Photostim trials ----

@schema
class PhotostimTrial(dj.Imported):
    definition = """
    -> SessionTrial
    """


@schema
class PhotostimEvent(dj.Imported):
    definition = """
    -> PhotostimTrial
    photostim_event_id: smallint
    ---
    -> Photostim
    photostim_event_time : decimal(8,3)   # (s) from trial start
    power : decimal(8,3)   # Maximal power (mW)
    """


@schema
class PassivePhotostimTrial(dj.Computed):
    definition = """
    -> SessionTrial
    """
    key_source = PhotostimTrial() - BehaviorTrial()

    def make(self, key):
        self.insert1(key)

# ----


@schema
class Period(dj.Lookup):
    definition = """  # time period between any two TrialEvent (eg the delay period is between delay and go)
    period: varchar(12)
    ---
    -> TrialEventType.proj(start_event_type='trial_event_type')
    start_time_shift: float  # (s) any time-shift amount with respect to the start_event_type
    -> TrialEventType.proj(end_event_type='trial_event_type')
    end_time_shift: float    # (s) any time-shift amount with respect to the end_event_type
    """

    contents = [('sample', 'sample', 0, 'delay', 0),
                ('delay', 'delay', 0, 'go', 0),
                ('response', 'go', 0, 'go', 1.2)]

# ============================= PROJECTS ==================================================


@schema
class Project(dj.Lookup):
    definition = """
    project_name: varchar(128)
    ---
    project_desc='': varchar(1000) 
    publication='': varchar(256)  # e.g. publication doi    
    """

    contents = [('MAP', 'The Mesoscale Activity Map project', '')]
