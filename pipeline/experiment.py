import logging
import datajoint as dj
import numpy as np
import pathlib
from datetime import datetime

from . import lab, ccf
from . import get_schema_name, create_schema_settings

schema = dj.schema(get_schema_name('experiment'), **create_schema_settings)

ephys = dj.create_virtual_module('ephys', get_schema_name('ephys'))  # avoid circular dependency


log = logging.getLogger(__name__)


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
    task            : varchar(24)                  # task type
    ----
    task_description : varchar(4000)
    """
    contents = [
        ('audio delay', 'auditory delayed response task (2AFC)'),
        ('audio mem', 'auditory working memory task'),
        ('s1 stim', 'S1 photostimulation task (2AFC)'),
        ('foraging', 'foraging task based on Bari-Cohen 2019'),
        ('foraging 3lp', 'foraging task based on Bari-Cohen 2019 with variable delay period'),
        ('multi-target-licking', 'multi-target-licking task')
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
        ('foraging 3lp', 101, 'moving lickports, delay period, early lick punishment, sound GO cue then free choice from three lickports'),
        ('multi-target-licking', 1, 'multi-target-licking task - 2D'),
        ('multi-target-licking', 2, 'multi-target-licking task - Spontaneous')
    ]


@schema
class WaterPort(dj.Lookup):
    definition = """
    water_port: varchar(16)  # e.g. left, right, middle, top-left, purple
    """

    contents = zip(['left', 'right', 'middle',
                    'mtl-1', 'mtl-2', 'mtl-3',    # The "mtl-" refers to multi-target-licking
                    'mtl-4', 'mtl-5', 'mtl-6',    # water ports, arranged in a 3x3 "number-pad"
                    'mtl-7', 'mtl-8', 'mtl-9'])   # fashion, with the numbering as shown here


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
    contents = zip(('autolearn', 'protocol #', 'bad',
                    'bitcode', 'autowater', 'random_seed_start'))


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
    contents = zip(('hit', 'miss', 'ignore', 'N/A'))


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
    free_water=0: bool  # "empty" trial with water given (go-cue not played, no trial structure) 
    """


@schema
class WaterPortChoice(dj.Imported):
    definition = """  # The water port selected by the animal for each trial
    -> BehaviorTrial
    ---
    -> [nullable] WaterPort
    """


@schema
class TrialEventType(dj.Lookup):
    definition = """
    trial_event_type  : varchar(12)  
    """
    contents = zip(('delay', 'go', 'sample', 'presample', 'trialend',
                    'videostart', 'videoend', 'bitcodestart', 'choice', 'reward', 'doubledip',   # Added for foraging
                    'bpodstart', 'zaberstep', 'zaberready'    # Events only available from NIDQ channels
                    ))


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
    # Session block for Foraging experiment
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

# ---- Multi-target-licking paradigm specifics ----


@schema
class MultiTargetLickingSessionBlock(dj.Imported):
    definition = """
    # Session block for multi-target-licking experiments
    -> Session
    block : smallint 		# block number
    ---
    block_start_time : decimal(10, 4)  # (s) relative to session beginning
    trial_count: int                   # total number of trials in this block
    num_licks_for_reward: smallint     # how many licks are needed to trigger a water release. The mouse can collect the water drop on num_licks_for_reward +1 lick
    roll_deg: double                   # roll angle of the lickport positions in deg. Positve roll means the mouse is turning its head towards its right shoulder
    position_x_bins=3: int             # number of water port possible positions in X
    position_y_bins=1: int             # number of water port possible positions in Y
    position_z_bins=3: int             # number of water port possible positions in Z
    """

    class WaterPort(dj.Part):
        definition = """
        # the water port used for all trials in a given block
        -> master
        ---
        -> WaterPort
        position_x: float  # X position of this water port (in motor unit) - lateral
        position_y: float  # Y position of this water port (in motor unit) - rostrocaudal
        position_z: float  # Z position of this water port (in motor unit) - dorsoventral
        """

    class BlockTrial(dj.Part):
        definition = """
        -> master
        -> BehaviorTrial
        ---
        block_trial_number: int  # the ordering of this trial in this block
        """


@schema
class Breathing(dj.Imported):
    definition = """
    -> SessionTrial
    ---
    breathing: longblob
    breathing_timestamps: longblob  # (s) relative to the start of the trial
    """

    key_source = Session & ephys.ProbeInsertion & (BehaviorTrial & 'task = "multi-target-licking"')

    def make(self, key):
        # channel 2 is for breathing data
        breathing_trials_data = extract_nidq_trial_data(key, channel=2)
        for d in breathing_trials_data:
            d['breathing'] = d.pop('data')
            d['breathing_timestamps'] = d.pop('timestamps')

        self.insert(breathing_trials_data, allow_direct_insert=True)


@schema
class Piezoelectric(dj.Imported):
    definition = """
    -> SessionTrial
    ---
    piezoelectric: longblob
    piezoelectric_timestamps: longblob  # (s) relative to the start of the trial
    """

    key_source = Session & ephys.ProbeInsertion & (
                BehaviorTrial & 'task = "multi-target-licking"')

    def make(self, key):
        # channel 3 is for piezoelectric data
        piezoelectric_trials_data = extract_nidq_trial_data(key, channel=3)
        for d in piezoelectric_trials_data:
            d['piezoelectric'] = d.pop('data')
            d['piezoelectric_timestamps'] = d.pop('timestamps')

        self.insert(piezoelectric_trials_data, allow_direct_insert=True)


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


@schema
class PeriodForaging(dj.Lookup):
    # Totally different from delay-response task, so create a new table
    definition = """  # time period between any two TrialEvent (eg the delay period is between delay and go)
    period: varchar(30)
    ---
    -> TrialEventType.proj(start_event_type='trial_event_type')
    start_trial_shift=0: smallint  # see psth_foraging.AlignType.trial_offset 
    start_time_shift: float  # (s) any time-shift amount with respect to the start_event_type
    -> TrialEventType.proj(end_event_type='trial_event_type')
    end_trial_shift=0: smallint   # see psth_foraging.AlignType.trial_offset 
    end_time_shift: float    # (s) any time-shift amount with respect to the end_event_type
    """

    contents = [
        ('before_2', 'bitcodestart', 0, -2, 'bitcodestart', 0, 0),  # = iti_last_2 of the *last* trial
        ('delay', 'zaberready', 0, 0, 'go', 0, 0),
        ('go_to_end', 'go', 0, 0, 'trialend', 0, 0),
        ('go_1.2', 'go', 0, 0, 'go', 0, 1.2),  # Is ths reasonable?

        ('iti_all', 'trialend', 0, 0, 'bitcodestart', 1, 0),  # To be precise, should use 'zaberstart'??
        ('iti_first_2', 'trialend', 0, 0, 'trialend', 0, 2),
        ('iti_last_2', 'bitcodestart', 1, -2, 'bitcodestart', 1, 0),

        ('delay_bitcode', 'bitcodestart', 0, 0.146, 'go', 0, 0),
        # TODO ('delay_effective')   # If early lick, from the last lick before go cue to go cue
    ]


# ============================= PROJECTS ==================================================


@schema
class Project(dj.Lookup):
    definition = """
    project_name: varchar(128)
    ---
    project_desc='': varchar(1000) 
    publication='': varchar(256)  # e.g. publication doi    
    """

    contents = [('MAP', 'The Mesoscale Activity Map project', ''),
                ('Brain-wide neural activity underlying memory-guided movement',
                 'Brain-wide neural activity underlying memory-guided movement - 2023',
                 'https://doi.org/10.1101/2023.03.01.530520')]


@schema
class ProjectSession(dj.Manual):
    definition = """
    -> Project
    -> Session
    """


# ============================= HELPER METHODS ==========================

def get_session_ephys_data_directory(session_key):
    from pipeline.ingest import ephys as ephys_ingest

    h2o = (lab.WaterRestriction & session_key).fetch1('water_restriction_number')
    sess_datetime = (Session & session_key).proj(
        sess_datetime="cast(concat(session_date, ' ', session_time) as datetime)").fetch1(
        'sess_datetime')

    rigpaths = ephys_ingest.get_ephys_paths()
    for rigpath in rigpaths:
        session_ephys_dir, dglob = ephys_ingest.get_sess_dir(rigpath, h2o, sess_datetime)
        if session_ephys_dir is not None:
            break
    else:
        raise FileNotFoundError(
            'Error - No session folder found for {}/{}'.format(h2o, sess_datetime))

    return session_ephys_dir


def extract_nidq_trial_data(session_key, channel):
    from pipeline.ingest import ephys as ephys_ingest
    session_ephys_dir = get_session_ephys_data_directory(session_key)

    try:
        nidq_bin_fp = next(session_ephys_dir.glob('*.nidq.bin'))
    except StopIteration:
        raise FileNotFoundError('*.nidq.bin file not found in {}'.format(session_ephys_dir))

    ephys_bitcodes, trial_start_times = ephys_ingest.build_bitcode(session_ephys_dir)
    behav_trials, behavior_bitcodes = (TrialNote
                                       & {**session_key, 'trial_note_type': 'bitcode'}).fetch(
        'trial', 'trial_note', order_by='trial')

    if ephys_bitcodes is None:
        # no bitcodes (the nidq.XA.txt file for bitcode is not available)
        if (Session & session_key).fetch1('rig') == 'RRig-MTL' and len(trial_start_times) == len(behav_trials):
            # for MTL rig, this is glitch in the recording system
            # if the number of trial matches with behavior, then this is a 1-to-1 mapping
            log.info('.... Unable to read bitcode! Recognize RRig-MTL session with matching number of behavior and ephys trials, using one-to-one trial mapping')
            ephys_bitcodes = behavior_bitcodes
        else:
            raise FileNotFoundError('Generate bitcode failed. No bitcode file'
                                    ' "*.XA_0_0.txt"'
                                    ' found in {}'.format(session_ephys_dir))

    chan_list = [channel]
    breathing_data, sampling_rate = ephys_ingest.read_SGLX_bin(nidq_bin_fp, chan_list)

    trial_starts_indices = (trial_start_times * sampling_rate).astype(int)
    # segment to per-trial
    all_trials_data = []
    for idx in range(len(trial_starts_indices)):
        start_idx = trial_starts_indices[idx]
        end_idx = trial_starts_indices[idx + 1] if start_idx < trial_starts_indices[-1] else -1
        trial_data = breathing_data[:, start_idx:end_idx].flatten()

        ephys_bitcode = ephys_bitcodes[idx]
        matched_trial_idx = np.where(behavior_bitcodes == ephys_bitcode)[0]

        if len(matched_trial_idx):
            all_trials_data.append({
                **session_key, 'trial': behav_trials[matched_trial_idx[0]],
                'data': trial_data,
                'timestamps': np.arange(len(trial_data)) / sampling_rate})
    return all_trials_data


def get_wr_sessdatetime(key):
    water_res_num, session_datetime = (lab.WaterRestriction * Session.proj(
        session_datetime="cast(concat(session_date, ' ', session_time) as datetime)") & key).fetch1(
        'water_restriction_number', 'session_datetime')
    return water_res_num, datetime.strftime(session_datetime, '%Y%m%d_%H%M%S')