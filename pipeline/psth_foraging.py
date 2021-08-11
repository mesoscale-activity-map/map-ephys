import logging
from functools import partial
from inspect import getmembers
import numpy as np
import datajoint as dj

from . import (lab, experiment, ephys)
[lab, experiment, ephys]  # NOQA

from . import get_schema_name, dict_to_hash

schema = dj.schema(get_schema_name('psth_foraging'))
log = logging.getLogger(__name__)

# NOW:
# - rework Condition to TrialCondition funtion+arguments based schema

# The new psth_foraging schema is only for foraging sessions. 
foraging_sessions = experiment.Session & (experiment.BehaviorTrial & 'task LIKE "foraging%"')


@schema
class TrialCondition(dj.Lookup):
    '''
    TrialCondition: Manually curated condition queries.

    Used to define sets of trials which can then be keyed on for downstream
    computations.
    '''

    definition = """
    trial_condition_name:       varchar(128)     # user-friendly name of condition
    ---
    trial_condition_hash:       varchar(32)     # trial condition hash - hash of func and arg
    unique index (trial_condition_hash)
    trial_condition_func:       varchar(36)     # trial retrieval function
    trial_condition_arg:        longblob        # trial retrieval arguments
    """

    @property
    def contents(self):
        contents_data = [
                     
            # ----- Foraging task -------
            {
                'trial_condition_name': 'L_hit_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'L_miss_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_hit_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right',
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_miss_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right',
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },            {
                'trial_condition_name': 'L_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'outcome': 'hit',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'L_miss',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'outcome': 'miss',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right',
                    'outcome': 'hit',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_miss',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right',
                    'outcome': 'miss',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'LR_hit_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'LR_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'outcome': 'hit',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'LR_miss_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'foraging',
                    'task_protocol': 100,
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'LR_all_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'foraging',
                    'task_protocol': 100,
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'L_all_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'left',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            },
            {
                'trial_condition_name': 'R_all_noearlylick',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'foraging',
                    'task_protocol': 100,
                    'water_port': 'right', 
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0
                    }
            }
            
        ]

        # PHOTOSTIM conditions. Not implemented for now 
        
        # stim_locs = [('left', 'alm'), ('right', 'alm'), ('both', 'alm')]
        # for hemi, brain_area in stim_locs:
        #     for instruction in (None, 'left', 'right'):
        #         condition = {'trial_condition_name': '_'.join(filter(None, ['all', 'noearlylick',
        #                                                                     '_'.join([hemi, brain_area]), 'stim',
        #                                                                     instruction])),
        #                      'trial_condition_func': '_get_trials_include_stim',
        #                      'trial_condition_arg': {
        #                          **{'_outcome': 'ignore',
        #                             'task': 'audio delay',
        #                             'task_protocol': 1,
        #                             'early_lick': 'no early',
        #                             'auto_water': 0,
        #                             'free_water': 0,
        #                             'stim_laterality': hemi,
        #                             'stim_brain_area': brain_area},
        #                          **({'trial_instruction': instruction} if instruction else {})}
        #                      }
        #         contents_data.append(condition)

        return ({**d, 'trial_condition_hash':
            dict_to_hash({'trial_condition_func': d['trial_condition_func'],
                          **d['trial_condition_arg']})}
                for d in contents_data)

    @classmethod
    def get_trials(cls, trial_condition_name, trial_offset=0):
        return cls.get_func({'trial_condition_name': trial_condition_name}, trial_offset)()

    @classmethod
    def get_cond_name_from_keywords(cls, keywords):
        matched_cond_names = []
        for cond_name in cls.fetch('trial_condition_name'):
            match = True
            tmp_cond = cond_name
            for k in keywords:
                if k in tmp_cond:
                    tmp_cond = tmp_cond.replace(k, '')
                else:
                    match = False
                    break
            if match:
                matched_cond_names.append(cond_name)
        return sorted(matched_cond_names)

    @classmethod
    def get_func(cls, key, trial_offset=0):
        self = cls()

        func, args = (self & key).fetch1(
            'trial_condition_func', 'trial_condition_arg')

        return partial(dict(getmembers(cls))[func], trial_offset, **args)

    @classmethod
    def _get_trials_exclude_stim(cls, trial_offset, **kwargs):
        # Note: inclusion (attr) is AND - exclusion (_attr) is OR
        log.debug('_get_trials_exclude_stim: {}'.format(kwargs))

        restr, _restr = {}, {}
        for k, v in kwargs.items():
            if k.startswith('_'):
                _restr[k[1:]] = v
            else:
                restr[k] = v

        stim_attrs = set((experiment.Photostim * experiment.PhotostimBrainRegion
                          * experiment.PhotostimEvent).heading.names) - set(experiment.Session.heading.names)
        behav_attrs = set((experiment.BehaviorTrial * experiment.WaterPortChoice).heading.names)

        _stim_key = {k: v for k, v in _restr.items() if k in stim_attrs}
        _behav_key = {k: v for k, v in _restr.items() if k in behav_attrs}

        stim_key = {k: v for k, v in restr.items() if k in stim_attrs}
        behav_key = {k: v for k, v in restr.items() if k in behav_attrs}
        
        q = (((experiment.BehaviorTrial * experiment.WaterPortChoice & behav_key) - [{k: v} for k, v in _behav_key.items()]) -
                ((experiment.PhotostimEvent * experiment.PhotostimBrainRegion * experiment.Photostim & stim_key)
                 - [{k: v} for k, v in _stim_key.items()]).proj())
        
        if trial_offset:
            return experiment.BehaviorTrial & q.proj(_='trial', trial=f'trial + {trial_offset}')
        else:
            return q

    @classmethod
    def _get_trials_include_stim(cls, trial_offset, **kwargs):
        # Note: inclusion (attr) is AND - exclusion (_attr) is OR
        log.debug('_get_trials_include_stim: {}'.format(kwargs))

        restr, _restr = {}, {}
        for k, v in kwargs.items():
            if k.startswith('_'):
                _restr[k[1:]] = v
            else:
                restr[k] = v

        stim_attrs = set((experiment.Photostim * experiment.PhotostimBrainRegion
                          * experiment.PhotostimEvent).heading.names) - set(experiment.Session.heading.names)
        behav_attrs = set((experiment.BehaviorTrial * experiment.WaterPortChoice).heading.names)

        _stim_key = {k: v for k, v in _restr.items() if k in stim_attrs}
        _behav_key = {k: v for k, v in _restr.items() if k in behav_attrs}

        stim_key = {k: v for k, v in restr.items() if k in stim_attrs}
        behav_key = {k: v for k, v in restr.items() if k in behav_attrs}

        q = (((experiment.BehaviorTrial * experiment.WaterPortChoice & behav_key) - [{k: v} for k, v in _behav_key.items()]) &
                ((experiment.PhotostimEvent * experiment.PhotostimBrainRegion * experiment.Photostim & stim_key)
                 - [{k: v} for k, v in _stim_key.items()]).proj())
    
        if trial_offset:
            return experiment.BehaviorTrial & q.proj(_='trial', trial=f'trial + {trial_offset}')
        else:
            return q


@schema
class UnitPsth(dj.Computed):
    definition = """
    -> TrialCondition
    -> ephys.Unit
    ---
    unit_psth=NULL: longblob
    """
    psth_params = {'xmin': -3, 'xmax': 3, 'binsize': 0.04}

    @property
    def key_source(self):
        """
        For those conditions that include stim, process those with PhotostimBrainRegion already computed only
        Only units not of type "all"
        """
        nostim = (ephys.Unit * (TrialCondition & 'trial_condition_func = "_get_trials_exclude_stim"')
                  & 'unit_quality != "all"')
        stim = ((ephys.Unit & (experiment.Session & experiment.PhotostimBrainRegion))
                * (TrialCondition & 'trial_condition_func = "_get_trials_include_stim"') & 'unit_quality != "all"')
        return nostim.proj() + stim.proj() & foraging_sessions

    def make(self, key):
        log.debug('UnitPsth.make(): key: {}'.format(key))

        # expand TrialCondition to trials,
        trials = TrialCondition.get_trials(key['trial_condition_name'])

        # fetch related spike times
        q = (ephys.Unit.TrialSpikes & key & trials.proj())
        spikes = q.fetch('spike_times')

        if len(spikes) == 0:
            log.warning('no spikes found for key {} - null psth'.format(key))
            self.insert1(key)
            return

        # compute psth & store
        unit_psth = self.compute_psth(spikes)

        self.insert1({**key, 'unit_psth': unit_psth})

    @staticmethod
    def compute_psth(session_unit_spikes):
        spikes = np.concatenate(session_unit_spikes)

        xmin, xmax, bins = UnitPsth.psth_params.values()
        psth, edges = np.histogram(spikes, bins=np.arange(xmin, xmax, bins))
        psth = psth / len(session_unit_spikes) / bins

        return np.array([psth, edges[1:]])

    @classmethod
    def get_plotting_data(cls, unit_key, condition_key):
        """
        Retrieve / build data needed for a Unit PSTH Plot based on the given
        unit condition and included / excluded condition (sub-)variables.
        Returns a dictionary of the form:
          {
             'trials': ephys.Unit.TrialSpikes.trials,
             'spikes': ephys.Unit.TrialSpikes.spikes,
             'psth': UnitPsth.unit_psth,
             'raster': Spike * Trial raster [np.array, np.array]
          }
        """
        # from sys import exit as sys_exit  # NOQA
        # from code import interact
        # from collections import ChainMap
        # interact('unitpsth make', local=dict(ChainMap(locals(), globals())))

        trials = TrialCondition.get_func(condition_key)()
 
        unit_psth = (UnitPsth & {**condition_key, **unit_key}).fetch1('unit_psth')
        if unit_psth is None:
            raise Exception('No spikes found for this unit and trial-condition')

        spikes, trials = (ephys.Unit.TrialSpikes & trials & unit_key).fetch(
            'spike_times', 'trial', order_by='trial asc')

        raster = [np.concatenate(spikes),
                  np.concatenate([[t] * len(s)
                                  for s, t in zip(spikes, trials)])]

        return dict(trials=trials, spikes=spikes, psth=unit_psth, raster=raster)
    
    
@schema 
class AlignType(dj.Lookup):
    """
    Define flexible psth alignment types for the foraging task.

    For the previous delay-response task, we only align spikes to the go cue, and all PSTH has been aligned during ephys
    ingestion (see `ephys.Unit.TrialSpikes`).

    Here, for the foraging task, we would like to align spikes to any event, such as go cue, choice (reward), ITI, and
    even the next trial's trial start. Because of this flexibility, we chose not to do alignment during ingestion, but
    use this table to define all possible alignment types we want, and compute the PSTHs on-the-fly.

    Notes:
    1. When we fetch event times for `experiment.TrialEventType`, we use NI time (`ephys.TrialEvent`) instead of bpod
       time (`experiment.TrialEvent`). See `compute_unit_psth_and_raster()`. See also
       `ingest.util.compare_ni_and_bpod_times()` for a comparison between NI and bpod time and why NI time is more
       reliable.

    2. `trial_offset` is used to specify alignment type like the trial start of the *next* trial. This is
       implemented by adding a `trial_offset` in the method `TrialCondition.get_trials()` above.

       Say, we want to compute PSTH conditioned on all rewarded trials and aligned to the next trial start. Assuming
       trials #1, 5, and 10 are the rewarded trials, then with `trial_offset` = 1, `TrialCondition.get_trials()` will
       return trials #2, 6, and 11. Using this shifted `trial_keys`, `compute_unit_psth_and_raster` will effectively
       align the spikes to the *next* trial start of all rewarded trials.

    """
    
    definition = """
    align_type_name: varchar(32)   # user-friendly name of alignment type
    -> experiment.TrialEventType
    ---
    align_type_description='':    varchar(256)    # description of this align type
    trial_offset=0:      smallint         # e.g., offset = 1 means the psth will be aligned to the event of the *next* trial.
    time_offset=0:       Decimal(10, 5)   # will be added to the event time for manual correction (e.g., bitcodestart to actual zaberready)  
    psth_win:            tinyblob    # [t_min, t_max], time window by which `compute_unit_psth_and_raster` counts spikes
    xlim:                tinyblob    # [x_min, x_max], default xlim for plotting PSTH (could be overridden during plotting)
    """
    contents = [
        ['trial_start', 'zaberready', '', 0, 0, [-3, 2], [-2, 1]],
        ['go_cue', 'go', '', 0, 0, [-2, 5], [-1, 3]],
        ['first_lick_after_go_cue', 'choice', 'first non-early lick', 0, 0, [-2, 5], [-1, 3]],
        ['iti_start', 'trialend', '', 0, 0, [-3, 10], [-2, 5]],
        ['next_trial_start', 'zaberready', '', 1, 0, [-10, 3], [-8, 1]],
        ['next_two_trial_start', 'zaberready', '', 2, 0, [-10, 5], [-8, 3]],

        # In the first few sessions, zaber moter feedback is not recorded,
        # so we don't know the exact time of trial start ('zaberready').
        # We have to estimate actual trial start by
        #   bitcodestart + bitcode width (42 ms for first few sessions) + zaber movement duration (~ 104 ms, very repeatable)
        ['trial_start_bitcode', 'bitcodestart', 'estimate actual trial start by bitcodestart + 146 ms', 0, 0.146, [-3, 2], [-2, 1]],
        ['next_trial_start_bitcode', 'bitcodestart', 'estimate actual trial start by bitcodestart + 146 ms', 1, 0.146, [-10, 3], [-8, 1]],
        ['next_two_trial_start_bitcode', 'bitcodestart', '', 2, 0.146, [-10, 5], [-8, 3]],
    ]


def compute_unit_psth_and_raster(unit_key, trial_keys, align_type='go_cue', bin_size=0.04):
    """
    Align spikes of specified unit and trial-set to specified align_event_type,
    compute psth with specified window and binsize, and generate data for raster plot.
    (for foraging task only)

    @param unit_key: key of a single unit to compute the PSTH for
    @param trial_keys: list of all the trial keys to compute the PSTH over
    @param align_type: psth_foraging.AlignType
    @param bin_size: (in sec)
    
    Returns a dictionary of the form:
      {
         'bins': time bins,
         'trials': ephys.Unit.TrialSpikes.trials,
         'spikes_aligned': aligned spike times per trial
         'psth': (bins x 1)
         'psth_per_trial': (trial x bins)
         'raster': Spike * Trial raster [np.array, np.array]
      }
    """
    
    q_align_type = AlignType & {'align_type_name': align_type}
    
    # -- Get global times for spike and event --
    q_spike = ephys.Unit & unit_key  # Using ephys.Unit, not ephys.Unit.TrialSpikes
    q_event = ephys.TrialEvent & trial_keys & q_align_type   # Using ephys.TrialEvent, not experiment.TrialEvent
    if not q_spike or not q_event:
        return None

    # Session-wise spike times (relative to the first sTrig, i.e. 'bitcodestart'. see line 212 of ingest.ephys)
    spikes = q_spike.fetch1('spike_times')
    
    # Session-wise event times (relative to session start)
    events, trials = q_event.fetch('trial_event_time', 'trial', order_by='trial asc')
    # Make event times also relative to the first sTrig
    events -= (ephys.TrialEvent & trial_keys.proj(_='trial') & {'trial_event_type': 'bitcodestart', 'trial': 1}).fetch1('trial_event_time')
    events = events.astype(float)
    
    # Manual correction of trialstart, if necessary
    events += q_align_type.fetch('time_offset').astype(float)
    
    # -- Align spike times to each event --
    win = q_align_type.fetch1('psth_win')
    spikes_aligned = []
    for e_t in events:
        s_t = spikes[(e_t + win[0] <= spikes) & (spikes < e_t + win[1])]
        spikes_aligned.append(s_t - e_t)
    
    # -- Compute psth --
    binning = np.arange(win[0], win[1], bin_size)
    
    # psth (bins x 1)
    all_spikes = np.concatenate(spikes_aligned)
    psth, edges = np.histogram(all_spikes, bins=binning)
    psth = psth / len(q_event) / bin_size
    
    # psth per trial (trial x bins)
    psth_per_trial = np.vstack(np.histogram(trial_spike, bins=binning)[0] / bin_size for trial_spike in spikes_aligned)

    # raster (all spike time, all trial number)
    raster = [all_spikes,
              np.concatenate([[t] * len(s)
                              for s, t in zip(spikes_aligned, trials)])]

    return dict(bins=binning[1:], trials=trials, spikes_aligned=spikes_aligned,
                psth=psth, psth_per_trial=psth_per_trial, raster=raster)
