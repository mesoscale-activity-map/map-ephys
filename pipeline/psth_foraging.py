import logging
from functools import partial
from inspect import getmembers
import numpy as np
import datajoint as dj
import pandas as pd
import statsmodels.api as sm

from . import (lab, experiment, ephys)
[lab, experiment, ephys]  # NOQA

from . import get_schema_name, dict_to_hash
from pipeline import foraging_model
from pipeline.util import _get_unit_independent_variable

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
class AlignType(dj.Lookup):
    '''
    Define flexible psth alignment types
    '''
    definition = """
    idx:                 smallint
    -> experiment.TrialEventType
    ---
    align_type_name:     varchar(128)     # user-friendly name of alignment type
    trial_offset=0:      smallint         # e.g., offset = 1 means the psth will be aligned to the event of the *next* trial (offset *align* relative to *condition*)
    time_offset=0:       Decimal(10, 5)   # will be added to the event time for manual correction (e.g., bitcodestart to actual zaberready)  
    psth_win:            tinyblob   
    xlim:                tinyblob
    """
    contents = [
        [0, 'zaberready', 'trial_start', 0, 0, [-3, 2], [-2, 1]],
        [1, 'go', 'go_cue', 0, 0, [-2, 5], [-1, 3]],
        [2, 'choice', 'first_lick_after_go_cue', 0, 0, [-2, 5], [-1, 3]],
        [3, 'trialend', 'iti_start', 0, 0, [-3, 10], [-2, 5]],
        [4, 'zaberready', 'next_trial_start', 1, 0, [-10, 5], [-8, 3]],
        [5, 'zaberready', 'next_two_trial_start', 2, 0, [-10, 5], [-8, 3]],

        # In the first few sessions, zaber moter feedback is not recorded,
        # so we don't know the exact time of trial start ('zaberready').
        # We have to estimate actual trial start by
        #   bitcodestart + bitcode width (42 ms for first few sessions) + zaber movement duration (~ 104 ms, very repeatable)
        [6, 'bitcodestart', 'trial_start_bitcode', 0, 0.146, [-3, 2], [-2, 1]],
        [7, 'bitcodestart', 'next_trial_start_bitcode', 1, 0.146, [-10, 5], [-8, 3]],
        [8, 'bitcodestart', 'next_two_trial_start_bitcode', 2, 0.146, [-10, 5], [-8, 3]],
    ]


@schema
class IndependentVariable(dj.Lookup):
    """
    Define independent variables over trial to generate psth or design matrix of regression
    """
    definition = """
    var_name:  varchar(50)
    ---
    desc:   varchar(200)
    """

    @property
    def contents(self):
        contents = [
            # Model-independent (trial facts)
            ['choice_lr', 'left (0) or right (1)'],
            ['choice_ic', 'ipsi (0) or contra (1)'],
            ['reward', 'miss (0) or hit (1)'],

            # Model-dependent (latent variables)
            ['relative_action_value_lr', 'relative action value (Q_r - Q_l)'],
            ['relative_action_value_ic', 'relative action value (Q_contra - Q_ipsi)'],
            ['total_action_value', 'total action value (Q_r + Q_l)'],
            ['rpe', 'outcome - Q_chosen']
        ]

        latent_vars = foraging_model.FittedSessionModel.TrialLatentVariable.heading.secondary_attributes

        for side in ['left', 'right', 'ipsi', 'contra']:
            for var in (latent_vars):
                contents.append([f'{side}_{var}', f'{side} {var}'])

        return contents


@schema
class LinearModel(dj.Lookup):
    """
    Define multivariate linear models for PeriodSelectivity fitting
    """
    definition = """
    multi_linear_model: varchar(30)
    ---
    if_intercept: tinyint   # Whether intercept is included
    """

    class X(dj.Part):
        definition = """
        -> master
        -> IndependentVariable
        """

    @classmethod
    def load(cls):
        contents = [
            ['Q_l + Q_r + rpe', 1, ['left_action_value', 'right_action_value', 'rpe']],
        ]

        for m in contents:
            cls.insert1(m[:2], skip_duplicates=True)
            for iv in m[2]:
                cls.X.insert1([m[0], iv], skip_duplicates=True)


@schema
class LinearModelPeriodToFit(dj.Lookup):
    """
    Subset of experiment.PeriodForaging (or adding sliding window here in the future?)
    """
    definition = """
    -> experiment.PeriodForaging
    """

    contents = [
        ('before_2', ), ('delay', ), ('go_to_end', ), ('go_1.2', ),
        ('iti_all', ), ('iti_first_2', ), ('iti_last_2', )
    ]


@schema
class LinearModelBehaviorModelToFit(dj.Lookup):
    definition = """
    behavior_model:  varchar(30)    # If 'best_aic' etc, the best behavioral model for each session; otherwise --> Model.model_id
    """
    contents = [['best_aic',]]


@schema
class UnitPeriodLinearFit(dj.Computed):
    definition = """
    -> ephys.Unit
    -> LinearModelPeriodToFit
    -> LinearModelBehaviorModelToFit
    -> LinearModel
    ---
    actual_behavior_model:   int
    model_r2=Null:  float   # r square
    model_r2_adj=Null:   float  # r square adj.
    model_p=Null:   float
    """

    class Param(dj.Part):
        definition = """
        -> master
        -> LinearModel.X
        ---
        beta=Null: float
        std_err=Null: float
        p=Null:    float
        t=Null:    float  #  t statistic

        """
        
    def make(self, key):
        # -- Fetech data --
        period, behavior_model = key['period'], key['behavior_model']

        # Parse period
        if period in ['delay'] and not ephys.TrialEvent & key & 'trial_event_type = "zaberready"':
            period = period + '_bitcode'  # Manually correction of bitcodestart to zaberready, if necessary

        # Parse behavioral model_id
        if behavior_model.isnumeric():
            model_id = int(behavior_model)
        else:
            model_id = (foraging_model.FittedSessionModelComparison.BestModel &
                        key & 'model_comparison_idx=0').fetch1(behavior_model)

        # Parse independent variable
        independent_variables = (LinearModel.X & key).fetch('var_name')
        if_intercept = (LinearModel & key).fetch1('if_intercept')

        # Get data
        period_activity = compute_unit_period_activity(key, period)
        all_iv = _get_unit_independent_variable(key, model_id=model_id)

        # TODO Align ephys event with behavior using bitcode! (and save raw bitcodes)
        trial = all_iv.trial  # Without ignored trials
        trial_with_ephys = trial <= max(period_activity['trial'])
        trial = trial[trial_with_ephys]  # Truncate behavior trial to max ephys length (this assumes the first trial is aligned, see ingest.ephys)
        all_iv = all_iv[trial_with_ephys]  # Also truncate all ivs
        firing = period_activity['firing_rates'][trial - 1]  # Align ephys trial and model trial (e.g., no ignored trials in model fitting)

        # -- Fit --
        y = pd.DataFrame({f'{period} firing': firing})
        x = all_iv[independent_variables].astype(float)
        model = sm.OLS(y, sm.add_constant(x) if if_intercept else x)
        model_fit = model.fit()

        # -- Insert --
        self.insert1({**key,
                      'model_r2': model_fit.rsquared,
                      'model_r2_adj': model_fit.rsquared_adj,
                      'model_p': model_fit.f_pvalue,
                      'actual_behavior_model': model_id})

        for para in [p for p in model.exog_names if p!='const']:
            self.Param.insert1({**key,
                                'var_name': para,
                                'beta': model_fit.params[para],
                                'std_err': model_fit.bse[para],
                                'p': model_fit.pvalues[para],
                                't': model_fit.tvalues[para]
                                })


# ============= Helpers =============

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


def compute_unit_period_activity(unit_key, period):
    """
    Given unit and period, compute average firing rate over trials
    I tried to put this in a table, but it's too slow... (too many elements)
    @param unit_key:
    @param period: -> experiment.PeriodForaging, or arbitrary list in the same format
    @return: DataFrame(trial, spike_count, duration, firing_rate)
    """

    q_spike = ephys.Unit & unit_key
    q_event = ephys.TrialEvent & unit_key
    if not q_spike or not q_event:
        return None

    # for (the very few) sessions without zaber feedback signal, use 'bitcodestart' with manual correction
    if period == 'delay' and \
            not q_event & 'trial_event_type = "zaberready"':
        period = 'delay_bitcode'

    # -- Fetch global session times of given period, for each trial --
    try:
        (start_event_type, start_trial_shift, start_time_shift,
         end_event_type, end_trial_shift, end_time_shift) = (experiment.PeriodForaging & {'period': period}
                  ).fetch1('start_event_type', 'start_trial_shift', 'start_time_shift',
                           'end_event_type', 'end_trial_shift', 'end_time_shift')
    except:
        (start_event_type, start_trial_shift, start_time_shift,
         end_event_type, end_trial_shift, end_time_shift) = period
    
    start = {k['trial']: float(k['start_event_time'])
             for k in (q_event & {'trial_event_type': start_event_type}).proj(
            start_event_time=f'trial_event_time + {start_time_shift}').fetch(as_dict=True)}
    end = {k['trial']: float(k['end_event_time'])
             for k in (q_event & {'trial_event_type': end_event_type}).proj(
            end_event_time=f'trial_event_time + {end_time_shift}').fetch(as_dict=True)}

    # Handle edge effects due to trial shift
    trials = np.array(list(start.keys()))
    actual_trials = trials[(trials <= max(trials) - end_trial_shift) &
                           (trials >= min(trials) - start_trial_shift)]

    # -- Fetch and count spikes --
    spikes = q_spike.fetch1('spike_times')
    spike_counts, durations = [], []

    for trial in actual_trials:
        t_s = start[trial + start_trial_shift]
        t_e = end[trial + end_trial_shift]

        spike_counts.append(((t_s <= spikes) & (spikes < t_e)).sum())  # Much faster than sum(... & ...) (python sum on np array)!!
        durations.append(t_e - t_s)

    return {'trial': actual_trials, 'spike_counts': np.array(spike_counts),
            'durations': np.array(durations), 'firing_rates': np.array(spike_counts) / np.array(durations)}