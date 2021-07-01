import logging
import hashlib

from functools import partial
from inspect import getmembers
from itertools import repeat
import numpy as np
import datajoint as dj
import scipy.stats as sc_stats

from . import (lab, experiment, ephys)
[lab, experiment, ephys]  # NOQA

from . import get_schema_name, dict_to_hash
from .util import _get_units_hemisphere

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
                'trial_condition_name': 'foraging_L_hit_noearlylick',
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
                'trial_condition_name': 'foraging_L_miss_noearlylick',
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
                'trial_condition_name': 'foraging_R_hit_noearlylick',
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
                'trial_condition_name': 'foraging_R_miss_noearlylick',
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
            },
            {
                'trial_condition_name': 'foraging_LR_hit_noearlylick',
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
                'trial_condition_name': 'foraging_LR_miss_noearlylick',
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
                'trial_condition_name': 'foraging_LR_all_noearlylick',
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
                'trial_condition_name': 'foraging_L_all_noearlylick',
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
                'trial_condition_name': 'foraging_R_all_noearlylick',
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
    def get_trials(cls, trial_condition_name):
        return cls.get_func({'trial_condition_name': trial_condition_name})()

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
    def get_func(cls, key):
        self = cls()

        func, args = (self & key).fetch1(
            'trial_condition_func', 'trial_condition_arg')

        return partial(dict(getmembers(cls))[func], **args)

    @classmethod
    def _get_trials_exclude_stim(cls, **kwargs):
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

        return (((experiment.BehaviorTrial * experiment.WaterPortChoice & behav_key) - [{k: v} for k, v in _behav_key.items()]) -
                ((experiment.PhotostimEvent * experiment.PhotostimBrainRegion * experiment.Photostim & stim_key)
                 - [{k: v} for k, v in _stim_key.items()]).proj())

    @classmethod
    def _get_trials_include_stim(cls, **kwargs):
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

        return (((experiment.BehaviorTrial * experiment.WaterPortChoice & behav_key) - [{k: v} for k, v in _behav_key.items()]) &
                ((experiment.PhotostimEvent * experiment.PhotostimBrainRegion * experiment.Photostim & stim_key)
                 - [{k: v} for k, v in _stim_key.items()]).proj())


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


def compute_unit_psth(unit_key, trial_keys, per_trial=False):
    """
    Compute unit-level psth for the specified unit and trial-set - return (time,)
    If per_trial == True, compute trial-level psth - return ((trial x time), time_vec)
    :param unit_key: key of a single unit to compute the PSTH for
    :param trial_keys: list of all the trial keys to compute the PSTH over
    """
    q = (ephys.Unit.TrialSpikes & unit_key & trial_keys)
    if not q:
        return None

    xmin, xmax, bin_size = UnitPsth.psth_params.values()
    binning = np.arange(xmin, xmax, bin_size)

    spikes = q.fetch('spike_times')

    if per_trial:
        trial_psth = np.vstack(np.histogram(spike, bins=binning)[0] / bin_size for spike in spikes)
        return trial_psth, binning[1:]
    else:
        spikes = np.concatenate(spikes)
        psth, edges = np.histogram(spikes, bins=binning)
        psth = psth / len(q) / bin_size
        return psth, edges[1:]


def compute_coding_direction(contra_psths, ipsi_psths, time_period=None):
    """
    Coding direction here is a vector of length: len(unit_keys)
    This coding direction vector (vcd) is the normalized difference between contra-trials firing rate
    and ipsi-trials firing rate per unit, within the specified time period
    :param contra_psths: unit# x (trial-ave psth, psth_edge)
    :param ipsi_psths: unit# x (trial-ave psth, psth_edge)
    :param time_period: (time_from, time_to) in seconds, relative to go-cue
    """
    if not time_period:
        contra_tmin, contra_tmax = zip(*((k[1].min(), k[1].max()) for k in contra_psths))
        ipsi_tmin, ipsi_tmax = zip(*((k[1].min(), k[1].max()) for k in ipsi_psths))
        time_period = max(min(contra_tmin), min(ipsi_tmin)), min(max(contra_tmax), max(ipsi_tmax))

    p_start, p_end = time_period

    contra_ave_spk_rate = np.array([spk_rate[np.logical_and(spk_edge >= p_start, spk_edge < p_end)].mean()
                                    for spk_rate, spk_edge in contra_psths])
    ipsi_ave_spk_rate = np.array([spk_rate[np.logical_and(spk_edge >= p_start, spk_edge < p_end)].mean()
                                  for spk_rate, spk_edge in ipsi_psths])

    cd_vec = contra_ave_spk_rate - ipsi_ave_spk_rate
    return cd_vec / np.linalg.norm(cd_vec)


def compute_CD_projected_psth(units, time_period=None):
    """
    Routine for Coding Direction computation on all the units in the specified unit_keys
    Coding Direction is calculated in the specified time_period
    Unit PSTH are computed over no early-lick, correct-response trials
    :param: unit_keys - list of unit_keys
    :param time_period: (time_from, time_to) in seconds, relative to go-cue
    :return: coding direction unit-vector,
             contra-trials CD projected trial-psth,
             ipsi-trials CD projected trial-psth
             psth time-stamps
    """
    unit_hemi = _get_units_hemisphere(units)
    session_key = experiment.Session & units
    if len(session_key) != 1:
        raise Exception('Units from multiple sessions found')

    # -- the computation part
    # get units and trials - ensuring they have trial-spikes
    contra_trials = (TrialCondition().get_trials(
        'good_noearlylick_right_hit' if unit_hemi == 'left' else 'good_noearlylick_left_hit')
                     & session_key & ephys.Unit.TrialSpikes).fetch('KEY')
    ipsi_trials = (TrialCondition().get_trials(
        'good_noearlylick_left_hit' if unit_hemi == 'left' else 'good_noearlylick_right_hit')
                     & session_key & ephys.Unit.TrialSpikes).fetch('KEY')

    # get per-trial unit psth for all units - unit# x (trial# x time)
    contra_trial_psths, contra_edges = zip(*(compute_unit_psth(unit, contra_trials, per_trial=True)
                                             for unit in units))
    ipsi_trial_psths, ipsi_edges = zip(*(compute_unit_psth(unit, ipsi_trials, per_trial=True)
                                         for unit in units))

    # compute trial-ave unit psth
    contra_psths = zip((p.mean(axis=0) for p in contra_trial_psths), contra_edges)
    ipsi_psths = zip((p.mean(axis=0) for p in ipsi_trial_psths), ipsi_edges)

    # compute coding direction
    cd_vec = compute_coding_direction(contra_psths, ipsi_psths, time_period=time_period)

    # get time vector, relying on all units PSTH shares the same time vector
    time_stamps = contra_edges[0]

    # get coding projection per trial - trial# x unit# x time
    contra_psth_per_trial = np.dstack(contra_trial_psths)
    ipsi_psth_per_trial = np.dstack(ipsi_trial_psths)

    proj_contra_trial = np.vstack(np.dot(tr_u, cd_vec) for tr_u in contra_psth_per_trial)  # trial# x time
    proj_ipsi_trial = np.vstack(np.dot(tr_u, cd_vec) for tr_u in ipsi_psth_per_trial)    # trial# x time

    return cd_vec, proj_contra_trial, proj_ipsi_trial, time_stamps, unit_hemi

