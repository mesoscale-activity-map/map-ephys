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

from . import get_schema_name
from .util import _get_units_hemisphere

schema = dj.schema(get_schema_name('psth'))
log = logging.getLogger(__name__)

# NOW:
# - rework Condition to TrialCondition funtion+arguments based schema


def dict_to_hash(input_dict):
    """
    Given a dictionary, returns an md5 hash string of its ordered keys-values.
    """
    hashed = hashlib.md5()
    for k in sorted(input_dict.keys()):
        hashed.update(str(k).encode())
        hashed.update(str(input_dict[k]).encode())
    return hashed.hexdigest()


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
            {
                'trial_condition_name': 'good_noearlylick_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'good_noearlylick_left_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0,
                    'trial_instruction': 'left'}
            },
            {
                'trial_condition_name': 'good_noearlylick_right_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0,
                    'trial_instruction': 'right'}
            },
            {
                'trial_condition_name': 'good_noearlylick_left_miss',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0,
                    'trial_instruction': 'left'}
            },
            {
                'trial_condition_name': 'good_noearlylick_right_miss',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0,
                    'trial_instruction': 'right'}
            },
            {
                'trial_condition_name': 'all_noearlylick_nostim',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0}
            },
            {
                'trial_condition_name': 'all_noearlylick_nostim_left',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0,
                    'trial_instruction': 'left'}
            },
            {
                'trial_condition_name': 'all_noearlylick_nostim_right',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early',
                    'auto_water': 0,
                    'free_water': 0,
                    'trial_instruction': 'right'}
            }
        ]

        # PHOTOSTIM conditions
        stim_locs = [('left', 'alm'), ('right', 'alm'), ('both', 'alm')]
        for hemi, brain_area in stim_locs:
            for instruction in (None, 'left', 'right'):
                condition = {'trial_condition_name': '_'.join(filter(None, ['all', 'noearlylick',
                                                                            '_'.join([hemi, brain_area]), 'stim',
                                                                            instruction])),
                             'trial_condition_func': '_get_trials_include_stim',
                             'trial_condition_arg': {
                                 **{'_outcome': 'ignore',
                                    'task': 'audio delay',
                                    'task_protocol': 1,
                                    'early_lick': 'no early',
                                    'auto_water': 0,
                                    'free_water': 0,
                                    'stim_laterality': hemi,
                                    'stim_brain_area': brain_area},
                                 **({'trial_instruction': instruction} if instruction else {})}
                             }
                contents_data.append(condition)

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
        behav_attrs = set(experiment.BehaviorTrial.heading.names)

        _stim_key = {k: v for k, v in _restr.items() if k in stim_attrs}
        _behav_key = {k: v for k, v in _restr.items() if k in behav_attrs}

        stim_key = {k: v for k, v in restr.items() if k in stim_attrs}
        behav_key = {k: v for k, v in restr.items() if k in behav_attrs}

        return (((experiment.BehaviorTrial & behav_key) - [{k: v} for k, v in _behav_key.items()]) -
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
        behav_attrs = set(experiment.BehaviorTrial.heading.names)

        _stim_key = {k: v for k, v in _restr.items() if k in stim_attrs}
        _behav_key = {k: v for k, v in _restr.items() if k in behav_attrs}

        stim_key = {k: v for k, v in restr.items() if k in stim_attrs}
        behav_key = {k: v for k, v in restr.items() if k in behav_attrs}

        return (((experiment.BehaviorTrial & behav_key) - [{k: v} for k, v in _behav_key.items()]) &
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
        return nostim.proj() + stim.proj()

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
class Selectivity(dj.Lookup):
    """
    Selectivity lookup values
    """

    definition = """
    selectivity: varchar(24)
    """

    contents = zip(['contra-selective', 'ipsi-selective', 'non-selective'])


@schema
class PeriodSelectivity(dj.Computed):
    """
    Multi-trial selectivity for a specific trial subperiod
    """

    definition = """
    -> ephys.Unit
    -> experiment.Period
    ---
    -> Selectivity.proj(period_selectivity='selectivity')
    ipsi_firing_rate:           float  # mean firing rate of all ipsi-trials
    contra_firing_rate:         float  # mean firing rate of all contra-trials
    p_value:                    float  # all trial spike rate t-test p-value
    """

    alpha = 0.05  # default alpha value

    key_source = experiment.Period * (ephys.Unit & ephys.ProbeInsertion.InsertionLocation & 'unit_quality != "all"')

    def make(self, key):
        '''
        Compute Period Selectivity for a given unit.
        '''
        log.debug('PeriodSelectivity.make(): key: {}'.format(key))

        hemi = _get_units_hemisphere(key)

        # retrieving the spikes of interest,
        spikes_q = ((ephys.Unit.TrialSpikes & key)
                    * (experiment.BehaviorTrial
                       & {'task': 'audio delay',
                          'early_lick': 'no early',
                          'outcome': 'hit',
                          'free_water': 0,
                          'auto_water': 0})
                    & (experiment.TrialEvent & 'trial_event_type = "delay"' & 'duration = 1.2')
                    - experiment.PhotostimEvent)

        if not spikes_q:  # no spikes found
            self.insert1({**key, 'period_selectivity': 'non-selective'})
            return

        # retrieving event times
        start_event, start_tshift, end_event, end_tshift = (experiment.Period & key).fetch1(
            'start_event_type', 'start_time_shift', 'end_event_type', 'end_time_shift')
        start_event_q = {k['trial']: float(k['start_event_time'])
                         for k in (experiment.TrialEvent & key & {'trial_event_type': start_event}).proj(
            start_event_time=f'trial_event_time + {start_tshift}').fetch(as_dict=True)}
        end_event_q = {k['trial']: float(k['end_event_time'])
                       for k in (experiment.TrialEvent & key & {'trial_event_type': end_event}).proj(
            end_event_time=f'trial_event_time + {end_tshift}').fetch(as_dict=True)}
        cue_event_q = {k['trial']: float(k['trial_event_time'])
                       for k in (experiment.TrialEvent & key & {'trial_event_type': 'go'}).fetch(as_dict=True)}

        # compute spike rate during the period-of-interest for each trial
        freq_i, freq_c = [], []
        for trial, trial_instruct, spike_times in zip(*spikes_q.fetch('trial', 'trial_instruction', 'spike_times')):
            start_time = start_event_q[trial] - cue_event_q[trial]
            stop_time = end_event_q[trial] - cue_event_q[trial]
            spk_rate = np.logical_and(spike_times >= start_time, spike_times < stop_time).sum() / (stop_time - start_time)
            if hemi == trial_instruct:
                freq_i.append(spk_rate)
            else:
                freq_c.append(spk_rate)

        # and testing for selectivity.
        t_stat, pval = sc_stats.ttest_ind(freq_i, freq_c, equal_var=True)
        # try:
        #     _stat, pval = sc_stats.mannwhitneyu(freq_i, freq_c)
        # except ValueError:
        #     pval = np.nan

        freq_i_m = np.average(freq_i)
        freq_c_m = np.average(freq_c)

        pval = 1 if np.isnan(pval) else pval
        if pval > self.alpha:
            pref = 'non-selective'
        else:
            pref = ('ipsi-selective' if freq_i_m > freq_c_m
                    else 'contra-selective')

        self.insert1({**key, 'p_value': pval,
                      'period_selectivity': pref,
                      'ipsi_firing_rate': freq_i_m,
                      'contra_firing_rate': freq_c_m})


@schema
class UnitSelectivity(dj.Computed):
    """
    Multi-trial selectivity at unit level
    """

    definition = """
    -> ephys.Unit
    ---
    -> Selectivity.proj(unit_selectivity='selectivity')
    """

    # Unit Selectivity is computed only for units
    # that has PeriodSelectivity computed for "sample" and "delay" and "response"
    key_source = (ephys.Unit
                  & (PeriodSelectivity & 'period = "sample"')
                  & (PeriodSelectivity & 'period = "delay"')
                  & (PeriodSelectivity & 'period = "response"'))

    def make(self, key):
        '''
        calculate 'global' selectivity for a unit -
        '''
        log.debug('UnitSelectivity.make(): key: {}'.format(key))

        # fetch region selectivity,
        sels = (PeriodSelectivity & key).fetch('period_selectivity')

        if (sels == 'non-selective').all():
            log.debug('... no UnitSelectivity for unit')
            self.insert1({**key, 'unit_selectivity': 'non-selective'})
            return

        contra_frate, ipsi_frate = (PeriodSelectivity & key & 'period in ("sample", "delay", "response")').fetch(
            'contra_firing_rate', 'ipsi_firing_rate')

        pref = ('ipsi-selective' if ipsi_frate.mean() > contra_frate.mean() else 'contra-selective')

        log.debug('... prefers: {}'.format(pref))

        self.insert1({**key, 'unit_selectivity': pref})


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
    :param time_period: (time_from, time_to) in seconds
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
    :param time_period: (time_from, time_to) in seconds
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


