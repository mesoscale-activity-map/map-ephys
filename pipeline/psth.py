import logging
import math
import hashlib

from functools import partial
from inspect import getmembers
from itertools import repeat

import numpy as np
import datajoint as dj

import scipy.stats as sc_stats

from . import lab
from . import experiment
from . import ephys
[lab, experiment, ephys]  # NOQA

from . import get_schema_name

schema = dj.schema(get_schema_name('psth'))
log = logging.getLogger(__name__)

# NOW:
# - rework Condition to TrialCondition funtion+arguments based schema


def key_hash(key):
    """
    Given a dictionary `key`, returns an md5 hash string of its values.

    For use in building dictionary-keyed tables.
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(v).encode())
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
        contents_data = (
            {
                'trial_condition_name': 'good_noearlylick_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'hit',
                    'early_lick': 'no early'}
            },
            {
                'trial_condition_name': 'good_noearlylick_left_hit',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'hit',
                    'early_lick': 'no early',
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
                    'trial_instruction': 'right'}
            },
            {
                'trial_condition_name': 'all_noearlylick_both_alm_stim',
                'trial_condition_func': '_get_trials_include_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early',
                    'brain_location_name': 'both_alm'}
            },
            {
                'trial_condition_name': 'all_noearlylick_both_alm_nostim',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early'}
            },
            {
                'trial_condition_name': 'all_noearlylick_both_alm_stim_left',
                'trial_condition_func': '_get_trials_include_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early',
                    'trial_instruction': 'left',
                    'brain_location_name': 'both_alm'}
            },
            {
                'trial_condition_name': 'all_noearlylick_both_alm_nostim_left',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early',
                    'trial_instruction': 'left'}
            },
            {
                'trial_condition_name': 'all_noearlylick_both_alm_stim_right',
                'trial_condition_func': '_get_trials_include_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early',
                    'trial_instruction': 'right',
                    'brain_location_name': 'both_alm'}
            },
            {
                'trial_condition_name': 'all_noearlylick_both_alm_nostim_right',
                'trial_condition_func': '_get_trials_exclude_stim',
                'trial_condition_arg': {
                    '_outcome': 'ignore',
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'early_lick': 'no early',
                    'trial_instruction': 'right'}
            },
        )
        # generate key XXX: complicated why not just key from description?
        return ({**d, 'trial_condition_hash':
                 key_hash({'trial_condition_func': d['trial_condition_func'],
                           **d['trial_condition_arg']})}
                for d in contents_data)

    @classmethod
    def get_trials(cls, trial_condition_name):
        return cls.get_func({'trial_condition_name': trial_condition_name})()

    @classmethod
    def get_func(cls, key):
        self = cls()

        func, args = (self & key).fetch1(
            'trial_condition_func', 'trial_condition_arg')

        return partial(dict(getmembers(cls))[func], **args)

    @classmethod
    def _get_trials_exclude_stim(cls, **kwargs):

        log.debug('_get_trials_exclude_stim: {}'.format(kwargs))

        restr, _restr = {}, {}
        for k, v in kwargs.items():
            if k.startswith('_'):
                _restr[k[1:]] = v
            else:
                restr[k] = v

        stim_attrs = set(experiment.Photostim.heading.names) - set(experiment.Session.heading.names)
        behav_attrs = set(experiment.BehaviorTrial.heading.names)

        _stim_key = {k: v for k, v in _restr.items() if k in stim_attrs}
        _behav_key = {k: v for k, v in _restr.items() if k in behav_attrs}

        stim_key = {k: v for k, v in restr.items() if k in stim_attrs}
        behav_key = {k: v for k, v in restr.items() if k in behav_attrs}

        return (((experiment.BehaviorTrial & behav_key) - (_behav_key if _behav_key else [])) -
                (experiment.PhotostimEvent * (experiment.Photostim & stim_key) - (_stim_key if _stim_key else [])).proj())

    @classmethod
    def _get_trials_include_stim(cls, **kwargs):

        log.debug('_get_trials_include_stim: {}'.format(kwargs))

        restr, _restr = {}, {}
        for k, v in kwargs.items():
            if k.startswith('_'):
                _restr[k[1:]] = v
            else:
                restr[k] = v

        stim_attrs = set(experiment.Photostim.heading.names) - set(experiment.Session.heading.names)
        behav_attrs = set(experiment.BehaviorTrial.heading.names)

        _stim_key = {k: v for k, v in _restr.items() if k in stim_attrs}
        _behav_key = {k: v for k, v in _restr.items() if k in behav_attrs}

        stim_key = {k: v for k, v in restr.items() if k in stim_attrs}
        behav_key = {k: v for k, v in restr.items() if k in behav_attrs}

        return (((experiment.BehaviorTrial & behav_key) - (_behav_key if _behav_key else [])) &
                (experiment.PhotostimEvent * (experiment.Photostim & stim_key) - (_stim_key if _stim_key else [])).proj())


@schema
class UnitPsth(dj.Computed):
    definition = """
    -> TrialCondition
    -> ephys.Unit
    ---
    unit_psth=NULL: longblob
    """
    psth_params = {'xmin': -3, 'xmax': 3, 'binsize': 0.04}

    def make(self, key):
        log.info('UnitPsth.make(): key: {}'.format(key))

        # expand TrialCondition to trials,
        trials = TrialCondition.get_trials(key['trial_condition_name'])

        # fetch related spike times
        q = (ephys.TrialSpikes & key & trials.proj())
        spikes = q.fetch('spike_times')

        if len(spikes) == 0:
            log.warning('no spikes found for key {} - null psth'.format(key))
            self.insert1(key)
            return

        # compute psth & store.
        # XXX: xmin, xmax+bins (149 here vs 150 in matlab)..
        #   See also [:1] slice in plots..
        unit_psth = self.compute_psth(spikes)

        self.insert1({**key, 'unit_psth': unit_psth})

    @staticmethod
    def compute_psth(session_unit_spikes):
        spikes = np.concatenate(session_unit_spikes)

        xmin, xmax, bins = UnitPsth.psth_params.values()
        psth = list(np.histogram(spikes, bins=np.arange(xmin, xmax, bins)))
        psth[0] = psth[0] / len(session_unit_spikes) / bins

        return np.array(psth)

    @classmethod
    def get_plotting_data(cls, unit_key, condition_key):
        """
        Retrieve / build data needed for a Unit PSTH Plot based on the given
        unit condition and included / excluded condition (sub-)variables.
        Returns a dictionary of the form:
          {
             'trials': ephys.TrialSpikes.trials,
             'spikes': ephys.TrialSpikes.spikes,
             'psth': UnitPsth.unit_psth,
             'raster': Spike * Trial raster [np.array, np.array]
          }
        """
        # from sys import exit as sys_exit  # NOQA
        # from code import interact
        # from collections import ChainMap
        # interact('unitpsth make', local=dict(ChainMap(locals(), globals())))

        trials = TrialCondition.get_func(condition_key)()

        psth = (UnitPsth & {**condition_key, **unit_key}).fetch1()['unit_psth']

        spikes, trials = (ephys.TrialSpikes & trials & unit_key).fetch(
            'spike_times', 'trial', order_by='trial asc')

        raster = [np.concatenate(spikes),
                  np.concatenate([[t] * len(s)
                                  for s, t in zip(spikes, trials)])]

        return dict(trials=trials, spikes=spikes, psth=psth, raster=raster)


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

    key_source = experiment.Period * (ephys.Unit & 'unit_quality != "all"')

    def make(self, key):
        '''
        Compute Period Selectivity for a given unit.
        '''
        log.debug('PeriodSelectivity.make(): key: {}'.format(key))

        # Verify insertion location is present,
        egpos = None
        try:
            egpos = (ephys.ProbeInsertion.InsertionLocation
                     * experiment.BrainLocation & key).fetch1()
        except dj.DataJointError as e:
            if 'exactly one tuple' in repr(e):
                log.error('... Insertion Location missing. skipping')
                return

        # retrieving the spikes of interest,
        spikes_q = ((ephys.TrialSpikes & key)
                    & (experiment.BehaviorTrial()
                       & {'task': 'audio delay'}
                       & {'early_lick': 'no early'}
                       & {'outcome': 'hit'}) - experiment.PhotostimEvent)

        # and their corresponding behavior,
        lr = ['left', 'right']
        behav = (experiment.BehaviorTrial & spikes_q.proj()).fetch(
            order_by='trial asc')
        behav_lr = {k: np.where(behav['trial_instruction'] == k)[0] for k in lr}

        if egpos['hemisphere'] == 'left':
            behav_i = behav_lr['left']
            behav_c = behav_lr['right']
        else:
            behav_i = behav_lr['right']
            behav_c = behav_lr['left']

        # constructing a square, nan-padded trial x spike array
        spikes = spikes_q.fetch(order_by='trial asc')
        ydim = max(len(i['spike_times']) for i in spikes)
        square = np.array(
            np.array([np.concatenate([st, pad])[:ydim]
                      for st, pad in zip(spikes['spike_times'],
                                         repeat([math.nan]*ydim))]))

        # with which to calculate the selectivity over the given period
        period = (experiment.Period & key).fetch1()

        # by determining the period boundaries,
        bounds = (period['period_start'], period['period_end'])

        # masking the appropriate spikes,
        lower_mask = np.ma.masked_greater_equal(square, bounds[0])
        upper_mask = np.ma.masked_less_equal(square, bounds[1])
        inrng_mask = np.logical_and(lower_mask.mask, upper_mask.mask)

        # computing their spike rate,
        rsum = np.sum(inrng_mask, axis=1)
        dur = bounds[1] - bounds[0]
        freq = rsum / dur

        # and testing for selectivity.
        freq_i = freq[behav_i]
        freq_c = freq[behav_c]
        t_stat, pval = sc_stats.ttest_ind(freq_i, freq_c, equal_var=True)

        freq_i_m = np.average(freq_i)
        freq_c_m = np.average(freq_c)


        pval = 1 if np.isnan(pval) else pval
        if pval > self.alpha:
            pref = 'non-selective'
        else:
            pref = ('ipsi-selective' if freq_i_m > freq_c_m
                    else 'contra-selective')

        self.insert1({
            **key,
            'period_selectivity': pref,
            'ipsi_firing_rate': freq_i_m,
            'contra_firing_rate': freq_c_m,
            'p_value': pval
        })


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
    If per_trial == True, compute trial-level psth - return (trial#, time)
    """
    q = (ephys.TrialSpikes & unit_key & trial_keys)
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
    :param: unit_keys - list of unit_keys
    :return: coding direction unit-vector,
             contra-trials CD projected trial-psth,
             ipsi-trials CD projected trial-psth
             psth time-stamps
    """
    unit_hemi = (ephys.ProbeInsertion.InsertionLocation * experiment.BrainLocation
                 & units).fetch('hemisphere')
    if len(unit_hemi) != 1:
        raise Exception('Units from both hemispheres found')
    else:
        unit_hemi = unit_hemi[0]

    session_key = experiment.Session & units
    if len(session_key) != 1:
        raise Exception('Units from multiple sessions found')

    # -- the computation part
    # get units and trials - ensuring they have trial-spikes
    contra_trials = (TrialCondition().get_trials(
        'good_noearlylick_right_hit' if unit_hemi == 'left' else 'good_noearlylick_left_hit')
                     & session_key & ephys.TrialSpikes).fetch('KEY')
    ipsi_trials = (TrialCondition().get_trials(
        'good_noearlylick_left_hit' if unit_hemi == 'left' else 'good_noearlylick_right_hit')
                     & session_key & ephys.TrialSpikes).fetch('KEY')

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

    return cd_vec, proj_contra_trial, proj_ipsi_trial, time_stamps


