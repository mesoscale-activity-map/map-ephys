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
    trial_condition_id:         varchar(32)     # hash of trial_condition_arg
    ---
    trial_condition_desc:       varchar(1000)   # trial condition description
    trial_condition_func:       varchar(36)     # trial retrieval function
    trial_condition_arg:        longblob        # trial retrieval arguments
    """

    @property
    def contents(self):
        contents_data = (
            {
                'trial_condition_desc': 'good_noearlylick_hit',
                'trial_condition_func': '_get_trials_no_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'hit',
                    'early_lick': 'no early'}
            },
            {
                'trial_condition_desc': 'good_noearlylick_left_hit',
                'trial_condition_func': '_get_trials_no_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'trial_instruction': 'left'}
            },
            {
                'trial_condition_desc': 'good_noearlylick_right_hit',
                'trial_condition_func': '_get_trials_no_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'hit',
                    'early_lick': 'no early',
                    'trial_instruction': 'right'}
            },
            {
                'trial_condition_desc': 'good_noearlylick_left_miss',
                'trial_condition_func': '_get_trials_no_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'trial_instruction': 'left'}
            },
            {
                'trial_condition_desc': 'good_noearlylick_right_miss',
                'trial_condition_func': '_get_trials_no_stim',
                'trial_condition_arg': {
                    'task': 'audio delay',
                    'task_protocol': 1,
                    'outcome': 'miss',
                    'early_lick': 'no early',
                    'trial_instruction': 'right'}
            },
        )

        return ({**d, 'trial_condition_id': key_hash(d['trial_condition_arg'])}
                for d in contents_data)

    @classmethod
    def get_trials(cls,  trial_condition_desc):
        return cls.get_func({'trial_condition_desc': trial_condition_desc})()

    @classmethod
    def get_func(cls, key):
        self = cls()

        func, args = (self & key).fetch1(
            'trial_condition_func', 'trial_condition_arg')

        return partial(dict(getmembers(cls))[func], **args)

    @classmethod
    def _get_trials_no_stim(cls, task=None, task_protocol=None, outcome=None,
                            early_lick=None, trial_instruction=None):

        log.debug('_get_trials_no_stim', locals())

        return ((experiment.BehaviorTrial
                 & {'task': task}
                 & {'trial_instruction': trial_instruction}
                 & {'early_lick': early_lick}
                 & {'outcome': outcome}) - experiment.PhotostimEvent)

    @classmethod
    def _get_trials_stim(cls, task=None, task_protocol=None, outcome=None,
                         early_lick=None, trial_instruction=None):

        log.debug('_get_trials_stim', locals())

        return ((experiment.BehaviorTrial
                 & {'task': task}
                 & {'trial_instruction': trial_instruction}
                 & {'early_lick': early_lick}
                 & {'outcome': outcome}) & experiment.PhotostimEvent)


@schema
class UnitPsth(dj.Computed):
    definition = """
    -> TrialCondition
    -> ephys.Unit
    ---
    unit_psth=NULL:                             longblob
    """
    psth_params = {'xmin': -3, 'xmax': 3, 'binsize': 0.04}

    def make(self, key):
        log.info('UnitPsth.make(): key: {}'.format(key))

        unit = {k: v for k, v in key.items() if k in ephys.Unit.primary_key}

        # Expand Condition -
        # could conditionalize e.g.
        # if key['condition_id'] in [1,2,3]: self.make_thiskind(key), etc.
        # for now, we assume one method of processing.

        cond = TrialCondition.expand(key['condition_id'])

        all_trials = TrialCondition.trials({
            'TaskProtocol': cond['TaskProtocol'],
            'TrialInstruction': cond['TrialInstruction'],
            'EarlyLick': cond['EarlyLick'],
            'Outcome': cond['Outcome']})

        photo_trials = TrialCondition.trials({
            'PhotostimLocation': cond['PhotostimLocation']})

        # HACK special case stim condition logic -
        #   ... should be fixed by expanding Condition support logic.
        if 'onlystim' in cond['Condition']['condition_desc']:
            tgt_trials = [t for t in all_trials if t in photo_trials]
        elif 'nostim' in cond['Condition']['condition_desc']:
            tgt_trials = [t for t in all_trials if t not in photo_trials]
        else:
            tgt_trials = all_trials

        q = (ephys.TrialSpikes() & unit & tgt_trials)
        spikes = q.fetch('spike_times')

        if len(spikes) == 0:
            log.warning('no spikes found for key {} - null psth'.format(key))
            self.insert1(key)
            return

        spikes = np.concatenate(spikes)

        xmin, xmax, bins = self.psth_params.values()
        # XXX: xmin, xmax+bins (149 here vs 150 in matlab)..
        #   See also [:1] slice in plots..
        psth = list(np.histogram(spikes, bins=np.arange(xmin, xmax, bins)))
        psth[0] = psth[0] / len(tgt_trials) / bins

        self.insert1({**key, 'unit_psth': np.array(psth)})

    @classmethod
    def get(cls, condition_key, unit_key,
            incl_conds=['TaskProtocol', 'TrialInstruction', 'EarlyLick',
                        'Outcome'],
            excl_conds=['PhotostimLocation']):
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

        condition = TrialCondition.expand(condition_key['condition_id'])
        session_key = {k: unit_key[k] for k in experiment.Session.primary_key}

        psth_q = (UnitPsth & {**condition_key, **unit_key})
        psth = psth_q.fetch1()['unit_psth']

        i_trials = TrialCondition.trials({k: condition[k] for k in incl_conds},
                                         session_key)

        x_trials = TrialCondition.trials({k: condition[k] for k in excl_conds},
                                         session_key)

        st_q = ((ephys.TrialSpikes & i_trials & unit_key) -
                (experiment.SessionTrial & x_trials & unit_key))

        spikes, trials = st_q.fetch('spike_times', 'trial',
                                    order_by='trial asc')

        raster = [np.concatenate(spikes),
                  np.concatenate([[t] * len(s)
                                  for s, t in zip(spikes, trials)])]

        return dict(trials=trials, spikes=spikes, psth=psth, raster=raster)


class UnitPsthOld:

    psth_params = {'xmin': -3, 'xmax': 3, 'binsize': 0.04}

    @staticmethod
    def compute_unit_trial_psth(unit_key, trial_keys):

        q = (ephys.TrialSpikes() & unit_key & trial_keys)
        spikes = q.fetch('spike_times')
        return UnitPsth.compute_psth(spikes)

    @staticmethod
    def compute_psth(session_unit_spikes):
        spikes = np.concatenate(session_unit_spikes)

        xmin, xmax, bins = UnitPsth.psth_params.values()
        psth = list(np.histogram(spikes, bins=np.arange(xmin, xmax, bins)))
        psth[0] = psth[0] / len(session_unit_spikes) / bins

        return np.array(psth)

    @staticmethod
    def get_plotting_data(unit_key, trial_query):
        """
        Retrieve / build data needed for a Unit PSTH Plot based on the given
        unit / condition and included / excluded condition (sub-)variables.

        Returns a dictionary of the form:

          {
             'trials': ephys.TrialSpikes.trials,
             'spikes': ephys.TrialSpikes.spikes,
             'psth': UnitPsth.unit_psth,
             'raster': Spike * Trial raster [np.array, np.array]
          }

        """
        trials, spikes = (ephys.TrialSpikes & trial_query & unit_key).fetch(
            'trial', 'spike_times', order_by='trial asc')

        psth = UnitPsth.compute_psth(spikes)

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

    key_source = experiment.Period * (ephys.Unit & 'unit_quality = "good"')

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
        behav_lr = {k: np.where(behav['trial_instruction'] == k) for k in lr}

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

        # with which to calculate the selctivity over the given period
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
        t_stat, pval = sc_stats.ttest_ind(freq_i, freq_c, equal_var=False)

        freq_i_m = np.average(freq_i)
        freq_c_m = np.average(freq_c)

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

    key_source = ephys.Unit & PeriodSelectivity

    def make(self, key):
        '''
        calculate 'global' selectivity for a unit -
        '''
        log.debug('UnitSelectivity.make(): key: {}'.format(key))

        # verify insertion location is present,
        egpos = None
        try:
            egpos = (ephys.ProbeInsertion.InsertionLocation
                     * experiment.BrainLocation & key).fetch1()
        except dj.DataJointError as e:
            if 'exactly one tuple' in repr(e):
                log.error('... Insertion Location missing. skipping')
                return

        # fetch region selectivity,
        sel = (PeriodSelectivity & key
               & "p_value <= '{}'".format(PeriodSelectivity.alpha)).fetch(
                   as_dict=True)

        if not sel:
            log.debug('... no UnitSelectivity for unit')
            return

        # left/right spikes,
        spikes_l = ((ephys.TrialSpikes & key)
                    & (experiment.BehaviorTrial()
                       & {'task': 'audio delay'}
                       & {'early_lick': 'no early'}
                       & {'outcome': 'hit'}
                       & {'trial_instruction': 'left'})
                    - experiment.PhotostimEvent).fetch('spike_times')

        spikes_r = ((ephys.TrialSpikes & key)
                    & (experiment.BehaviorTrial()
                       & {'task': 'audio delay'}
                       & {'early_lick': 'no early'}
                       & {'outcome': 'hit'}
                       & {'trial_instruction': 'right'})
                    - experiment.PhotostimEvent).fetch('spike_times')

        # compute their average firing rate,
        dur = experiment.Period.trial_duration
        freq_l = np.sum(np.concatenate(spikes_l)) / (len(spikes_l) * dur)
        freq_r = np.sum(np.concatenate(spikes_r)) / (len(spikes_r) * dur)

        # and determine their ipsi/contra preference via frequency.
        if egpos['hemisphere'] == 'left':
            freq_i = freq_l
            freq_c = freq_r
        else:
            freq_i = freq_l
            freq_c = freq_r

        pref = ('ipsi-selective' if freq_i > freq_c else 'contra-selective')

        log.debug('... prefers: {}'.format(pref))

        self.insert1({**key, 'unit_selectivity': pref})
