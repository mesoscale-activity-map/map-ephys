import logging
import operator
import math

from functools import reduce
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
# - [X] rename UnitCondition to TrialCondition
# - [X] store actual Selectivity value
# - remove Condition & refactor
#   - provide canned queries
#   - (old? also null filtering funs?)


class Condition:
    '''
    Manually curated condition queries
    '''

    @staticmethod
    def q(q, spikes=None):
        if spikes is None:
            return q
        else:
            return ephys.TrialSpikes & q

    @staticmethod
    def audio_delay_ipsi_hit_nostim(spikes=None):
        return Condition.q(
            ((experiment.BehaviorTrial()
              & {'task': 'audio delay'}
              & {'trial_instruction': 'left'}
              & {'early_lick': 'no early'}
              & {'outcome': 'hit'}) - experiment.PhotostimEvent), spikes)

    @staticmethod
    def audio_delay_contra_hit_nostim(spikes=None):
        return Condition.q(
            ((experiment.BehaviorTrial()
              & {'task': 'audio delay'}
              & {'trial_instruction': 'right'}
              & {'early_lick': 'no early'}
              & {'outcome': 'hit'}) - experiment.PhotostimEvent), spikes)

    @staticmethod
    def audio_delay_ipsi_miss_nostim(spikes=None):
        return Condition.q(
            ((experiment.BehaviorTrial()
              & {'task': 'audio delay'}
              & {'trial_instruction': 'left'}
              & {'early_lick': 'no early'}
              & {'outcome': 'miss'}) - experiment.PhotostimEvent), spikes)

    @staticmethod
    def audio_delay_contra_miss_nostim(spikes=None):
        return Condition.q(
            ((experiment.BehaviorTrial()
              & {'task': 'audio delay'}
              & {'trial_instruction': 'right'}
              & {'early_lick': 'no early'}
              & {'outcome': 'miss'}) - experiment.PhotostimEvent), spikes)


class UnitPsth:

    psth_params = {'xmin': -3, 'xmax': 3, 'binsize': 0.04}

    @staticmethod
    def compute_unit_trial_psth(unit_key, trial_keys):
        q = (ephys.TrialSpikes() & unit_key & trial_keys)
        spikes = q.fetch('spike_times')

        return UnitPsth.compute_unit_psth_ll(spikes)

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
class UnitSelectivityChris(dj.Computed):
    """
    Compute unit selectivity for a unit in a particular time period.
    Calculation:
    2 tail t significance of unit firing rate for trial type: CorrectLeft vs. CorrectRight (no stim, no early lick)
    frequency = nspikes(period)/len(period)
    """
    definition = """
    -> ephys.Unit
    ---
    sample_selectivity=Null:    float         # sample period selectivity
    delay_selectivity=Null:     float         # delay period selectivity
    go_selectivity=Null:        float         # go period selectivity
    global_selectivity=Null:    float         # global selectivity
    min_selectivity=Null:       float         # (sample|delay|go) selectivity
    sample_preference=Null:     boolean       # sample period pref. (i|c)
    delay_preference=Null:      boolean       # delay period pref. (i|c)
    go_preference=Null:         boolean       # go period pref. (i|c)
    global_preference=Null:     boolean       # global non-period pref. (i|c)
    any_preference=Null:        boolean       # any period pref. (i|c)
    """

    alpha = 0.05  # default alpha value

    @property
    def selective(self):
        return 'min_selectivity<{}'.format(self.alpha)

    ipsi_preferring = 'global_preference=1'
    contra_preferring = 'global_preference=0'

    key_source = ephys.Unit & 'unit_quality = "good"'

    def make(self, key):
        log.debug('Selectivity.make(): key: {}'.format(key))

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

        criteria = {}  # with which to calculate the selctivity criteria.

        ranges = self.ranges
        periods = list(ranges.keys())

        for period in periods:
            bounds = ranges[period]
            name = period + '_selectivity'
            pref = period + '_preference'

            lower_mask = np.ma.masked_greater_equal(square, bounds[0])
            upper_mask = np.ma.masked_less_equal(square, bounds[1])
            inrng_mask = np.logical_and(lower_mask.mask, upper_mask.mask)

            rsum = np.sum(inrng_mask, axis=1)
            dur = bounds[1] - bounds[0]
            freq = rsum / dur

            freq_i = freq[behav_i]
            freq_c = freq[behav_c]
            t_stat, pval = sc_stats.ttest_ind(freq_i, freq_c, equal_var=False)

            criteria[name] = pval

            if period != 'global':
                criteria[pref] = (1 if np.average(freq_i)
                                  > np.average(freq_c) else 0)
            else:
                min_sel = min([v for k, v in criteria.items()
                               if 'selectivity' in k])

                any_pref = any([v for k, v in criteria.items()
                                if 'preference' in k])

                criteria['min_selectivity'] = min_sel
                criteria['any_preference'] = any_pref

                # XXX: hacky.. best would be to have another value
                gbl_pref = (1 if ((np.average(freq_i) > np.average(freq_c))
                                  and (min_sel <= self.alpha)) else 0)

                criteria[pref] = gbl_pref

        self.insert1({**key, **criteria})


@schema
class Selectivity(dj.Lookup):
    definition = """
    selectivity: varchar(24)
    """

    contents = zip(['contra-selective', 'ipsi-selective', 'non-selective'])


@schema
class UnitSelectivity(dj.Computed):
    """
    Compute unit selectivity for a unit in a particular time period.
    Calculation:
    2 tail t significance of unit firing rate for trial type: CorrectLeft vs. CorrectRight (no stim, no early lick)
    frequency = nspikes(period)/len(period)
    """
    definition = """
    -> ephys.Unit
    ---
    -> Selectivity.proj(unit_selectivity='selectivity')
    """

    class PeriodSelectivity(dj.Part):
        definition = """
        -> master
        -> experiment.Period
        ---
        -> Selectivity.proj(period_selectivity='selectivity')
        contra_firing_rate: float  # mean firing rate of all contra-trials
        ipsi_firing_rate: float  # mean firing rate of all ipsi-trials
        p_value: float  # p-value of the t-test of spike-rate of all trials
        """

    key_source = ephys.Unit & 'unit_quality = "good"'

    def make(self, key):
        log.debug('Selectivity.make(): key: {}'.format(key))

        trial_restrictor = {'task': 'audio delay', 'task_protocol': 1,
                            'outcome': 'hit', 'early_lick': 'no early'}
        correct_right = {**trial_restrictor, 'trial_instruction': 'right'}
        correct_left = {**trial_restrictor, 'trial_instruction': 'left'}

        # get trial spike times
        right_trialspikes = (ephys.TrialSpikes * experiment.BehaviorTrial
                             - experiment.PhotostimTrial & key & correct_right).fetch('spike_times', order_by='trial')
        left_trialspikes = (ephys.TrialSpikes * experiment.BehaviorTrial
                            - experiment.PhotostimTrial & key & correct_left).fetch('spike_times', order_by='trial')

        unit_hemi = (ephys.ProbeInsertion.InsertionLocation * experiment.BrainLocation & key).fetch1('hemisphere')

        if unit_hemi not in ('left', 'right'):
            raise Exception('Hemisphere Error! Unit not belonging to either left or right hemisphere')

        contra_trialspikes = right_trialspikes if unit_hemi == 'left' else left_trialspikes
        ipsi_trialspikes = left_trialspikes if unit_hemi == 'left' else right_trialspikes

        period_selectivity = []
        for period in experiment.Period.fetch(as_dict=True):
            period_dur = period['period_end'] - period['period_start']
            contra_trial_spk_rate = [(np.logical_and(t >= period['period_start'],
                                                      t < period['period_end'])).astype(int).sum() / period_dur
                             for t in contra_trialspikes]
            ipsi_trial_spk_rate = [(np.logical_and(t >= period['period_start'],
                                                    t < period['period_end'])).astype(int).sum() / period_dur
                           for t in ipsi_trialspikes]

            contra_frate = np.mean(contra_trial_spk_rate)
            ipsi_frate = np.mean(ipsi_trial_spk_rate)

            # do t-test on the spike-count per trial for all contra trials vs. ipsi trials
            t_stat, pval = sc_stats.ttest_ind(contra_trial_spk_rate, ipsi_trial_spk_rate)

            if pval > 0.05:
                pref = 'non-selective'
            else:
                pref = 'ipsi-selective' if ipsi_frate > contra_frate else 'contra-selective'

            period_selectivity.append(dict(key, **period, period_selectivity=pref, p_value=pval,
                                            contra_firing_rate=contra_frate, ipsi_firing_rate=ipsi_frate))

        unit_selective = not (np.array([p['period_selectivity'] for p in period_selectivity]) == 'non-selective').all()
        if not unit_selective:
            unit_pref = 'non-selective'
        else:
            ave_ipsi_frate = np.array([p['ipsi_firing_rate'] for p in period_selectivity]).mean()
            ave_contra_frate = np.array([p['contra_firing_rate'] for p in period_selectivity]).mean()
            unit_pref = 'ipsi-selective' if ave_ipsi_frate > ave_contra_frate else 'contra-selective'

        self.insert1(dict(**key, unit_selectivity=unit_pref))
        self.PeriodSelectivity.insert(period_selectivity, ignore_extra_fields=True)
