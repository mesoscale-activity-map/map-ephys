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
# - [ ] GroupCondition -> now notion of UnitCondition
#   - table notneeded
#   - provide canned queries
#   - also null filtering funs


@schema
class TrialCondition(dj.Manual):
    definition = """
    # manually curated conditions of interest
    condition_id:                               int
    ---
    condition_desc:                             varchar(4096)
    """

    class TaskProtocol(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.TaskProtocol
        """

    class TrialInstruction(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.TrialInstruction
        """

    class EarlyLick(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.EarlyLick
        """

    class Outcome(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.Outcome
        """

    class PhotostimLocation(dj.Part):
        definition = """
        -> master
        -> experiment.Photostim
        """

    @classmethod
    def expand(cls, condition_id):
        """
        Expand the given condition_id into a dictionary containing the
        fetched sub-parts of the condition.
        """

        self = cls()
        key = {'condition_id': condition_id}

        return {
            'Condition': (self & key).fetch1(),
            'TaskProtocol':
                (TrialCondition.TaskProtocol & key).fetch(as_dict=True),
            'TrialInstruction':
                (TrialCondition.TrialInstruction & key).fetch(as_dict=True),
            'EarlyLick':
                (TrialCondition.EarlyLick & key).fetch(as_dict=True),
            'Outcome':
                (TrialCondition.Outcome & key).fetch(as_dict=True),
            'PhotostimLocation':
                (TrialCondition.PhotostimLocation & key).fetch(as_dict=True)
        }

    @classmethod
    def trials(cls, cond, r={}):
        """
        Get trials for a Condition.

        Accepts either a condition_id as an integer, or the output of
        the 'expand' function, above.

        Each Condition 'part' defined in the Condition is filtered
        to a primary key for the associated child table (pk_map),
        and then restricted through the table defined in 'restrict_map'
        along with experiment.SessionTrial to retrieve the corresponding
        trials for that particular part. In other words, the pseudo-query:

          SessionTrial & restrict_map & pk_map[Condition.Part & cond]

        is performed for each of the trial-parts.

        The intersection of these trial-part results are then combined
        locally to build the result, which is a list of SessionTrial keys.

        The parameter 'r' can be used to add additional query restrictions,
        currently applied to all of the sub-queries.
        """

        self = cls()
        if type(cond) == int:
            cond = self.expand(cond)

        pk_map = {
            'TaskProtocol': experiment.TaskProtocol,
            'TrialInstruction': experiment.TrialInstruction,
            'EarlyLick': experiment.EarlyLick,
            'Outcome': experiment.Outcome,
            'PhotostimLocation': experiment.Photostim
        }
        restrict_map = {
            'TaskProtocol': experiment.BehaviorTrial,
            'TrialInstruction': experiment.BehaviorTrial,
            'EarlyLick': experiment.BehaviorTrial,
            'Outcome': experiment.BehaviorTrial,
            'PhotostimLocation': experiment.PhotostimEvent
        }

        res = []
        for c in cond:
            if c == 'Condition':
                continue

            tup = cond[c]
            tab = restrict_map[c]
            pk = pk_map[c].primary_key

            tup_keys = [{k: t[k] for k in t if k in pk}
                        for t in tup]
            trials = [(experiment.SessionTrial() & (tab() & t & r).proj())
                      for t in tup_keys]

            res.append({tuple(i.values()) for i in
                        reduce(operator.add, (t.proj() for t in trials))})

        return [{'subject_id': t[0], 'session': t[1], 'trial': t[2]}
                for t in sorted(set.intersection(*res))]

    @classmethod
    def populate(cls):
        """
        Table contents for Condition.

        Currently there is no way to initialize a dj.Lookup with parts,
        so we leave contents blank and create a function to explicitly insert.

        This is not run implicitly since it requires database write access.
        """
        self = cls()

        #
        # Condition 0: Audio Delay Task - Contra Hit
        #

        cond_key = {
            'condition_id': 0,
            'condition_desc': 'audio delay contra hit'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='right'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='hit'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 1: Audio Delay Task - Ipsi Hit
        #

        cond_key = {
            'condition_id': 1,
            'condition_desc': 'audio delay ipsi hit'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='left'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='hit'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 2: Audio Delay Task - Contra Error
        #

        cond_key = {
            'condition_id': 2,
            'condition_desc': 'audio delay contra error'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='right'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='miss'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 3: Audio Delay Task - Ipsi Error
        #

        cond_key = {
            'condition_id': 3,
            'condition_desc': 'audio delay ipsi error'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='left'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='miss'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)


@schema
class UnitPsth(dj.Computed):
    definition = """
    -> TrialCondition
    """
    psth_params = {'xmin': -3, 'xmax': 3, 'binsize': 0.04}

    class Unit(dj.Part):  # XXX: merge up to master; reason: recomputing:
        definition = """
        -> master
        -> ephys.Unit
        ---
        unit_psth:                                  longblob
        """

    def make(self, key):
        log.info('UnitPsth.make(): key: {}'.format(key))

        # can e.g. if key['condition_id'] in [1,2,3]: self.make_thiskind(key)
        # for now, we assume one method of processing

        cond = TrialCondition.expand(key['condition_id'])

        # XXX: if / else for different conditions as needed
        # e.g if key['condition_id'] > 3: ..., elif key['condition_id'] == 5

        all_trials = TrialCondition.trials({
            'TaskProtocol': cond['TaskProtocol'],
            'TrialInstruction': cond['TrialInstruction'],
            'EarlyLick': cond['EarlyLick'],
            'Outcome': cond['Outcome']})

        photo_trials = TrialCondition.trials({
            'PhotostimLocation': cond['PhotostimLocation']})

        unstim_trials = [t for t in all_trials if t not in photo_trials]

        # build unique session list from trial list
        sessions = {(t['subject_id'], t['session']) for t in all_trials}
        sessions = [{'subject_id': s[0], 'session': s[1]}
                    for s in sessions]

        # find good units
        units = ephys.Unit & [dict(s, unit_quality='good') for s in sessions]

        # fetch spikes and create per-unit PSTH record
        self.insert1(key)

        i = 0
        n_units = len(units)

        for unit in ({k: u[k] for k in ephys.Unit.primary_key} for u in units):
            i += 1
            if i % 50 == 0:
                log.info('.. unit {}/{} ({:.2f}%)'
                         .format(i, n_units, (i/n_units)*100))
            else:
                log.debug('.. unit {}/{} ({:.2f}%)'
                          .format(i, n_units, (i/n_units)*100))

            q = (ephys.TrialSpikes() & unit & unstim_trials)
            spikes = q.fetch('spike_times')
            spikes = np.concatenate(spikes)

            xmin, xmax, bins = self.psth_params.values()
            psth = list(np.histogram(spikes, bins=np.arange(xmin, xmax, bins)))
            psth[0] = psth[0] / len(unstim_trials) / bins

            self.Unit.insert1({**key, **unit, 'unit_psth': np.array(psth)},
                              allow_direct_insert=True)

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
             'psth': UnitPsth.Unit.unit_psth,
             'raster': Spike * Trial raster [np.array, np.array]
          }

        """

        condition = TrialCondition.expand(condition_key['condition_id'])
        session_key = {k: unit_key[k] for k in experiment.Session.primary_key}

        psth_q = (UnitPsth.Unit & {**condition_key, **unit_key})
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
            t_stat, pval = sc_stats.ttest_ind(contra_trial_spk_rate, ipsi_trial_spk_rate, equal_var=False)

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


@schema
class SelectivityCriteria(dj.Lookup):
    '''
    Selectivity Criteria -
    Indicate significance of unit firing rate for trial type left vs right.

    *_selectivity variables indicate if the unit displays selectivity
    *_preference variables indicate the unit ipsi/contra preference
    '''

    definition = """
    sample_selectivity:                         boolean
    delay_selectivity:                          boolean
    go_selectivity:                             boolean
    global_selectivity:                         boolean
    sample_preference:                          boolean
    delay_preference:                           boolean
    go_preference:                              boolean
    global_preference:                          boolean
    """

    @property
    def contents(self):
        fourset = [(0, 0, 0, 0,), (0, 0, 0, 1,), (0, 0, 1, 0,), (0, 0, 1, 1,),
                   (0, 1, 0, 0,), (0, 1, 0, 1,), (0, 1, 1, 0,), (0, 1, 1, 1,),
                   (1, 0, 0, 0,), (1, 0, 0, 1,), (1, 0, 1, 0,), (1, 0, 1, 1,),
                   (1, 1, 0, 0,), (1, 1, 0, 1,), (1, 1, 1, 0,), (1, 1, 1, 1,)]
        return (i + j for i in fourset for j in fourset)

    ranges = {   # time ranges in SelectivityCriteria order
        'sample_selectivity': (-2.4, -1.2),
        'delay_selectivity': (-1.2, 0),
        'go_selectivity': (0, 1.2),
        'global_selectivity': (-2.4, 1.2),
    }


@schema
class Selectivity(dj.Computed):
    '''
    Unit Selectivity

    Compute unit selectivity based on fixed time regions.

    Calculation:
    2 tail t significance of unit firing rate for trial type l vs r
    frequency = nspikes(period)/len(period)
    '''

    definition = """
    # Unit Response Selectivity (WIP)
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
    global_preference=Null:     boolean       # global period pref. (i|c)
    any_preference=Null:        boolean       # any period pref. (i|c)
    """

    alpha = 0.05  # default alpha value

    @property
    def selective(self):
        return 'min_selectivity<{}'.format(self.alpha)

    ipsi_preferring = 'any_preference=1'
    contra_preferring = 'any_preference=0'

    ranges = {   # time ranges in SelectivityCriteria order
        'sample': (-2.4, -1.2),
        'delay':  (-1.2, -0.0),
        'go':     (+0.0, +1.2),
        'global': (-2.4, +1.2),
    }

    def make(self, key):

        if key['unit'] % 50 == 0:
            log.info('Selectivity.make(): key % 50: {}'.format(key))
        else:
            log.debug('Selectivity.make(): key: {}'.format(key))

        ranges = self.ranges
        spikes_q = ((ephys.TrialSpikes & key)
                    & (experiment.BehaviorTrial()
                       & {'early_lick': 'no early'}))

        lr = ['left', 'right']
        behav = (experiment.BehaviorTrial & spikes_q.proj()).fetch(
            order_by='trial asc')
        behav_lr = {k: np.where(behav['trial_instruction'] == k) for k in lr}

        try:
            egpos = (ephys.ProbeInsertion.InsertionLocation
                     * experiment.BrainLocation & key).fetch1()
        except dj.DataJointError as e:
            if 'exactly one tuple' in repr(e):
                log.error('... Insertion Location missing. skipping')
                return

        # construct a square-shaped spike array, create 'valid value' index
        spikes = spikes_q.fetch(order_by='trial asc')
        ydim = max(len(i['spike_times']) for i in spikes)
        square = np.array(
            np.array([np.concatenate([st, pad])[:ydim]
                      for st, pad in zip(spikes['spike_times'],
                                         repeat([math.nan]*ydim))]))

        criteria = {}

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

            if egpos['hemisphere'] == 'left':
                behav_i = behav_lr['left']
                behav_c = behav_lr['right']
            else:
                behav_i = behav_lr['right']
                behav_c = behav_lr['left']

            freq_i = freq[behav_i]
            freq_c = freq[behav_c]
            t_stat, pval = sc_stats.ttest_ind(freq_i, freq_c, equal_var=False)

            # criteria[name] = 1 if pval <= alpha else 0
            criteria[name] = pval
            criteria[pref] = 1 if np.average(freq_i) > np.average(freq_c) else 0

        criteria['min_selectivity'] = min([v for k, v in criteria.items()
                                           if 'selectivity' in k])

        criteria['any_preference'] = any([v for k, v in criteria.items()
                                          if 'preference' in k])

        self.insert1(dict(key, **criteria))
