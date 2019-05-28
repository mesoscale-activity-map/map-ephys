
import logging
import operator
import math

from functools import reduce
from itertools import repeat

import numpy as np
import scipy as sc
import datajoint as dj

import scipy.stats  # NOQA

from pipeline import lab
from pipeline import experiment
from pipeline import ephys
[lab, experiment, ephys]  # NOQA

log = logging.getLogger(__name__)
schema = dj.schema(dj.config.get('psth.database', 'map_psth'))


@schema
class Condition(dj.Manual):
    definition = """
    # manually curated conditions of interest
    condition_id:                               int
    ---
    condition_desc:                             varchar(4096)
    """

    class TaskProtocol(dj.Part):
        definition = """
        -> master
        -> experiment.TaskProtocol
        """

    class TrialInstruction(dj.Part):
        definition = """
        -> master
        -> experiment.TrialInstruction
        """

    class EarlyLick(dj.Part):
        definition = """
        -> master
        -> experiment.EarlyLick
        """

    class Outcome(dj.Part):
        definition = """
        -> master
        -> experiment.Outcome
        """

    class PhotostimLocation(dj.Part):
        definition = """
        -> master
        -> experiment.Photostim
        """

    @classmethod
    def expand(cls, condition_id):

        self = cls()
        key = {'condition_id': condition_id}

        return {
            'Condition': (self & key).fetch1(),
            'TaskProtocol':
                (Condition.TaskProtocol & key).fetch(as_dict=True),
            'TrialInstruction':
                (Condition.TrialInstruction & key).fetch(as_dict=True),
            'EarlyLick':
                (Condition.EarlyLick & key).fetch(as_dict=True),
            'Outcome':
                (Condition.Outcome & key).fetch(as_dict=True),
            'PhotostimLocation':
                (Condition.PhotostimLocation & key).fetch(as_dict=True)
        }

    @classmethod
    def trials(cls, cond, r={}):
        """
        get trials for a condition.
        accepts either a condition_id as an integer,
        or the output of the 'expand' function, above.

        the parameter 'r' can be used add additional query restrictions.
        """

        self = cls()
        if type(cond) == int:
            cond = self.expand(cond)

        pk_map = {
            'TaskProtocol': experiment.TaskProtocol,
            'TrialInstruction': experiment.TrialInstruction,
            'EarlyLick': experiment.EarlyLick,
            'Outcome': experiment.Outcome,
            'PhotostimLocation': experiment.PhotostimLocation
        }
        restrict_map = {
            'TaskProtocol': experiment.BehaviorTrial,
            'TrialInstruction': experiment.BehaviorTrial,
            'EarlyLick': experiment.BehaviorTrial,
            'Outcome': experiment.BehaviorTrial,
            'PhotostimLocation': experiment.PhotostimTrialEvent
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
            trials = [(experiment.SessionTrial() & (tab() & t & r))
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

        Condition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='right'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.Outcome.insert1(
            dict(cond_key, outcome='hit'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.PhotostimLocation & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 1: Audio Delay Task - Ipsi Hit
        #

        cond_key = {
            'condition_id': 1,
            'condition_desc': 'audio delay ipsi hit'
        }
        self.insert1(cond_key, skip_duplicates=True)

        Condition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='left'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.Outcome.insert1(
            dict(cond_key, outcome='hit'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.PhotostimLocation & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 2: Audio Delay Task - Contra Error
        #

        cond_key = {
            'condition_id': 2,
            'condition_desc': 'audio delay contra error'
        }
        self.insert1(cond_key, skip_duplicates=True)

        Condition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='right'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.Outcome.insert1(
            dict(cond_key, outcome='miss'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.PhotostimLocation & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 3: Audio Delay Task - Ipsi Error
        #

        cond_key = {
            'condition_id': 3,
            'condition_desc': 'audio delay ipsi error'
        }
        self.insert1(cond_key, skip_duplicates=True)

        Condition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='left'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.Outcome.insert1(
            dict(cond_key, outcome='miss'),
            skip_duplicates=True, ignore_extra_fields=True)

        Condition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.PhotostimLocation & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)


@schema
class CellPsth(dj.Computed):
    definition = """
    -> Condition
    """
    psth_params = {'xmin': -3, 'xmax': 3, 'binsize': 0.04}

    class Unit(dj.Part):
        definition = """
        -> master
        -> ephys.Unit
        ---
        cell_psth:                                  longblob
        """

    def make(self, key):
        log.info('CellPsth.make(): key: {}'.format(key))

        # can e.g. if key['condition_id'] in [1,2,3]: self.make_thiskind(key)
        # for now, we assume one method of processing

        cond = Condition.expand(key['condition_id'])

        all_trials = Condition.trials({
            'TaskProtocol': cond['TaskProtocol'],
            'TrialInstruction': cond['TrialInstruction'],
            'EarlyLick': cond['EarlyLick'],
            'Outcome': cond['Outcome']})

        photo_trials = Condition.trials({
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

            xmin, xmax, bins = CellPsth.psth_params.values()
            psth = list(np.histogram(spikes, bins=np.arange(xmin, xmax, bins)))
            psth[0] = psth[0] / len(unstim_trials) / bins

            CellPsth.Unit.insert1({**key, **unit, 'cell_psth': np.array(psth)},
                                  allow_direct_insert=True)


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

    lick instruction(-2.4,-1.2)  # sound is playing,
    pre trial delay wait(-1.2,0)  # delay
    go(0,1.2)  # perform lick
    '''

    definition = """
    # Unit Response Selectivity
    -> ephys.Unit
    ---
    -> SelectivityCriteria
    """

    def make(self, key):

        if key['unit'] % 50 == 0:
            log.info('Selectivity.make(): key % 50: {}'.format(key))
        else:
            log.debug('Selectivity.make(): key: {}'.format(key))

        alpha = 0.05  # TODO: confirm
        ranges = SelectivityCriteria.ranges
        spikes_q = ((ephys.TrialSpikes & key)
                    & (experiment.BehaviorTrial()
                       & {'early_lick': 'no early'}))

        lr = ['left', 'right']
        behav = (experiment.BehaviorTrial & spikes_q.proj()).fetch(
            order_by='trial asc')
        behav_lr = {k: np.where(behav['trial_instruction'] == k) for k in lr}

        try:
            egpos = (ephys.ElectrodeGroup.ElectrodeGroupPosition()
                     & key).fetch1()
        except dj.DataJointError as e:
            if 'exactly one tuple' in repr(e):
                log.error('... ElectrodeGroupPosition missing. skipping')
                return

        # construct a square-shaped spike array, create 'valid value' index
        spikes = spikes_q.fetch(order_by='trial asc')
        ydim = max(len(i['spike_times']) for i in spikes)
        square = np.array(
            np.array([np.concatenate([st, pad])[:ydim]
                      for st, pad in zip(spikes['spike_times'],
                                         repeat([math.nan]*ydim))]))

        criteria = {}

        for name, bounds in ranges.items():
            pref = name.split('_')[0] + '_preference'

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
            t_stat, pval = sc.stats.ttest_ind(freq_i, freq_c, equal_var=False)

            criteria[name] = 1 if pval <= alpha else 0
            criteria[pref] = 1 if np.average(freq_i) > np.average(freq_c) else 0

        log.info('criteria: {}'.format(criteria))
        self.insert1(dict(key, **criteria))


@schema
class CellGroupCondition(dj.Manual):
    definition = """
    # manually curated cell groups of interest
    -> Condition
    cell_group_condition_id:                    int
    ---
    cell_group_condition_desc:                  varchar(4096)
    -> lab.BrainArea
    -> SelectivityCriteria
    """

    @classmethod
    def populate(cls):
        """
        Table contents for CellGroupCondition
        """
        self = cls()
        self.insert1({
            'condition_id': 0,
            'cell_group_condition_id': 0,
            'cell_group_condition_desc': '''
            audio delay contra hit - high selectivity; ALM
            ''',
            'brain_area': 'ALM',
            'sample_selectivity': 1,
            'delay_selectivity': 1,
            'go_selectivity': 1,
            'global_selectivity': 1,
            'sample_preference': 1,
            'delay_preference': 1,
            'go_preference': 1,
            'global_preference': 1,
        }, skip_duplicates=True)

        self.insert1({
            'condition_id': 0,
            'cell_group_condition_id': 0,
            'cell_group_condition_desc': '''
            audio delay contra hit - high selectivity; ALM
            ''',
            'brain_area': 'ALM',
            'sample_selectivity': 1,
            'delay_selectivity': 1,
            'go_selectivity': 1,
            'global_selectivity': 1,
            'sample_preference': 0,
            'delay_preference': 0,
            'go_preference': 0,
            'global_preference': 0,
        }, skip_duplicates=True)


        '''
        take average PSTH for all (contra|ipsi) trials
        refile: bitcode checking
        '''


@schema
class CellGroupPsth(dj.Computed):
    definition = """
    -> CellGroupCondition
    ---
    cell_group_psth:                            longblob
    """

    class Unit(dj.Part):
        definition = """
        # unit backreference for group psth
        -> master
        -> ephys.Unit
        """

    def make(self, key):
        log.info('CellGroupPsth.make(): key: {}'.format(key))

        # CellPsth.Unit & {k: key[k] for k in Condition.primary_key}

        group_cond = (CellGroupCondition & key).fetch1()

        unit_psth_q = (
            (CellPsth.Unit & {k: key[k] for k in Condition.primary_key})
            & (Selectivity & {k: group_cond[k]
                              for k in SelectivityCriteria.primary_key}))

        unit_psth = unit_psth_q.fetch()
        [unit_psth]

        # BOOKMARK: calculations
        # from code import interact
        # from collections import ChainMap
        # interact('cellgrouppsth', local=dict(ChainMap(locals(), globals())))
