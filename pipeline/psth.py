
import logging
import operator

from functools import reduce

import numpy as np
import datajoint as dj

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
        -> experiment.PhotostimLocation
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
    def trials(cls, cond):
        """
        get trials for a condition.
        accepts either a condition_id as an integer,
        or the output of the 'expand' function, above.
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
            trials = [(experiment.SessionTrial() & (tab() & t))
                      for t in tup_keys]

            res.append({tuple(i.values()) for i in
                        reduce(operator.add, (t.proj() for t in trials))})

        return [{'subject_id': t[0], 'session': t[1], 'trial': t[2]}
                for t in set.intersection(*res)]

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
            'condition_id': 2,
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

        for unit in ({k: u[k] for k in ephys.Unit.primary_key} for u in units):
            log.debug('.. per unit psth: {}'.format(unit))

            q = (ephys.TrialSpikes() & unit & unstim_trials)
            spikes = q.fetch('spike_times')
            spikes = np.concatenate(spikes)

            xmin, xmax, bins = CellPsth.psth_params.values()
            psth = list(np.histogram(spikes, bins=np.arange(xmin, xmax, bins)))
            psth[0] = psth[0] / len(unstim_trials) / bins

            CellPsth.Unit.insert1({**key, **unit, 'cell_psth': psth[0]},
                                  allow_direct_insert=True)


@schema
class SelectivityCriteria(dj.Lookup):
    '''
    Celectivity Criteria -
    significance of unit firing rate for trial type left vs right
    significant for right
    '''

    definition = """
    sample_selectivity:                         boolean
    delay_selectivity:                          boolean
    go_selectivity:                             boolean
    """
    contents = [(0, 0, 0,), (0, 0, 1,),
                (0, 1, 0,), (0, 1, 1,),
                (1, 0, 0,), (1, 0, 1,),
                (1, 1, 0,), (1, 1, 1,)]


@schema
class Selectivity(dj.Computed):
    '''
    Unit selectivity
    significance of unit firing rate for trial type l vs r
    '''
    definition = """
    # Unit Response Selectivity
    -> ephys.Unit
    ---
    -> SelectivityCriteria
    """

    def make(self, key):
        log.info('Selectivity.make(): key: {}'.format(key))

        # unit = (ephys.Unit & key).fetch1()
        # trials = (ephys.Unit.UnitTrial & key).fetch()

        session = {k: key[k] for k in experiment.Session.primary_key}

        active = (ephys.TrialSpikes & key).fetch()
        trials = [{k: a[k] for k in experiment.SessionTrial.primary_key}
                  for a in active]

        events = (experiment.TrialEvent & key).fetch()

        # BOOKMARK:
        # - dealing with duplicate sample events;
        # - building arrays so that bulk computation can be done in one shot.
        presample = events[np.where(events['trial_event_type'] == 'presample')]
        go = events[np.where(events['trial_event_type'] == 'go')]
        sample = events[np.where(events['trial_event_type'] == 'sample')]
        trialend = events[np.where(events['trial_event_type'] == 'trialend')]

        # np.where((active2['subject_id'] == 90211) & (active2['session'] == 1) & (active2['trial'] == 169))

        # for t in (trials[0],):
        # for t in trials:
        #     tevent = (experiment.TrialEvent & t).fetch(as_dict=True)
        #     tspike_idx = np.where((active['subject_id'] == t['subject_id'])
        #                           & (active['session'] == t['session'])
        #                           & (active['trial'] == t['trial']))
        #     tspike = active[tspike_idx]
        # go, presample, sample, trialend

        from code import interact
        from collections import ChainMap
        interact('Selectvity make REPL',
                 local=dict(ChainMap(globals(), locals())))


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
