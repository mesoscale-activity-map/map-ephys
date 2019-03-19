
import datajoint as dj

from . import lab
from . import experiment
from . import ephys
[lab, experiment, ephys]  # NOQA


schema = dj.schema(dj.config.get('psth.database', 'map_psth'))


@schema
class Condition(dj.Lookup):
    # TODO: this mirrors non-sessiontrial attrs of experiment.BehaviorTrial;
    # should this be moved to experiment and used to build up that table?
    # (implies reingestion, etc)
    # TODO: condition_id pkey??
    definition = """
    -> experiment.TaskProtocol
    -> experiment.TrialInstruction
    -> experiment.EarlyLick
    -> experiment.Outcome
    """

    @property
    def contents(self):
        return (dj.U('task', 'task_protocol', 'trial_instruction',
                     'early_lick', 'outcome') &
                experiment.TaskProtocol *
                experiment.TrialInstruction *
                experiment.EarlyLick * experiment.Outcome)


@schema
class BrainAreaCondition(dj.Computed):
    # see also w/r/t PhtotstimTrial (not same functionality.. but data wise)
    # how recover recording location otherwise?
    definition = """
    -> Condition
    -> lab.BrainArea
    """

    def make(self, key):
        pass


@schema
class BACSession(dj.Computed):
    # see 'Condition' note r.e. duplication of experiment.BehaviorTrial;
    # essentially the same; could simply filter unique sessions from list
    definition = """
    -> BrainAreaCondition
    -> experiment.Session
    """

    def make(self, key):
        for bt in (experiment.BehaviorTrial & key):
            self.insert1(key, skip_duplicates=True, ignore_extra_fields=True)


@schema
class BACTrial(dj.Computed):
    # see 'Condition' note r.e. duplication of experiment.BehaviorTrial
    definition = """
    -> BrainAreaCondition
    -> experiment.SessionTrial
    """

    def make(self, key):
        for bt in (experiment.BehaviorTrial & key):
            self.insert1(key, skip_duplicates=True, ignore_extra_fields=True)


@schema
class CellPsth(dj.Computed):
    definition = """
    -> BrainAreaCondition
    -> ephys.Unit
    ---
    cell_psth:          longblob
    """


@schema
class CellTypePsth(dj.Computed):
    # todo: how group? for now, used celltype as placeholder;
    # potentially >1x tables or another intermediate table to be created..
    definition = """
    -> BrainAreaCondition
    -> ephys.CellType
    ---
    cell_type_psth:          longblob
    """

    class CTPUnit(dj.Part):
        definition = """
        -> master
        -> ephys.Unit
        """
