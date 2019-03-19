
import datajoint as dj

from . import lab
from . import experiment
from . import ephys
[lab, experiment, ephys]  # NOQA


schema = dj.schema(dj.config.get('psth.database', 'map_psth'))


@schema
class Condition(dj.Lookup):
    definition = """
    # manually curated conditions of interest
    condition_id:                               int
    ---
    condition_desc:                             varchar(4096)
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
class CellGroupCondition(dj.Manual):
    definition = """
    # manually curated cell groups of interest
    -> Condition
    cell_group_condition_id:                    int
    ---
    condition_desc:                             varchar(4096)
    -> lab.BrainArea
    """

    class CellGroupConditionSessions(dj.Part):
        definition = """
        -> master
        -> experiment.Session
        """


@schema
class CellPsth(dj.Computed):
    definition = """
    -> ephys.Unit
    -> Condition
    ---
    cell_psth:          longblob
    """


@schema
class CellGroupPsth(dj.Computed):
    definition = """
    -> CellGroupCondition
    ---
    cell_group_psth:          longblob
    """
