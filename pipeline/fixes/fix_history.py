
import datajoint as dj

from pipeline import experiment
from .. import get_schema_name


schema = dj.schema(get_schema_name('map_v1_fix_history'))


@schema
class FixHistory(dj.Manual):
    definition = """
    fix_name:   varchar(255)
    fix_timestamp:   timestamp
    """

    class FixHistorySession(dj.Part):
        definition = """
        -> FixHistory
        -> experiment.Session
        """
