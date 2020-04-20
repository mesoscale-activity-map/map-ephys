import datajoint as dj

from pipeline import get_schema_name

schema = dj.schema(get_schema_name('fixes'))


@schema
class FixHistory(dj.Manual):
    """
    Any fixes requiring accompanying tables, those tables must be children of this FixHistory
    """
    definition = """
    fix_name:   varchar(255)
    fix_timestamp:   timestamp
    """
