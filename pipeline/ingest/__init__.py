import datajoint as dj

try:
    _ingest_db_prefix = dj.config['custom'].get('ingest.behavior.database')
except:
    _ingest_db_prefix = '{}_ingest_'.format(dj.config['database.user'])


def get_schema_name(name):
    try:
         return dj.config['custom']['ingest.{}.database'.format(name)]
    except KeyError:
        return _ingest_db_prefix + name
