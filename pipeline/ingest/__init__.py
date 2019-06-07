import datajoint as dj

try:
    ingest_db_prefix = dj.config['custom'].get('ingest.behavior.database')
except:
    ingest_db_prefix = '{}_ingest_'.format(dj.config['database.user'])
