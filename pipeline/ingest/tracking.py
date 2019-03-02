# tracking_data_path removed from main; should go here += file traversal

'''
dl56_side_3-0000.csv : h2o_<camera>_3_XXXX (ignore XXXX)
file paths: root/h2o/date/'string of tracking'/file
(same as ephys, 'tracking' instead of '1','2',etc)
  - values are detected positions and video frame number
  - paths root/h2o/date/tracking/*.csv
  - file, 'dl59_20181207_side.txt' tracking record no -> trial no
'''

import os
import logging
from glob import glob

import datajoint as dj

from pipeline import experiment
from pipeline.ingest import behavior as ingest_behavior

log = logging.getLogger(__name__)


schema = dj.schema(dj.config.get(
    'ingest.tracking.database',
    '{}_ingestTracking'.format(dj.config['database.user'])))


@schema
class TrackingDataPath(dj.Lookup):
    # ephys data storage location(s)
    definition = """
    tracking_data_path:         varchar(255)            # rig data path
    """

    @property
    def contents(self):
        if 'tracking_data_paths' in dj.config:  # for local testing
            return dj.config['tracking_data_paths']

        return [(r'H:\\data\MAP',)]


@schema
class TrackingIngest(dj.Imported):
    definition = """
    -> ingest_behavior.BehaviorIngest
    """

    class TrackingFile(dj.Part):
        definition = '''
        -> TrackingIngest
        -> experiment.SessionTrial
        ---
        tracking_file:          varchar(255)            # tracking file subpath
        '''

    def make(self, key):
        '''
        TrackingIngest .make() function
        '''
        log.info('TrackingIngest().make(): key: {k}'.format(k=key))
