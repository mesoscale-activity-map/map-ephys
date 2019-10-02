
import os
import logging
import pathlib

from textwrap import dedent

import datajoint as dj

from . import lab
from . import experiment
from . import ephys
from . import tracking


from pipeline.globus import GlobusStorageManager
from . import get_schema_name

schema = dj.schema(get_schema_name('publication'))

log = logging.getLogger(__name__)
__all__ = [experiment, ephys]


@schema
class GlobusStorageLocation(dj.Lookup):
    """ globus storage locations """

    definition = """
    globus_alias:       varchar(32)     # name for location (e.g. 'raw-ephys')
    ---
    globus_endpoint:    varchar(255)    # globus endpoint (user#endpoint)
    globus_path:        varchar(1024)   # unix-style path within endpoint
    """

    @property
    def contents(self):
        custom = dj.config.get('custom', None)
        if custom and 'globus.storage_locations' in custom:  # test config
            return custom['globus.storage_locations']

        return (('raw-ephys', '5b875fda-4185-11e8-bb52-0ac6873fc732', '/'),
                ('raw-tracking', '5b875fda-4185-11e8-bb52-0ac6873fc732', '/'),)

    @classmethod
    def local_endpoint(cls, globus_alias=None):
        '''
        return local endpoint for globus_alias from dj.config
        expects:
          globus.local_endpoints: {
            globus_alias: {
              'endpoint': uuid,  # UUID of local endpoint
              'endpoint_subdir': str,  # unix-style path within endpoint
              'endpoint_path': str  # corresponding local path
          }
        '''
        le = dj.config.get('custom', {}).get('globus.local_endpoints', None)

        if le is None or globus_alias not in le:

            raise dj.DataJointError(
                "globus_local_endpoints for {} not configured".format(
                    globus_alias))

        return le[globus_alias]


@schema
class ArchivedSession(dj.Imported):
    definition = """
    -> experiment.Session
    ---
    -> GlobusStorageLocation
    """


@schema
class DataSetType(dj.Lookup):
    definition = """
    dataset_type: varchar(64)
    """

    contents = zip(['ephys-raw-trialized',
                    'ephys-raw-continuous',
                    'ephys-sorted',
                    'tracking-video'])


@schema
class FileType(dj.Lookup):
    definition = """
    file_type:            varchar(32)           # file type short name
    ---
    file_glob:            varchar(64)           # file match pattern
    file_descr:           varchar(255)          # file type long description
    """

    @property
    def contents(self):

        data = [('3a-ap-trial',
                 '*_g0_t[0-9]*imec.ap.bin',
                 '''
                 3A Probe per-trial AP channels high pass filtered at 
                 300Hz and sampled at 30kHz - recording file
                 '''),
                ('3a-ap-trial-meta',
                 '*_g0_t[0-9]*.imec.ap.meta',
                 '''
                 3A Probe per-trial AP channels high pass 
                 filtered at 300Hz and sampled at 30kHz - file metadata
                 '''),
                ('3a-lf-trial',
                 '*_g0_t[0-9]*.imec.lf.bin',
                 '''
                 3A Probe per-trial AP channels low pass filtered at 
                 300Hz and sampled at 2.5kHz - recording file
                 '''),
                ('3a-lf-trial-meta',
                 '*_g0_t[0-9]*.imec.lf.meta',
                 '''
                 3A Probe per-trial AP channels low pass filtered at 
                 300Hz and sampled at 2.5kHz - file metadata
                 '''),
                ('3b-ap-trial',
                 '*_????????_g?_t[0-9]*.imec.ap.bin',
                 '''
                 3B Probe per-trial AP channels high pass filtered at 
                 300Hz and sampled at 30kHz - recording file
                 '''),
                ('3b-ap-trial-meta',
                 '*_????????_g?_t[0-9]*.imec.ap.bin',
                 '''
                 3B Probe per-trial AP channels high pass 
                 filtered at 300Hz and sampled at 30kHz - file metadata
                 '''),
                ('3b-lf-t[0-9]*ial',
                 '*_????????_g?_t[0-9]*.imec.ap.bin',
                 '''
                 3B Probe per-trial AP channels low pass filtered at 
                 300Hz and sampled at 2.5kHz - recording file
                 '''),
                ('3b-lf-trial-meta',
                 '*_????????_g?_t[0-9]*.imec.ap.bin',
                 '''
                 3B Probe per-trial AP channels low pass filtered at 
                 300Hz and sampled at 2.5kHz - file metadata
                 '''),
                ('3b-ap-concat',
                 '*_????????_g?_tcat.imec.ap.bin',
                 '''
                 3B Probe concatenated AP channels high pass filtered at 
                 300Hz and sampled at 30kHz - recording file
                 '''),
                ('3b-ap-concat-meta',
                 '*_??????_g?_tcat.imec.ap.bin',
                 '''
                 3B Probe concatenated AP channels high pass 
                 filtered at 300Hz and sampled at 30kHz - file metadata
                 '''),
                ('3b-lf-concat',
                 '*_????????_g?_tcat.imec.ap.bin',
                 '''
                 3B Probe concatenated AP channels low pass filtered at 
                 300Hz and sampled at 2.5kHz - recording file
                 '''),
                ('3b-lf-concat-meta',
                 '*_????????_g?_tcat.imec.ap.bin',
                 '''
                 3B Probe concatenated AP channels low pass filtered at 
                 300Hz and sampled at 2.5kHz - file metadata
                 ''')]

        return [[dedent(i).lstrip('\n') for i in r] for r in data]


@schema
class DataSet(dj.Manual):
    definition = """
    -> GlobusStorageLocation
    dataset_name:               varchar(128)
    ---
    -> DataSetType
    """

    class PhysicalFile(dj.Part):
        definition = """
        -> master
        file_subpath:           varchar(128)
        ---
        -> FileType
        """


@schema
class ArchivedRawEphys(dj.Imported):
    definition = """
    -> ArchivedSession
    -> DataSet
    probe_folder:               tinyint
    """
    
    key_source = experiment.Session

    class RawEphysTrial(dj.Part):
        """ file:trial mapping if applicable """

        definition = """
        -> master
        -> experiment.SessionTrial
        ---
        -> DataSet.PhysicalFile
        """

    def make(self, key):
        """
        discover files in local endpoint and transfer/register
        """

        log.debug(key)
        globus_alias = 'raw-ephys'
        le = GlobusStorageLocation.local_endpoint(globus_alias)
        lep, lep_sub, lep_dir = (le['endpoint'],
                                 le['endpoint_subdir'],
                                 le['endpoint_path'])

        log.info('local_endpoint: {}:{} -> {}'.format(lep, lep_sub, lep_dir))

        # get session related information needed for filenames/records
        # XXX: trial still needed? not registering trials
        sinfo = (lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session() & key).fetch1()

        # XX: only for trialized
        # tinfo = ((lab.WaterRestriction
        #           * lab.Subject.proj()
        #           * experiment.Session()
        #           * experiment.SessionTrial) & key).fetch1()

        h2o = sinfo['water_restriction_number']
        sdate = sinfo['session_date']

        from code import interact
        from collections import ChainMap
        interact('RawEphysTrial.make() repl',
                 local=dict(ChainMap(locals(), globals())))

        subdir = os.path.join(h2o, str(sdate).replace('-',''), str(eg))

        # XXX: session:probe



@schema
class ArchivedSortedEphys(dj.Imported):
    definition = """
    -> ArchivedSession
    -> DataSet
    probe_folder:               tinyint
    ---
    sorting_time=null:          datetime
    """

    key_source = experiment.Session

    def make(self, key):
        """
        discover files in local endpoint and transfer/register
        """
        pass




@schema
class ArchivedVideoTracking(dj.Imported):
    definition = """
    -> ArchivedSession
    ---
    -> DataSet
    """

    class TrialVideo(dj.Part):
        definition = """
        -> tracking.Tracking
        ---
        -> DataSet.PhysicalFile
        """

    def make(self, key):
        """
        discover files in local endpoint and transfer/register
        """
        pass


