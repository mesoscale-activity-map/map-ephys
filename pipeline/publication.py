
import os
import logging
import pathlib

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

        return (('raw-ephys', '5b875fda-4185-11e8-bb52-0ac6873fc732', '/')
                ('raw-video', '5b875fda-4185-11e8-bb52-0ac6873fc732', '/'))

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
    

@schema
class GlobusPublishedDataSet(dj.Manual):
    """ Datasets published via Globus """
    definition = """
    globus_collection_name:    varchar(255)    # globus publication collection
    globus_dataset_name:       varchar(255)    # globus dataset name
    ---
    -> GlobusStorageLocation
    globus_doi:                varchar(1000)   # dataset DOI URL
    """


@schema
class RawEphysFileTypes(dj.Lookup):
    """
    Raw Ephys file types/file suffixes:
    """
    # decimal(8, 4)
    definition = """
    raw_ephys_filetype:         varchar(32)     # short filetype description
    ---
    raw_ephys_suffix:           varchar(16)     # file suffix
    raw_ephys_freq=NULL:        Decimal(8, 4)   # kHz; NULL if n/a
    raw_ephys_descr:            varchar(255)    # more detailed description
    """
    contents = [('ap-30kHz',
                 '.imec.ap.bin',
                 30.0,
                 'ap channels @30kHz'),
                ('ap-30kHz-meta',
                 '.imec.ap.meta',
                 None,
                 "recording metadata for 'ap-30kHz' files"),
                ('lf-2.5kHz',
                 '.imec.lf.bin',
                 2.5,
                 'lf channels @2.5kHz'),
                ('lf-2.5kHz-meta',
                 '.imec.lf.meta',
                 None,
                 "recording metadata for 'lf-2.5kHz' files")]

@schema
class ArchivedRawEphysTrial(dj.Imported):
    """
    Table to track archive of raw ephys trial data.

    Directory locations of the form:

    {Water restriction number}\{Session Date}\{electrode_group number}

    with file naming convention of the form:

    {water_restriction_number}_{session_date}_{electrode_group}_g0_t{trial}.{raw_ephys_suffix}
    """

    definition = """
    -> experiment.SessionTrial
    -> ephys.ProbeInsertion
    ---
    -> GlobusStorageLocation
    """

    gsm = None  # for GlobusStorageManager

    class ArchivedApChannel(dj.Part):
        definition = """
        -> ArchivedRawEphysTrial
        """

    class ArchivedApMeta(dj.Part):
        definition = """
        -> ArchivedRawEphysTrial
        """

    class ArchivedLfChannel(dj.Part):
        definition = """
        -> ArchivedRawEphysTrial
        """

    class ArchivedLfMeta(dj.Part):
        definition = """
        -> ArchivedRawEphysTrial
        """

    def get_gsm(self):
        log.debug('ArchivedRawEphysTrial.get_gsm()')
        if self.gsm is None:
            self.gsm = GlobusStorageManager()

        return self.gsm

    def make(self, key):
        '''
        determine available files from local endpoint and publish
        (create database records and transfer to globus)
        '''

        # >>> list(key.keys())
        # ['subject_id', 'session', 'trial', 'electrode_group', 'globus_alias']

        log.debug(key)
        globus_alias = 'raw-ephys'
        le = GlobusStorageLocation.local_endpoint(globus_alias)
        lep, lep_sub, lep_dir = (le['endpoint'],
                                 le['endpoint_subdir'],
                                 le['endpoint_path'])

        log.info('local_endpoint: {}:{} -> {}'.format(lep, lep_sub, lep_dir))

        # get session related information needed for filenames/records
        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session()
                  * experiment.SessionTrial) & key).fetch1()

        h2o = sinfo['water_restriction_number']
        sdate = sinfo['session_date']
        eg = key['electrode_group']
        trial = key['trial']

        # build file locations:
        # subdir - common subdirectory for globus/native filesystem
        # fpat: base file pattern for this sessions files
        # fbase: filesystem base path for this sessions files
        # gbase: globus-url base path for this sessions files

        subdir = os.path.join(h2o, str(sdate), str(eg))
        fpat = '{}_{}_{}_g0_t{}'.format(h2o, sdate, eg, trial)
        fbase = os.path.join(lep_dir, subdir, fpat)
        gbase = '/'.join((h2o, str(sdate), str(eg), fpat))

        # check for existence of actual files & use to build xfer list
        log.debug('checking {}'.format(fbase))

        ffound = []
        ftypes = RawEphysFileTypes.contents
        for ft in ftypes:
            fname = '{}{}'.format(fbase, ft[1])
            gname = '{}{}'.format(gbase, ft[1])
            if not os.path.exists(fname):
                log.debug('... {}: not found'.format(fname))
                continue

            log.debug('... {}: found'.format(fname))
            ffound.append((ft, gname,))

        # if files are found, transfer and create publication schema records

        if not len(ffound):
            log.info('no files found for key')
            return

        log.info('found files for key: {}'.format([f[1] for f in ffound]))

        repname, rep, rep_sub = (GlobusStorageLocation()
                                 & {'globus_alias': globus_alias}).fetch(
                                     limit=1)[0]

        gsm = self.get_gsm()
        gsm.activate_endpoint(lep)  # XXX: cache this / prevent duplicate RPC?
        gsm.activate_endpoint(rep)  # XXX: cache this / prevent duplicate RPC?

        if not self & key:
            log.info('ArchivedRawEphysTrial.insert1()')
            self.insert1({**key, 'globus_alias': globus_alias})

        ftmap = {'ap-30kHz': ArchivedRawEphysTrial.ArchivedApChannel,
                 'ap-30kHz-meta': ArchivedRawEphysTrial.ArchivedApMeta,
                 'lf-2.5kHz': ArchivedRawEphysTrial.ArchivedLfChannel,
                 'lf-2.5kHz-meta': ArchivedRawEphysTrial.ArchivedLfMeta}

        for ft, gname in ffound:  # XXX: transfer/insert could be batched
            ft_class = ftmap[ft[0]]
            if not ft_class & key:
                srcp = '{}:/{}/{}'.format(lep, lep_sub, gname)
                dstp = '{}:/{}/{}'.format(rep, rep_sub, gname)

                log.info('transferring {} to {}'.format(srcp, dstp))

                # XXX: check if exists 1st? (manually or via API copy-checksum)
                if not gsm.cp(srcp, dstp):
                    emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                    log.error(emsg)
                    raise dj.DataJointError(emsg)

                log.info('ArchivedRawEphysTrial.{}.insert1()'
                         .format(ft_class.__name__))

                ft_class.insert1(key)

    @classmethod
    def retrieve(cls):
        self = cls()
        for key in self:
            self.retrieve1(key)

    @classmethod
    def retrieve1(cls, key):
        '''
        retrieve related files for a given key
        '''
        self = cls()

        # >>> list(key.keys())
        # ['subject_id', 'session', 'trial', 'electrode_group', 'globus_alia

        log.debug(key)
        lep, lep_sub, lep_dir = GlobusStorageLocation().local_endpoint
        log.info('local_endpoint: {}:{} -> {}'.format(lep, lep_sub, lep_dir))

        # get session related information needed for filenames/records
        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session()
                  * experiment.SessionTrial) & key).fetch1()

        h2o = sinfo['water_restriction_number']
        sdate = sinfo['session_date']
        eg = key['electrode_group']
        trial = key['trial']

        # build file locations:
        # fpat: base file pattern for this sessions files
        # gbase: globus-url base path for this sessions files

        fpat = '{}_{}_{}_g0_t{}'.format(h2o, sdate, eg, trial)
        gbase = '/'.join((h2o, str(sdate), str(eg), fpat))

        repname, rep, rep_sub = (GlobusStorageLocation() & key).fetch()[0]

        gsm = self.get_gsm()
        gsm.activate_endpoint(lep)  # XXX: cache this / prevent duplicate RPC?
        gsm.activate_endpoint(rep)  # XXX: cache this / prevent duplicate RPC?

        sfxmap = {'.imec.ap.bin': ArchivedRawEphysTrial.ArchivedApChannel,
                  '.imec.ap.meta': ArchivedRawEphysTrial.ArchivedApMeta,
                  '.imec.lf.bin': ArchivedRawEphysTrial.ArchivedLfChannel,
                  '.imec.lf.meta': ArchivedRawEphysTrial.ArchivedLfMeta}

        for sfx, cls in sfxmap.items():
            if cls & key:
                log.debug('record found for {} & {}'.format(cls.__name__, key))
                gname = '{}{}'.format(gbase, sfx)

                srcp = '{}:/{}/{}'.format(rep, rep_sub, gname)
                dstp = '{}:/{}/{}'.format(lep, lep_sub, gname)

                log.info('transferring {} to {}'.format(srcp, dstp))

                # XXX: check if exists 1st? (manually or via API copy-checksum)
                if not gsm.cp(srcp, dstp):
                    emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                    log.error(emsg)
                    raise dj.DataJointError(emsg)


@schema
class ArchivedVideoFile(dj.Imported):
    '''
    ArchivedVideoFile storage

    Note: video_file_name tracked here as trial->file map is non-deterministic

    Directory locations of the form:

    {Water restriction number}\{Session Date}\video

    with file naming convention of the form:

    {Water restriction number}_{camera-position-string}_NNN-NNNN.avi

    Where 'NNN' is determined from the 'tracking map file' which maps
    trials to videos as outlined in tracking.py

    XXX:

    Using key-source based loookup as is currently done,
    may have trials for which there is no tracking,
    so camera cannot be determined to do file lookup, thus videos are missed.
    This could be resolved via schema adjustment, or file-traversal
    based 'opportunistic' registration strategy.
    '''

    definition = """
    -> tracking.Tracking
    ---
    -> GlobusStorageLocation
    video_file_name:                     varchar(1024)  # file name for trial
    """

    ingest = None  # ingest module reference
    gsm = None  # for GlobusStorageManager

    @classmethod
    def get_ingest(cls):
        '''
        return tracking_ingest module
        not imported globally to prevent ingest schema creation for client case
        '''
        log.debug('ArchivedVideoFile.get_ingest()')
        if cls.ingest is None:
            from .ingest import tracking as tracking_ingest
            cls.ingest = tracking_ingest

        return cls.ingest

    def get_gsm(self):
        log.debug('ArchivedVideoFile.get_gsm()')
        if self.gsm is None:
            self.gsm = GlobusStorageManager()

        return self.gsm

    def make(self, key):
        '''
        determine available files from local endpoint and publish
        (create database records and transfer to globus)
        '''
        log.info('ArchivedVideoFile.make(): {}'.format(key))

        globus_alias = 'raw-video'
        le = GlobusStorageLocation.local_endpoint(globus_alias)
        lep, lep_sub, lep_dir = (le['endpoint'],
                                 le['endpoint_subdir'],
                                 le['endpoint_path'])

        log.info('local_endpoint: {}:{} -> {}'.format(lep, lep_sub, lep_dir))

        h2o = (lab.WaterRestriction & key).fetch1('water_restriction_number')

        trial = key['trial']
        session = (experiment.Session & key).fetch1()
        sdate = session['session_date']
        sdate_iso = sdate.isoformat()  # YYYY-MM-DD
        sdate_sml = "{}{:02d}{:02d}".format(sdate.year, sdate.month, sdate.day)

        trk = (tracking.TrackingDevice
               * (tracking.Tracking & key).proj()).fetch1()

        tdev = trk['tracking_device']  # NOQA: notused
        tpos = trk['tracking_position']

        camtrial = '{}_{}_{}.txt'.format(h2o, sdate_sml, tpos)

        tracking_ingest = self.get_ingest()
        tpaths = tracking_ingest.TrackingDataPath.fetch(as_dict=True)

        campath = None
        tbase, vbase = None, None  # tracking, video session base paths
        for p in tpaths:

            tdat = p['tracking_data_path']

            tbase = pathlib.Path(tdat, h2o, sdate_iso, 'tracking')
            vbase = pathlib.Path(tdat, h2o, sdate_iso, 'video')

            campath = tbase / camtrial

            log.debug('trying camera position trial map: {}'.format(campath))

            if campath.exists():  # XXX: uses 1st found
                break

            log.debug('tracking path {} n/a - skipping'.format(tbase))
            campath = None

        if not campath:
            log.warning('tracking data not found for {} '.format(tpos))
            return

        tmap = tracking_ingest.TrackingIngest.load_campath(campath)

        if trial not in tmap:
            log.warning('nonexistant trial {}.. skipping'.format(trial))
            return

        repname, rep, rep_sub = (GlobusStorageLocation
                                 & {'globus_alias': globus_alias}).fetch(
                                     limit=1)[0]

        vmatch = '{}_{}_{}-*'.format(h2o, tpos, tmap[trial])
        vglob = list(vbase.glob(vmatch))

        if len(vglob) != 1:  # XXX: error instead of warning?
            log.warning('more than one video found: {}'.format(vglob))
            return

        vfile = vglob[0].name
        gfile = '{}/{}/{}/{}'.format(h2o, sdate_iso, 'video', vfile)  # subpath
        srcp = '{}:{}/{}'.format(lep, lep_sub, gfile)  # source path
        dstp = '{}:{}/{}'.format(rep, rep_sub, gfile)  # dest path

        gsm = self.get_gsm()
        gsm.activate_endpoint(lep)  # XXX: cache this / prevent duplicate RPC?
        gsm.activate_endpoint(rep)  # XXX: cache this / prevent duplicate RPC?

        log.info('transferring {} to {}'.format(srcp, dstp))
        if not gsm.cp(srcp, dstp):
            emsg = "couldn't transfer {} to {}".format(srcp, dstp)
            log.error(emsg)
            raise dj.DataJointError(emsg)

        self.insert1({**key, 'globus_alias': globus_alias,
                      'video_file_name': vfile})

    def retrieve(self):
        for key in self:
            self.retrieve1(key)

    def retrieve1(self, key):
        '''
        retrieve related files for a given key
        '''
        log.debug(key)

        # get remote file information
        linfo = (self * GlobusStorageLocation & key).fetch1()

        rep = linfo['globus_endpoint']
        rep_sub = linfo['globus_path']
        vfile = linfo['video_file_name']

        # get session related information needed for filenames/records
        sinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * ((tracking.TrackingDevice
                      * tracking.Tracking.proj()) & key)
                  * experiment.Session
                  * experiment.SessionTrial) & key).fetch1()

        h2o = sinfo['water_restriction_number']
        sdate_iso = sinfo['session_date'].isoformat()   # YYYY-MM-DD

        # get local endpoint information
        globus_alias = 'raw-video'
        le = GlobusStorageLocation.local_endpoint(globus_alias)
        lep, lep_sub = le['endpoint'], le['endpoint_subdir']

        # build source/destination paths & initiate transfer
        gfile = '{}/{}/{}/{}'.format(h2o, sdate_iso, 'video', vfile)

        srcp = '{}:{}/{}'.format(rep, rep_sub, gfile)  # source path
        dstp = '{}:{}/{}'.format(lep, lep_sub, gfile)  # dset path

        gsm = self.get_gsm()
        gsm.activate_endpoint(lep)  # XXX: cache this / prevent duplicate RPC?
        gsm.activate_endpoint(rep)  # XXX: cache this / prevent duplicate RPC?

        log.info('transferring {} to {}'.format(srcp, dstp))
        if not gsm.cp(dstp, srcp):
            emsg = "couldn't transfer {} to {}".format(srcp, dstp)
            log.error(emsg)
            raise dj.DataJointError(emsg)
