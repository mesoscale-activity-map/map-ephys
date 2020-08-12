
import logging
import os

from fnmatch import fnmatch
from textwrap import dedent

import datajoint as dj

from . import lab
from . import experiment
from . import ephys

from pipeline.globus import GlobusStorageManager
from . import get_schema_name

PUBLICATION_TRANSFER_TIMEOUT = 10000
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

        return (('raw-ephys',
                 '5b875fda-4185-11e8-bb52-0ac6873fc732',
                 '/4ElectrodeRig_Ephys'),  # TODO: updated/final path
                ('raw-video',
                 '5b875fda-4185-11e8-bb52-0ac6873fc732',
                 '/'))

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

    contents = zip(['ephys-raw',
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
        '''
        FileType values.

        A list of 3-tuples of file_type, file_glob, file_descr.

        Should be kept
        '''

        data = [('unknown',
                 '',  # deliberately non-matching pattern for manual tagging
                 '''
                 Unknown File Type
                 '''),
                ('ephys-raw-unknown',
                 '',  # deliberately non-matching pattern for manual tagging
                 '''
                 Unknown Raw-Ephys File Type
                 '''),
                ('ephys-raw-3a-ap-trial',
                 '*_g0_t[0-9]*.imec.ap.bin',
                 '''
                 3A Probe per-trial AP channels high pass filtered at
                 300Hz and sampled at 30kHz - recording file
                 '''),
                ('ephys-raw-3a-ap-trial-meta',
                 '*_g0_t[0-9]*.imec.ap.meta',
                 '''
                 3A Probe per-trial AP channels high pass
                 filtered at 300Hz and sampled at 30kHz - file metadata
                 '''),
                ('ephys-raw-3a-lf-trial',
                 '*_g0_t[0-9]*.imec.lf.bin',
                 '''
                 3A Probe per-trial AP channels low pass filtered at
                 300Hz and sampled at 2.5kHz - recording file
                 '''),
                ('ephys-raw-3a-lf-trial-meta',
                 '*_g0_t[0-9]*.imec.lf.meta',
                 '''
                 3A Probe per-trial AP channels low pass filtered at
                 300Hz and sampled at 2.5kHz - file metadata
                 '''),
                ('ephys-raw-3b-ap-trial',
                 '*_????????_g?_t[0-9]*.imec.ap.bin',
                 '''
                 3B Probe per-trial AP channels high pass filtered at
                 300Hz and sampled at 30kHz - recording file
                 '''),
                ('ephys-raw-3b-ap-trial-meta',
                 '*_????????_g?_t[0-9]*.imec.ap.meta',
                 '''
                 3B Probe per-trial AP channels high pass
                 filtered at 300Hz and sampled at 30kHz - file metadata
                 '''),
                ('ephys-raw-3b-lf-trial',
                 '*_????????_g?_t[0-9]*.imec.lf.bin',
                 '''
                 3B Probe per-trial AP channels low pass filtered at
                 300Hz and sampled at 2.5kHz - recording file
                 '''),
                ('ephys-raw-3b-lf-trial-meta',
                 '*_????????_g?_t[0-9]*.imec.lf.meta',
                 '''
                 3B Probe per-trial AP channels low pass filtered at
                 300Hz and sampled at 2.5kHz - file metadata
                 '''),
                ('ephys-raw-3b-ap-concat',
                 '*_????????_g?_tcat.imec.ap.bin',
                 '''
                 3B Probe concatenated AP channels high pass filtered at
                 300Hz and sampled at 30kHz - recording file
                 '''),
                ('ephys-raw-3b-ap-concat-meta',
                 '*_??????_g?_tcat.imec.ap.meta',
                 '''
                 3B Probe concatenated AP channels high pass
                 filtered at 300Hz and sampled at 30kHz - file metadata
                 '''),
                ('ephys-raw-3b-lf-concat',
                 '*_????????_g?_tcat.imec.lf.bin',
                 '''
                 3B Probe concatenated AP channels low pass filtered at
                 300Hz and sampled at 2.5kHz - recording file
                 '''),
                ('ephys-raw-3b-lf-concat-meta',
                 '*_????????_g?_tcat.imec.lf.meta',
                 '''
                 3B Probe concatenated AP channels low pass filtered at
                 300Hz and sampled at 2.5kHz - file metadata
                 '''),
                ('tracking-video-unknown',
                 '',  # deliberately non-matching pattern for manual tagging
                 '''
                 Unknown Tracking Video File Type
                 '''),
                ('tracking-video-trial',
                 '*_*_[0-9]*-*.[am][vp][i4]',
                 '''
                 Video Tracking per-trial file at 300fps
                 '''),
                ('tracking-video-map',
                 '*_????????_*.txt',
                 '''
                 Video Tracking file-to-trial mapping
                 ''')]

        return [[dedent(i).replace('\n', ' ').strip(' ') for i in r]
                for r in data]

    @classmethod
    def fnmatch(cls, fname, file_type_filter=''):
        '''
        Get file type match for a given file name.

        The optional keyword argument 'file_type_filter' will be used
        to restrict the subset of possible matched and unkown filetype names.

        For example:

          >>> FileType.fnmatch('myfilename', 'ephys')

        Will return the specific 'ephys*' FileType record if its file_glob
        matches 'myfilename', and if not, an 'unknown' FileType
        matching 'ephys' (e.g. 'ephys-unknown') if one and only one is present.

        If no file_glob matches any file type, and a single 'unknown'
        FileType cannot be found matching file_type_filter, the
        generic 'unknown' filetype data will be returned.
        '''
        self = cls()

        ftmap = {t['file_type']: t for t in (
            self & "file_type like '{}%%'".format(file_type_filter))}

        unknown, isknown = {}, {}  # unknown filetypes, known filetype matches.

        for k, v in ftmap.items():
            if 'unknown' in k and file_type_filter in k:
                unknown[k] = v  # a file_type_filter matching unknown file type
            if fnmatch(fname, v['file_glob']):
                isknown[k] = v  # a file_glob matching file name

        # return type match or unknown type
        return (list(isknown.values())[0] if len(isknown) == 1 else (
            list(unknown.values())[0] if len(unknown) == 1 else (
                ftmap['unknown'] if 'unknown' in ftmap else (
                    self & {'file_type': 'unknown'}).fetch1())))


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
    -> experiment.Session
    -> DataSet
    """

    key_source = experiment.Session & ephys.Unit

    gsm = None  # for GlobusStorageManager

    def get_gsm(self):
        log.debug('ArchivedRawEphysTrial.get_gsm()')
        if self.gsm is None:
            self.gsm = GlobusStorageManager()
            self.gsm.wait_timeout = PUBLICATION_TRANSFER_TIMEOUT

        return self.gsm

    @classmethod
    def discover(cls):
        """
        Discover files on globus and attempt to register them.
        """
        def build_session(self, key):
            log.debug('discover: build_session {}'.format(key))

            gsm = self.get_gsm()
            globus_alias = 'raw-ephys'

            ra, rep, rep_sub = (
                GlobusStorageLocation() & {'globus_alias': globus_alias}
            ).fetch1().values()

            sdate = key['session_date']
            sdate_mdy = sdate.strftime('%m%d%g')

            h2o = (lab.WaterRestriction & lab.Subject.proj() & key).fetch1()
            h2o = h2o['water_restriction_number']

            # check for multi-session/day
            msess = (experiment.Session
                     & {'subject_id': key['subject_id'],
                        'session_date': key['session_date']}).fetch(
                            as_dict=True, order_by='session')

            if len(msess) == 1:
                log.info('processing single session/day case')

                # session: <root>/<h2o>/catgt_<h2o>_<mdy>_g0/
                # probe: <root>/<h2o>/catgt_<h2o>_<mdy>_g0/<h2o>_<mdy>_imecN

                rpath = '/'.join([rep_sub, h2o,
                                  'catgt_{}_{}_g0'.format(h2o, sdate_mdy)])

                rep_tgt = '{}:{}'.format(rep, rpath)

                log.debug('.. rpath: {}'.format(rpath))

                if not gsm.ls(rep_tgt):
                    log.info('no globus data found for {} session {}'.format(
                        h2o, key['session']))
                    return None

                dskey = {'globus_alias': globus_alias,
                         'dataset_name': 'ephys-raw-{}-{}'.format(
                             h2o, key['session'])}

                dsrec = {**dskey, 'dataset_type': 'ephys-raw'}

                dsfiles = []

                for f in (f for f in gsm.fts(rep_tgt) if type(f[2]) == dict):

                    dirname, basename = f[1], f[2]['name']

                    ftype = FileType.fnmatch(basename, dsrec['dataset_type'])

                    dsfile = {
                        'file_subpath': '{}/{}'.format(
                            dirname.lstrip(rep_sub), basename),
                        'file_type': ftype['file_type']
                    }

                    log.debug('.. file: {}'.format(dsfile))

                    dsfiles.append({**dskey, **dsfile})

            elif len(msess) > 1:
                # if session not in list, ValueError, we have problems.
                # idx = [i['session'] for i in msess].index(key['session'])
                # see also: ingest/ephys.py _get_sess_dir
                #   ... undecidable for {'subject_id': 456772, 'session': 5}
                #   a: /4ElectrodeRig_Ephys/SC033/catgt_SC033_111219_g0
                #   b: /4ElectrodeRig_Ephys/SC033/catgt_SC033_111219_surface_g0
                #   either transfer apdata local mess, or manuallly register
                #     if b: need to have an explicit 'discover1' method
                #     to allow for manual registration of on petrel data
                log.warning('multi session/day case not yet handled')
                return None

            else:
                log.error('key -> multisession query problem. skipping')
                return None

            return dsrec, dsfiles

        def commit_session(self, key, data):
            log.info('commit_session: {}'.format(key))

            with dj.conn().transaction:

                DataSet.insert1(data[0])
                DataSet.PhysicalFile.insert(data[1])

                self.insert1({**key, **data[0]}, ignore_extra_fields=True,
                             allow_direct_insert=True)

        self = cls()
        keys = self.key_source - self

        log.info('attempting discovery for {} sessions'.format(len(keys)))

        for key in keys:

            log.info('.. inspecting {} {}'.format(
                key['subject_id'], key['session']))

            data = build_session(self, key)

            if data:
                commit_session(self, key, data)

    def make(self, key):
        """
        discover files in local endpoint and transfer/register
        """
        def build_session(self, key):

            log.debug('build_session: {} '.format(key))

            # Get session related information needed for filenames/records
            sinfo = (lab.WaterRestriction
                     * lab.Subject.proj()
                     * experiment.Session() & key).fetch1()

            sdate = sinfo['session_date']
            sdate_mdy = sdate.strftime('%m%d%g')

            h2o = (lab.WaterRestriction & lab.Subject.proj() & key).fetch1()
            h2o = h2o['water_restriction_number']

            globus_alias = 'raw-ephys'
            le = GlobusStorageLocation.local_endpoint(globus_alias)
            lep, lep_sub, lep_dir = (le['endpoint'],
                                     le['endpoint_subdir'],
                                     le['endpoint_path'])

            log.debug('local_endpoint: {}:{} -> {}'.format(
                lep, lep_sub, lep_dir))

            # check for multi-session/day
            msess = (experiment.Session
                     & {'subject_id': sinfo['subject_id'],
                        'session_date': sinfo['session_date']}).fetch(
                            as_dict=True, order_by='session')

            if len(msess) == 1:
                log.info('processing single session/day case')

                # session: <root>/<h2o>/catgt_<h2o>_<mdy>_g0/
                # probe: <root>/<h2o>/catgt_<h2o>_<mdy>_g0/<h2o>_<mdy>_imecN

                lpath = os.path.join(lep_dir, h2o, 'catgt_{}_{}_g0'.format(
                    h2o, sdate_mdy))

                if not os.path.exists(lpath):
                    log.warning('session directory {} not found'.format(
                        lpath))
                    return None

                dskey = {'globus_alias': globus_alias,
                         'dataset_name': 'ephys-raw-{}-{}'.format(
                             h2o, sinfo['session'])}

                dsrec = {**dskey, 'dataset_type': 'ephys-raw'}

                dsfiles = []

                for cwd, dirs, files in os.walk(lpath):
                    log.debug('.. entering directory: {}'.format(cwd))

                    for f in files:

                        fname = os.path.join(cwd, f)
                        ftype = FileType.fnmatch(f, dsrec['dataset_type'])

                        dsfile = {
                            'file_subpath': os.path.relpath(fname, lep_dir),
                            'file_type': ftype['file_type']
                        }

                        log.debug('.... file: {}'.format(dsfile))

                        dsfiles.append({**dskey, **dsfile})

            elif len(msess) > 1:
                log.info('multi session/day case not yet handled')
                return None
            else:
                log.error('key -> multisession query problem. skipping')
                return None

            return dsrec, dsfiles

        def transfer_session(self, key, data):

            log.debug('transfer_session: {} '.format(key))

            gsm = self.get_gsm()
            globus_alias = 'raw-ephys'

            le = GlobusStorageLocation.local_endpoint(globus_alias)
            lep, lep_sub, _ = (le['endpoint'],
                               le['endpoint_subdir'],
                               le['endpoint_path'])

            ra, rep, rep_sub = (
                GlobusStorageLocation() & {'globus_alias': globus_alias}
            ).fetch1().values()

            gsm.activate_endpoint(lep)  # XXX: cache / prevent duplicate RPC?
            gsm.activate_endpoint(rep)  # XXX: cache / prevent duplicate RPC?

            for f in data[1]:
                fsp = f['file_subpath']
                srcp = '{}:{}/{}'.format(lep, lep_sub, fsp)
                dstp = '{}:{}/{}'.format(rep, rep_sub, fsp)

                log.info('transferring {} to {}'.format(srcp, dstp))

                # XXX: check if exists 1st?
                if not gsm.cp(srcp, dstp):
                    emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                    log.error(emsg)
                    raise dj.DataJointError(emsg)

        def commit_session(self, key, data):

            log.debug('commit_session: {}'.format(key))

            DataSet.insert1(data[0])
            DataSet.PhysicalFile.insert(data[1])

            self.insert1({**key, **data[0]}, ignore_extra_fields=True,
                         allow_direct_insert=True)

        # main():

        log.debug('make: {}'.format(key))

        data = build_session(self, key)

        if data:
            transfer_session(self, key, data)
            commit_session(self, key, data)

    @classmethod
    def retrieve(cls):
        self = cls()
        for key in self:
            self.retrieve1(key)

    @classmethod
    def retrieve1(cls, key):
        """
        retrieve related files for a given key
        """
        self = cls()

        log.info(str(key))

        lep = GlobusStorageLocation().local_endpoint(key['globus_alias'])
        lep, lep_sub, lep_dir = (
            lep[k] for k in ('endpoint', 'endpoint_subdir', 'endpoint_path'))

        repname, rep, rep_sub = (GlobusStorageLocation() & key).fetch()[0]

        log.info('local_endpoint: {}:{} -> {}'.format(lep, lep_sub, lep_dir))
        log.info('remote_endpoint: {}:{}'.format(rep, rep_sub))

        # get dataset file information
        finfo = (DataSet * DataSet.PhysicalFile
                 & (self & key).proj()).fetch(as_dict=True)

        gsm = self.get_gsm()
        gsm.activate_endpoint(lep)  # XXX: cache this / prevent duplicate RPC?
        gsm.activate_endpoint(rep)  # XXX: cache this / prevent duplicate RPC?

        for f in finfo:
            srcp = '{}:/{}/{}'.format(rep, rep_sub, f['file_subpath'])
            dstp = '{}:/{}/{}'.format(lep, lep_sub, f['file_subpath'])

            log.info('transferring {} to {}'.format(srcp, dstp))

            # XXX: check if exists 1st? (manually or via API copy-checksum)
            if not gsm.cp(srcp, dstp):
                emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                log.error(emsg)
                raise dj.DataJointError(emsg)


@schema
class ArchivedTrackingVideo(dj.Imported):
    """
    ArchivedTrackingVideo storage

    Note: video_file_name tracked here as trial->file map is non-deterministic

    Directory locations of the form::

      <Water restriction number>\<Session Date in MMDDYYYY>\video

    with file naming convention of the form:

    {Water restriction number}_{camera-position-string}_NNN-NNNN.avi

    Where 'NNN' is determined from the 'tracking map file' which maps
    trials to videos as outlined in tracking.py

    """
    definition = """
    -> experiment.Session
    -> DataSet
    """

    key_source = experiment.Session

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
            self.gsm.wait_timeout = PUBLICATION_TRANSFER_TIMEOUT

        return self.gsm

    @classmethod
    def discover(cls):
        """
        discover files on globus and attempt to register them

        video:trial mapping information is retrieved from TrackingIngest table.
        """
        def build_session(self, key):
            '''
            TODO: xref w/r/t real globus layout
            working with: /SC026/08082019/video/SC026_side_735-NNNN.avi
            '''

            log.info('build_session: {}'.format(key))

            gsm = self.get_gsm()
            globus_alias = 'raw-video'

            ra, rep, rep_sub = (
                GlobusStorageLocation() & {'globus_alias': globus_alias}
            ).fetch1().values()

            sdate = key['session_date']
            sdate_mdy = sdate.strftime('%m%d%G')

            h2o = (lab.WaterRestriction & lab.Subject.proj() & key).fetch1()
            h2o = h2o['water_restriction_number']

            # check for multi-session/day
            msess = (experiment.Session
                     & {'subject_id': key['subject_id'],
                        'session_date': key['session_date']}).fetch(
                            as_dict=True, order_by='session')

            if len(msess) == 1:
                log.info('processing single session/day case')

                rpath = '/'.join((rep_sub, h2o, sdate_mdy, 'video'))

                rep_tgt = '{}:{}'.format(rep, rpath)

                log.debug('.. rpath: {}'.format(rpath))

                if not gsm.ls(rep_tgt):
                    log.info('no globus data found for {} session {}'.format(
                        h2o, key['session']))
                    return None

                # traverse session directory, building fileset

                dskey = {'globus_alias': globus_alias,
                         'dataset_name': 'tracking-video-{}-{}'.format(
                             h2o, key['session'])}

                dsrec = {**dskey, 'dataset_type': 'tracking-video'}

                dsfiles = []

                for f in (f for f in gsm.fts(rep_tgt) if type(f[2]) == dict):

                    dirname, basename = f[1], f[2]['name']

                    ftype = FileType.fnmatch(basename, dsrec['dataset_type'])

                    dsfile = {
                        'file_subpath': '{}/{}'.format(
                            dirname.lstrip(rep_sub), basename),
                        'file_type': ftype['file_type']
                    }

                    log.debug('.. file: {}'.format(dsfile))

                    dsfiles.append({**dskey, **dsfile})

            elif len(msess) > 1:
                log.warning('multi session/day case not yet handled')
                return None

            else:
                log.error('key -> multisession query problem. skipping')
                return None

            return dsrec, dsfiles

        def commit_session(self, key, data):
            log.info('commit_session: {}'.format(key))

            with dj.conn().transaction:

                DataSet.insert1(data[0])
                DataSet.PhysicalFile.insert(data[1])

                self.insert1({**key, **data[0]}, ignore_extra_fields=True,
                             allow_direct_insert=True)

        self = cls()
        keys = self.key_source - self

        log.info('attempting discovery for {} sessions'.format(len(keys)))

        for key in keys:

            log.info('.. inspecting {} {}'.format(
                key['subject_id'], key['session']))

            data = build_session(self, key)

            if data:
                commit_session(self, key, data)

    def make(self, key):
        """
        discover files in local endpoint and transfer/register
        """
        def build_session(self, key):

            log.debug('build_session: {}'.format(key))

            # Get session related information needed for filenames/records
            sinfo = (lab.WaterRestriction
                     * lab.Subject.proj()
                     * experiment.Session() & key).fetch1()

            sdate = sinfo['session_date']
            sdate_mdy = sdate.strftime('%m%d%Y')

            h2o = (lab.WaterRestriction & lab.Subject.proj() & key).fetch1()
            h2o = h2o['water_restriction_number']

            globus_alias = 'raw-video'
            le = GlobusStorageLocation.local_endpoint(globus_alias)
            lep, lep_sub, lep_dir = (le['endpoint'],
                                     le['endpoint_subdir'],
                                     le['endpoint_path'])

            log.debug('local_endpoint: {}:{} -> {}'.format(
                lep, lep_sub, lep_dir))

            # check for multi-session/day
            msess = (experiment.Session
                     & {'subject_id': sinfo['subject_id'],
                        'session_date': sinfo['session_date']}).fetch(
                            as_dict=True, order_by='session')

            if len(msess) == 1:
                log.info('processing single session/day case')

                # <root>/<h2o>/MMDDYYYY/video/<h2o>_<campos>_NNN-NNN.{avi}

                lpath = os.path.join(lep_dir, h2o, sdate_mdy, 'video')

                if not os.path.exists(lpath):
                    log.warning('session directory {} not found'.format(
                        lpath))
                    return None

                dskey = {'globus_alias': globus_alias,
                         'dataset_name': 'tracking-video-{}-{}'.format(
                             h2o, sinfo['session'])}

                dsrec = {**dskey, 'dataset_type': 'tracking-video'}

                dsfiles = []

                for cwd, dirs, files in os.walk(lpath):
                    log.debug('.. entering directory: {}'.format(cwd))

                    for f in files:

                        fname = os.path.join(cwd, f)
                        ftype = FileType.fnmatch(f, dsrec['dataset_type'])

                        dsfile = {
                            'file_subpath': os.path.relpath(fname, lep_dir),
                            'file_type': ftype['file_type']
                        }

                        log.debug('.... file: {}'.format(dsfile))

                        dsfiles.append({**dskey, **dsfile})

            elif len(msess) > 1:
                log.info('multi session/day case not yet handled')
                return None
            else:
                log.error('key -> multisession query problem. skipping')
                return None

            return dsrec, dsfiles

        def transfer_session(self, key, data):

            log.debug('transfer_session: {} '.format(key))

            gsm = self.get_gsm()
            globus_alias = 'raw-video'

            le = GlobusStorageLocation.local_endpoint(globus_alias)
            lep, lep_sub, _ = (le['endpoint'],
                               le['endpoint_subdir'],
                               le['endpoint_path'])

            ra, rep, rep_sub = (
                GlobusStorageLocation() & {'globus_alias': globus_alias}
            ).fetch1().values()

            gsm.activate_endpoint(lep)  # XXX: cache / prevent duplicate RPC?
            gsm.activate_endpoint(rep)  # XXX: cache / prevent duplicate RPC?

            for f in data[1]:
                fsp = f['file_subpath']
                srcp = '{}:{}/{}'.format(lep, lep_sub, fsp)
                dstp = '{}:{}/{}'.format(rep, rep_sub, fsp)

                log.info('transferring {} to {}'.format(srcp, dstp))

                # XXX: check if exists 1st?
                if not gsm.cp(srcp, dstp):
                    emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                    log.error(emsg)
                    raise dj.DataJointError(emsg)

        def commit_session(self, key, data):

            log.debug('commit_session: {}'.format(key))

            DataSet.insert1(data[0])
            DataSet.PhysicalFile.insert(data[1])

            self.insert1({**key, **data[0]}, ignore_extra_fields=True,
                         allow_direct_insert=True)

        # main():

        log.debug('make: {}'.format(key))

        data = build_session(self, key)

        if data:
            transfer_session(self, key, data)
            commit_session(self, key, data)

    @classmethod
    def retrieve(cls):
        self = cls()
        for key in self:
            self.retrieve1(key)

    @classmethod
    def retrieve1(cls, key):
        """
        retrieve related files for a given key
        """
        self = cls()

        log.info(str(key))

        lep = GlobusStorageLocation().local_endpoint(key['globus_alias'])
        lep, lep_sub, lep_dir = (
            lep[k] for k in ('endpoint', 'endpoint_subdir', 'endpoint_path'))

        repname, rep, rep_sub = (GlobusStorageLocation() & key).fetch()[0]

        log.info('local_endpoint: {}:{} -> {}'.format(lep, lep_sub, lep_dir))
        log.info('remote_endpoint: {}:{}'.format(rep, rep_sub))

        # get dataset file information
        finfo = (DataSet * DataSet.PhysicalFile
                 & (self & key).proj()).fetch(as_dict=True)

        gsm = self.get_gsm()
        gsm.activate_endpoint(lep)  # XXX: cache this / prevent duplicate RPC?
        gsm.activate_endpoint(rep)  # XXX: cache this / prevent duplicate RPC?

        for f in finfo:
            srcp = '{}:/{}/{}'.format(rep, rep_sub, f['file_subpath'])
            dstp = '{}:/{}/{}'.format(lep, lep_sub, f['file_subpath'])

            log.info('transferring {} to {}'.format(srcp, dstp))

            # XXX: check if exists 1st? (manually or via API copy-checksum)
            if not gsm.cp(srcp, dstp):
                emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                log.error(emsg)
                raise dj.DataJointError(emsg)
