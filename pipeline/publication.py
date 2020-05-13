
import logging
import pathlib
import re
import os

from fnmatch import fnmatch
from textwrap import dedent
from collections import defaultdict

import datajoint as dj

from . import lab
from . import experiment
from . import ephys
from . import tracking
from .ingest.tracking import TrackingIngest


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

    # contents = zip(['ephys-raw-trialized',
    #                 'ephys-raw-continuous',
    #                 'ephys-sorted',
    #                 'tracking-video'])
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
            self & "file_type like '{}%%'".format(file_type_filter)) }

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

    # FIXME: full key source
    # key_source = experiment.Session & ephys.Unit
    key_source = experiment.Session & ephys.Unit & {'subject_id': 440956, 'session': 1}

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

            h2o_num = h2o['water_restriction_number']

            # session: <root>/<h2o>/catgt_<h2o>_<mdy>_g0/
            # probe: <root>/<h2o>/catgt_<h2o>_<mdy>_g0/<h2o>_<mdy>_imecN

            rpath = '/'.join([rep_sub, h2o_num,
                              'catgt_{}_{}_g0'.format(h2o_num, sdate_mdy)])

            rep_tgt = '{}:{}'.format(rep, rpath)

            if not gsm.ls(rep_tgt):
                log.info('no globus data found for {} session {}'.format(
                    h2o_num, key['session']))
                return None

            dskey = {'globus_alias': globus_alias,
                     'dataset_name': '{}-{}'.format(h2o_num, key['session'])}

            dsrec = {**dskey, 'dataset_type': 'ephys-raw'}

            dsfiles = []

            for f in (f for f in gsm.fts(rep_tgt) if type(f[2]) == dict):

                dirname, basename = f[1], f[2]['name']

                ftype = FileType.fnmatch(basename, dsrec['dataset_type'])

                dsfile = {'file_subpath': '{}/{}'.format(dirname, basename),
                          'file_type': ftype['file_type']}

                log.debug('.. file: {}'.format(dsfile))

                dsfiles.append({**dskey, **dsfile})

            return dsrec, dsfiles

        def commit_session(self, key, data):
            log.debug('discover: commit_session {}'.format(key))

            DataSet.insert1(data[0])
            DataSet.PhysicalFile.insert(data[1])

            self.insert1({**key, **data[0]})

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
        raise NotImplementedError('need adjustment to new layout')

        log.debug(key)
        globus_alias = 'raw-ephys'
        le = GlobusStorageLocation.local_endpoint(globus_alias)
        lep, lep_sub, lep_dir = (le['endpoint'],
                                 le['endpoint_subdir'],
                                 le['endpoint_path'])

        re, rep, rep_sub = (GlobusStorageLocation()
                            & {'globus_alias': globus_alias}).fetch1().values()

        log.info('local_endpoint: {}:{} -> {}'.format(lep, lep_sub, lep_dir))

        # Get session related information needed for filenames/records

        sinfo = (lab.WaterRestriction
                 * lab.Subject.proj()
                 * experiment.Session() & key).fetch1()

        tinfo = ((lab.WaterRestriction
                  * lab.Subject.proj()
                  * experiment.Session()
                  * experiment.SessionTrial) & key).fetch()

        h2o = sinfo['water_restriction_number']
        sdate = sinfo['session_date']

        subdir = pathlib.Path(h2o, str(sdate).replace('-', ''))  # + probeno
        lep_subdir = pathlib.Path(lep_dir, subdir)

        probechoice = [str(i) for i in range(1, 10)]  # XXX: hardcoded

        file_globs = {i['file_glob']: i['file_type']
                      for i in FileType & "file_type like 'ephys%%'"}

        # Process each probe folder

        for lep_probedir in lep_subdir.glob('*'):
            lep_probe = str(lep_probedir.relative_to(lep_subdir))
            if lep_probe not in probechoice:
                log.info('skipping lep_probedir: {} - unexpected name'.format(
                    lep_probedir))
                continue

            lep_matchfiles = {}
            lep_probefiles = lep_probedir.glob('*.*')

            for pf in lep_probefiles:
                pfbase = pf.relative_to(lep_probedir)
                pfmatch = {k: pfbase.match(k) for k in file_globs}
                if any(pfmatch.values()):
                    log.debug('found valid file: {}'.format(pf))
                    lep_matchfiles[pf] = tuple(k for k in pfmatch if pfmatch[k])
                else:
                    log.debug('skipping non-match file: {}'.format(pf))
                    continue

            # Build/Validate file records

            if not all([len(lep_matchfiles[i]) == 1 for i in lep_matchfiles]):
                # TODO: handle trial + concatenated match case...
                log.warning('files matched multiple types'.format(
                    lep_matchfiles))
                continue

            type_to_file = {file_globs[lep_matchfiles[mf][0]]: mf
                            for mf in lep_matchfiles}

            ds_key, ds_name, ds_files, ds_trials = (
                None, None, None, [], [])

            if all(['trial' in t for t in type_to_file]):
                dataset_type = 'ephys-raw-trialized'

                ds_name = '{}_{}_{}'.format(h2o, sdate.isoformat(),
                                            dataset_type)

                ds_key = {'dataset_name': ds_name,
                          'globus_storage_location': globus_alias}

                for t in type_to_file:
                    fsp = type_to_file[t].relative_to(lep_dir)
                    dsf = {**ds_key, 'file_subpath': str(fsp)}

                    # e.g : 'tw34_g0_t0.imec.ap.meta' -> *_t(trial).*
                    trial = int(fsp.name.split('_t')[1].split('.')[0])

                    if trial not in tinfo['trial']:
                        log.warning('unknown trial file: {}. skipping'.format(
                            dsf))
                        continue

                    ds_trials.append({**dsf, 'trial': trial})
                    ds_files.append({**dsf, 'file_type': t})

            elif all(['concat' in t for t in type_to_file]):
                dataset_type = 'ephys-raw-continuous'

                ds_name = '{}_{}_{}'.format(h2o, sdate.isoformat(),
                                            dataset_type)

                ds_key = {'dataset_name': ds_name,
                          'globus_storage_location': globus_alias}

                for t in type_to_file:
                    fsp = type_to_file[t].relative_to(lep_dir)
                    ds_files.append({**ds_key,
                                     'file_subpath': str(fsp),
                                     'file_type': t})

            else:
                log.warning("couldn't determine dataset type for {}".format(
                    lep_probedir))
                continue

            # Transfer Files

            gsm = self.get_gsm()
            gsm.activate_endpoint(lep)  # XXX: cache / prevent duplicate RPC?
            gsm.activate_endpoint(rep)  # XXX: cache / prevent duplicate RPC?

            DataSet.insert1({**ds_key, 'dataset_type': dataset_type},
                            allow_direct_insert=True)

            for f in ds_files:
                fsp = ds_files[f]['file_subpath']
                srcp = '{}:{}/{}'.format(lep, lep_sub, fsp)
                dstp = '{}:{}/{}'.format(rep, rep_sub, fsp)

                log.info('transferring {} to {}'.format(srcp, dstp))

                # XXX: check if exists 1st?
                if not gsm.cp(srcp, dstp):
                    emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                    log.error(emsg)
                    raise dj.DataJointError(emsg)

                DataSet.PhysicalFile.insert1({**ds_key, **ds_files[f]},
                                             allow_direct_insert=True)

            # Add Records
            ArchivedSession.insert1(
                {**key, 'globus_storage_location': globus_alias},
                skip_duplicates=True, allow_direct_insert=True)

            ArchivedRawEphys.insert1(
                {**key, **ds_key, 'probe_folder': int(str(lep_probe))},
                allow_direct_insert=True)

            if dataset_type == 'ephys-raw-trialized':
                ArchivedRawEphys.ArchivedTrials.insert(
                    [{**key, **t} for t in ds_trials],
                    allow_direct_insert=True)

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

        raise NotImplementedError('retrieve not yet implemented')

        # Old / to be updated:

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
        raise NotImplementedError('ArchivedSortedEphys.make to be implemented')


@schema
class ArchivedTrackingVideo(dj.Imported):
    '''
    ArchivedTrackingVideo storage

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
    -> experiment.Session
    -> DataSet
    """

    key_source = tracking.TrackingDevice * experiment.Session

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
        """
        self = cls()

        globus_alias = 'raw-video'

        le = GlobusStorageLocation.local_endpoint(globus_alias)
        lep, lep_sub, lep_dir = (le['endpoint'],
                                 le['endpoint_subdir'],
                                 le['endpoint_path'])

        ra, rep, rep_sub = (GlobusStorageLocation()
                            & {'globus_alias': globus_alias}).fetch1().values()

        smap = {'{}/{}'.format(s['water_restriction_number'],
                               s['session_date']).replace('-', ''): s
                for s in (experiment.Session()
                          * (lab.WaterRestriction() * lab.Subject.proj()))}

        tpos_dev = {s['tracking_position']: s['tracking_device']
                    for s in tracking.TrackingDevice()}  # position:device

        ftmap = {t['file_type']: t for t
                 in (FileType() & "file_type like 'tracking%%'")}

        skey = None
        sskip = set()
        sfiles = []  # {file_subpath:, trial:, file_type:,}

        gsm = self.get_gsm()
        gsm.activate_endpoint(lep)
        gsm.activate_endpoint(rep)

        def commit(skey, sfiles):
            log.info('commit. skey: {}'.format(skey))

            if not sfiles:
                log.info('commit skipping {}. no files in set'.format(skey))

            # log.debug('sfiles: {}'.format(sfiles))

            h2o, sdate, ftypes = set(), set(), set()

            dftmap = {}  # device:file:trial via load_campath mapping files
            dvfmap = defaultdict(lambda: defaultdict(list))  # device:video:file
            dtfmap = defaultdict(lambda: defaultdict(list))  # device:trial:file

            for s in sfiles:

                if s['file_type'] == 'tracking-video-trial':
                    dvfmap[s['position']][s['video']].append(s)
                    h2o.add(s['water_restriction_number'])
                    sdate.add(s['session_date'])
                    ftypes.add(s['file_type'])

                if s['file_type'] == 'tracking-video-map':
                    # xfer & load camera:trial map ex: dl55_20190108_side.txtb
                    fsp = s['file_subpath']
                    lsp = '/tmp/' + s['file_subpath'].split('/')[-1]

                    srcp = '{}:{}/{}'.format(rep, rep_sub, fsp)
                    dstp = '{}:{}/{}'.format(lep, lep_sub, lsp)

                    log.info('transferring {} to {}'.format(srcp, dstp))

                    if not gsm.cp(srcp, dstp):  # XXX: check if exists 1st?
                        emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                        log.error(emsg)
                        raise dj.DataJointError(emsg)

                    lfname = lep_dir + lsp  # local filesysem copy location

                    dftmap[s['position']] = TrackingIngest.load_campath(lfname)

            if len(h2o) != 1 or len(sdate) != 1:
                log.info('skipping. bad h2o {} or session date {}'.format(
                    h2o, sdate))
                return

            h2o, sdate = next(iter(h2o)), next(iter(sdate))

            for d in dvfmap:
                if d in dftmap:  # remap video no -> trial
                    dtfmap[d] = {dftmap[d][v]:
                                 dict(dvfmap[d][v], trial=dftmap[d][v])
                                 for v in dvfmap[d]}
                else:  # assign video no -> trial
                    dtfmap[d] = {k: dict(v, trial=v['video'])
                                 for k, v in dvfmap[d].items()}

            # DataSet
            ds_type = 'tracking-video'
            ds_name = '{}_{}_{}'.format(h2o, sdate, ds_type)
            ds_key = {'dataset_name': ds_name, 'globus_alias': globus_alias}

            if (DataSet & ds_key):
                log.info('DataSet: {} already exists. Skipping.'.format(
                    ds_key))
                return

            DataSet.insert1({**ds_key, 'dataset_type': ds_type},
                            allow_direct_insert=True)

            # ArchivedSession
            as_key = {k: v for k, v in smap[skey].items()
                      if k in ArchivedSession.primary_key}

            ArchivedSession.insert1(
                {**as_key, 'globus_alias': globus_alias},
                allow_direct_insert=True,
                skip_duplicates=True)

            for d in dtfmap:

                # ArchivedTrackingVideo
                atv_key = {**as_key, **ds_key, 'tracking_device': tpos_dev[d]}

                ArchivedTrackingVideo.insert1(
                    atv_key, allow_direct_insert=True)

                for t in dtfmap[d]:
                    for f in dtfmap[d][t]:

                        DataSet.PhysicalFile.insert1(
                            {**ds_key, **f}, allow_direct_insert=True,
                            ignore_extra_fields=True)

                        ArchivedTrackingVideo.TrialVideo.insert1(
                            {**atv_key, **ds_key,
                             'trial': t,
                             'file_subpath': f['file_subpath']},
                            allow_direct_insert=True)

            # end commit()

        for ep, dirname, node in gsm.fts('{}:{}'.format(rep, rep_sub)):

            vdir = re.match('([a-z]+[0-9]+)/([0-9]{8})/video', dirname)

            if not vdir or node['DATA_TYPE'] != 'file':
                continue

            h2o, sdate = vdir[1], vdir[2]

            skey_i = '{}/{}'.format(h2o, sdate)

            if skey_i != skey:
                if skey and skey in smap:
                    with dj.conn().transaction:
                        try:
                            commit(skey, sfiles)
                        except Exception as e:
                            log.error(
                                'Exception {} committing {}. files: {}'.format(
                                    repr(e), skey, sfiles))

                skey, sfiles = skey_i, []

            if skey not in smap:
                if skey not in sskip:
                    log.debug('session {} not known. skipping'.format(skey))
                    sskip.add(skey)

                continue

            fname = node['name']

            log.debug('checking {}/{}'.format(dirname, fname))

            if '.' not in fname:
                log.debug('skipping {} - no dot in fname'.format(fname))
                continue

            froot, fext = fname.split('.', 1)
            ftype = {g['file_type']: g for g in ftmap.values()
                     if fnmatch(fname, g['file_glob'])}

            if len(ftype) != 1:
                log.debug('skipping {} - incorrect type matches: {}'.format(
                    fname, ftype))
                continue

            ftype = next(iter(ftype.values()))['file_type']
            log.debug('processing as {}'.format(ftype))

            file_subpath = '{}/{}'.format(dirname, fname)

            if ftype == 'tracking-video-map':
                # e.g. dl55_20190108_side.txt
                h2o_f, fdate, pos = froot.split('_')
                sfiles.append({'water_restriction_number': h2o,
                               'session_date': '{}-{}-{}'.format(
                                   sdate[:4], sdate[4:6], sdate[6:]),
                               'position': pos,
                               'file_subpath': file_subpath,
                               'file_type': ftype})
            else:  # tracking-video-map
                # e.g. dl41_side_998-0000.avi or dl41_side_998-0000_00.avi
                h2o_f, pos, video = froot.replace('-', '_').split('_')[:3]
                sfiles.append({'water_restriction_number': h2o,
                               'session_date': '{}-{}-{}'.format(
                                   sdate[:4], sdate[4:6], sdate[6:]),
                               'position': pos,
                               'video': int(video),
                               'file_subpath': file_subpath,
                               'file_type': ftype})

    def make(self, key):
        """
        discover files in local endpoint and transfer/register
        """
        log.info('ArchivedVideoFile.make(): {}'.format(key))

        # {'tracking_device': 'Camera 0', 'subject_id': 432572, 'session': 1}

        globus_alias = 'raw-video'
        le = GlobusStorageLocation.local_endpoint(globus_alias)
        lep, lep_sub, lep_dir = (le['endpoint'],
                                 le['endpoint_subdir'],
                                 le['endpoint_path'])

        re = (GlobusStorageLocation & {'globus_alias': globus_alias}).fetch1()
        rep, rep_sub = re['globus_endpoint'], re['globus_path']

        log.info('local_endpoint: {}:{} -> {}'.format(lep, lep_sub, lep_dir))
        log.info('remote_endpoint: {}:{}'.format(rep, rep_sub))

        h2o = (lab.WaterRestriction & key).fetch1('water_restriction_number')

        session = (experiment.Session & key).fetch1()
        sdate = session['session_date']
        sdate_sml = "{}{:02d}{:02d}".format(sdate.year, sdate.month, sdate.day)

        dev = (tracking.TrackingDevice & key).fetch1()

        trls = (experiment.SessionTrial & key).fetch(
            order_by='trial', as_dict=True)

        tracking_ingest = self.get_ingest()

        tdev = dev['tracking_device']  # NOQA: notused
        tpos = dev['tracking_position']

        camtrial = '{}_{}_{}.txt'.format(h2o, sdate_sml, tpos)
        vbase = pathlib.Path(lep_dir, h2o, sdate_sml, 'video')
        campath = vbase / camtrial

        if not campath.exists():  # XXX: uses 1st found
            log.warning('trial map {} n/a! skipping.'.format(campath))
            return

        log.info('loading trial map: {}'.format(campath))
        vmap = {v: k for k, v in
                tracking_ingest.TrackingIngest.load_campath(campath).items()}
        log.debug('loaded video map: {}'.format(vmap))

        # add ArchivedSession

        as_key = {k: v for k, v in key.items()
                  if k in experiment.Session.primary_key}
        as_rec = {**as_key, 'globus_alias': globus_alias}

        ArchivedSession.insert1(as_rec, allow_direct_insert=True,
                                skip_duplicates=True)

        # add DataSet

        ds_type = 'tracking-video'
        ds_name = '{}_{}_{}_{}'.format(h2o, sdate.isoformat(), ds_type, tpos)
        ds_key = {'globus_alias': globus_alias, 'dataset_name': ds_name}
        ds_rec = {**ds_key, 'dataset_type': ds_type}

        DataSet.insert1(ds_rec, allow_direct_insert=True)

        # add ArchivedVideoTracking

        vt_key = {**as_key, 'tracking_device': tdev}
        vt_rec = {**vt_key, 'globus_alias': globus_alias,
                  'dataset_name': ds_name}

        self.insert1(vt_rec)

        filetype = 'tracking-video-trial'

        for t in trls:
            trial = t['trial']
            log.info('.. tracking trial {} ({})'.format(trial, t))

            if t['trial'] not in vmap:
                log.warning('trial {} not in video map. skipping!'.format(t))
                continue

            vmatch = '{}_{}_{}-*'.format(h2o, tpos, vmap[trial])
            log.debug('vbase: {}, vmatch: {}'.format(vbase, vmatch))
            vglob = list(vbase.glob(vmatch))

            if len(vglob) != 1:
                emsg = 'incorrect videos found in {}: {}'.format(vbase, vglob)
                log.warning(emsg)
                raise dj.DataJointError(emsg)

            vfile = vglob[0].name
            gfile = '{}/{}/{}/{}'.format(
                h2o, sdate_sml, 'video', vfile)  # subpath

            srcp = '{}:{}/{}'.format(lep, lep_sub, gfile)  # source path
            dstp = '{}:{}/{}'.format(rep, rep_sub, gfile)  # dest path

            gsm = self.get_gsm()
            gsm.activate_endpoint(lep)  # XXX: cache / prevent duplicate RPC?
            gsm.activate_endpoint(rep)  # XXX: cache / prevent duplicate RPC?

            log.info('transferring {} to {}'.format(srcp, dstp))

            if not gsm.cp(srcp, dstp):
                emsg = "couldn't transfer {} to {}".format(srcp, dstp)
                log.error(emsg)
                raise dj.DataJointError(emsg)

            pf_key = {**ds_key, 'file_subpath': vfile}
            pf_rec = {**pf_key, 'file_type': filetype}

            DataSet.PhysicalFile.insert1({**pf_rec}, allow_direct_insert=True)

            trk_key = {k: v for k, v in {**key, 'trial': trial}.items()
                       if k in experiment.SessionTrial.primary_key}

            tv_rec = {**vt_key, **trk_key, **pf_key}
            self.TrialVideo.insert1({**tv_rec})
