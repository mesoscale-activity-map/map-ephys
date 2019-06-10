
import os
import logging

import datajoint as dj

from . import lab, experiment, ephys
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
        if 'globus.storage_locations' in dj.config['custom']:  # for local testing
            return dj.config['custom']['globus.storage_locations']

        return (('raw-ephys',
                 '5b875fda-4185-11e8-bb52-0ac6873fc732',
                 'publication/raw-ephys'),)

    @property
    def local_endpoint(self):
        if 'globus.local_endpoint' in dj.config:
            return (dj.config['custom']['globus.local_endpoint'],
                    dj.config['custom']['globus.local_endpoint_subdir'],
                    dj.config['custom']['globus.local_endpoint_local_path'])
        else:
            raise dj.DataJointError("globus_local_endpoint not configured")


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

        repname, rep, rep_sub = (GlobusStorageLocation() & key).fetch()[0]

        gsm = self.get_gsm()
        gsm.activate_endpoint(lep)  # XXX: cache this / prevent duplicate RPC?
        gsm.activate_endpoint(rep)  # XXX: cache this / prevent duplicate RPC?

        if not ArchivedRawEphysTrial & key:
            log.info('ArchivedRawEphysTrial.insert1()')
            ArchivedRawEphysTrial.insert1(key)

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

    def retrieve(self):
        for key in self:
            self.retrieve1(key)

    def retrieve1(self, key):
        '''
        retrieve related files for a given key
        '''

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
