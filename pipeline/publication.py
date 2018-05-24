
import datajoint as dj


schema = dj.schema(dj.config['publication.database'])


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
        if 'globus_storage_locations' in dj.config:  # for local testing
            return dj.config['globus_storage_locations']

        return (('raw-ephys',
                 'petrel#mesoscaleactivityproject',
                 'publication/raw-ephys'))


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

    contents = [{
        'raw_ephys_filetype': 'ap-30kHz',
        'raw_ephys_suffix': '.imec.ap.bin',
        'raw_ephys_freq': 30.0,
        'raw_ephys_descr': 'ap channels @30kHz'
    }, {
        'raw_ephys_filetype': 'ap-30kHz-meta',
        'raw_ephys_suffix': '.imec.ap.meta',
        'raw_ephys_freq': None,
        'raw_ephys_descr': "recording metadata for 'ap-30kHz' files"
    }, {
        'raw_ephys_filetype': 'lf-2.5kHz',
        'raw_ephys_suffix': '.imec.lf.bin',
        'raw_ephys_freq': 2.5,
        'raw_ephys_descr': 'lf channels @2.5kHz'
    }, {
        'raw_ephys_filetype': 'lf-2.5kHz-meta',
        'raw_ephys_suffix': '.imec.lf.meta',
        'raw_ephys_freq': None,
        'raw_ephys_descr': "recording metadata for 'lf-2.5kHz' files"
    }]


@schema
class ArchivedRawEphysTrial(dj.Imported):
    """
    Table to track archive of raw ephys trial data.

    File naming convention:

    {water_restriction_number}_{session_date}_{electrode_group}_g0_t{trial}.{raw_ephys_suffix}
    """

    definition = """
    -> experiment.SessionTrial
    -> ephys.ElectrodeGroup
    -> GlobusStorageLocation
    """

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
