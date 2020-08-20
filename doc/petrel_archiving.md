
## Petrel / Raw Data Archival

Note: archival functionality currently requires updates to match new project
  filename conventions. (ok: ephys; todo: video)

The map-ephys pipeline does *not* directly handle processing of raw recording
files into the second-stage processed data used in later stages, however, some
facility is provided for tracking raw data files and transferring them to/from
the ANL [\'petrel\'](https://www.alcf.anl.gov/petrel) facility using 
the [globus toolkit](http://toolkit.globus.org/toolkit/) 
and [Globus Python SDK](https://globus-sdk-python.readthedocs.io/en/stable/).

## Petrel/Globus Archive Structure

The globus archive structure describes the layout of the raw files on
the archive system (petrel). Each archive type ('raw-ephys', 'raw-video')
has a corresponding StorageLocation which defines a globus endpoint
and a root path for that collection within the archive system.

Currently, these are:

raw-ephys 5b875fda-4185-11e8-bb52-0ac6873fc732:/4ElectrodeRig_Ephys
raw-video 5b875fda-4185-11e8-bb52-0ac6873fc732:/

### Electrophysiology

The electrophysiology recordings are layed out according to the following
format:

  - root
    - water restriction
      - session
        - probe

More specifically::

  <root>/<h2o>/catgt_<h2o>_<mdy>_g0/
  <root>/<h2o>/catgt_<h2o>_<mdy>_g0/<h2o>_<mdy>_imecN

In the event of multiple sessions in one day, additional sessions
should be labeled::

  <root>/<h2o>/catgt_<h2o>_<mdy>_<ident>_g0/
  <root>/<h2o>/catgt_<h2o>_<mdy>_<ident>_g0/<h2o>_<mdy>_<ident>_imecN

Where 'ident' is an identifier for that session. 

By default, 'ident' should be incrmentally numbered e.g. '1' 
is the second session for that day, etc. 

### Tracking Videos

The tracking video recordings are layed out according to the following
format:

  - root
    - water restriction
      - session date
        - 'video'
          - video file

More specifically::

  <root>/<h2o>/<MMDDYYYY>/video
  <root>/<h2o>/<MMDDYYYY>/video/<h2o>_<cameraposition>_NNN-NNNN.avi

In the event of multiple sessions in one day, additional sessions
should be labeled::

  <root>/<h2o>/<MMDDYYYY>_<ident>/video
  <root>/<h2o>/<MMDDYYYY>_<ident>/video/<h2o>_<cameraposition>_NNN-NNNN.avi

Where 'ident' is an identifier for that session. 

By default, 'ident' should be incrmentally numbered e.g. '1' 
is the second session for that day, etc. 

## MAP Pipeline Globus Configuration

The following section assumes you have setup the globus transfer client
and your account has been enabled for Petrel data access.

To use this facility, a 'globus endpoint' should be configured and the
following variables set in 'dj_local_conf.json' to match the configuration:

    "globus.local_endpoint": "<uuid value>",
    "globus.local_endpoint_subdir": "<path to storage inside globus>",
    "globus.local_endpoint_local_path": "<path to storage on local filesystem>",

The local endpoint UUID value can be obtained from the `manage endpoints` screen
within the globus web interface. The `endpoint_subdir` should be set to the
desired transfer location within the endpoint (as shown in the `location` bar
within the globus web interface `transfer files` screen), and the
`endpoint_local_path` should contain the 'real' filesystem location
corresponding to this location. For example, one might have the following
configuration on a mac or linux machine:

    "globus.local_endpoint": "C40E971D-0075-4A82-B12F-8F9717762E7B",
    "globus.local_endpoint_subdir": "map/raw",
    "globus.local_endpoint_local_path": "/globus/map/raw"

Or perhaps the following on a Windows machine:

    "globus.local_endpoint": "C40E971D-0075-4A82-B12F-8F9717762E7B",
    "globus.local_endpoint_subdir": "map/raw",
    "globus.local_endpoint_local_path": "C:\\globus\\map\\raw",

please note that the `local_endpoint_subdir` should use the globus
convention of using forward slashes (`/`) for directory separation,
whereas the `local_endpont_local_path` should use whatever convention
is used by the host operating system running the map-ephys code
('/' for mac/linux, '\' for windows).

## Login to Globus

Before the globus interface can be used, an application token must be generated.
This can be done from within the 'mapshell.py' shell as follows:

    $ ./scripts/mapshell.py shell
    Connecting chris@localhost:3306
    map shell.

    schema modules:

      - ephys
      - lab
      - experiment
      - ccf
      - publication
      - ingest.behavior
      - ingest.ephys

    >>> publication.GlobusStorageManager()
    Please login via: https://auth.globus.org/v2/oauth2/authorize?client_id=C40E971D-0075-4A82-B12F-8F9717762E7B&redirect_uri=https%3A%2F%2Fauth.globus.org%2Fv2%2Fweb%2Fauth-code&scope=openid+profile+email+urn%3Aglobus%3Aauth%3Ascope%3Atransfer.api.globus.org%3Aall&state=_default&response_type=code&code_challenge=cb9c10826b86ff9851cf477cad554e8d01c03a2b416b0210783c544deea1f372&code_challenge_method=S256&access_type=offline
    and enter code:f881e45fc2c52929b3924863c87f88
    INFO:datajoint.settings:Setting globus.token to c00264046ac1d484ec8db2b8acfa1f21ef0c68939d0ce2d562b21eb400eefd7802676efe1ddcc665141ef56f44699
    <pipeline.globus.GlobusStorageManager object at 0x10f874780>
    >>> dj.config.save('dj_local_conf.json')

This process only needs to be performed periodically to generate a valid
`dj_local_conf.json` file suitable for future sessions. Please note that after
login, the `dj_local_conf.json` file will contain a security sensitive access
token which should not be shared with anyone who should not have full access to
your globus account.

### Raw Recording Usage (Upload)

Once the local Globus endpoint has been configured, population and
data transfer of the raw data files can be performed using the
`mapshell.py` utility function `publication-publish`. This function
will call `.populate()` on the individual publication archive tables,
currently `ArchivedRawEphys` and `ArchivedTrackingVideo`.

The population logic will look for files matching project naming
conventions for the ingested experimental sessions, and if files
are found, transfer them to petrel and log an entry in the publication
schema. These records can then be used by users to query and retrieve
the raw data files (see below).

### Raw Recording Usage (Archive Discovery)

Once the local Globus endpoint has been configured, files contained
within the archive storage can be registered into the database using
the `.discover()` method of the publication archive tables.

This functionality is useful for rebuilding the archive schema in
the event local raw recording files are not available or in other,
similar cases.

### Raw Recording Usage (Download)

Once the local globus endpoint has been configured, retrieval of raw data
files can be done interactively using DataJoint operations in combination
with the `.retrieve()` method of the publication archive tables.

For example, to fetch all raw recordings related to subject #407513,
the following DataJoint expression can be used:

    >>> (publication.ArchivedRawEphys() 
         & {'subject_id': 407513}).retrieve()

This will save the related files to the configured local globus
storage path.  In the event of issues, be sure to check that the
endpoint is configured as writable. Also, to note, care should be
taken not to retrieve files on the same computer used to transmit
them, to keep the original files from being overwritten; Disabling
the writable setting on machines used for data upload should
help prevent this from possibly occurring.
