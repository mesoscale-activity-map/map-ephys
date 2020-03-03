# map-ephys advanced user guide

Mesoscale Activity Project Pipeline - Advanced User Guide

## Overview

This document outlines advanced usage of the MAP pipeline -
for example, pipeline developers, actual experimentors, or others
wishing to use the pipeline for more in-depth purposes than casual
browsing and analysis of the data. 

## Mapshell

The `mapshell.py` script contans utility functions to trigger file ingest for
behavior and ephys ingest, a facility to launch a basic python environment with
the various schema modules loaded, and a utlity to generate ERD diagrams from
the current code.

## Installation and Setup

Regular users who will be using the code for querying/analysis should
checkout the source code from git and install the package using pythons's `pip`
command. For example:

    $ git clone https://github.com/mesoscale-activity-map/map-ephys.git
    $ cd map-ephys
    $ pip install -e .

This will make the MAP pipeline modules available to your python interpreter as
'pipeline'. A convenience script, `mapshell.py` is available for basic queries
and use of administrative tasks. Account setup, test & usage synopsys using the
shell is as follows:

    $ mapshell.py shell
    Connecting chris@localhost:3306
    Please enter DataJoint username: chris
    Please enter DataJoint password: 
    map shell.

    schema modules:

    - ephys
    - lab
    - experiment
    - ccf
    - publication

    >>> lab.Person()
    *username    fullname
    +----------+ +----------+
    daveliu      Dave Liu
    (1 tuples)

    >>> dj.config.save_local()
    >>> sys.exit()
    $ mapshell.py 
    Connecting user@server:3306
    usage: mapshell.py [populateB|populateE|publish|shell|erd]

Direct installation without a source checkout may be desired for non-interactive
machines - this can be done directly via pip:

    $ pip install git+https://github.com/mesoscale-activity-map/map-ephys/

## Test configuration

For testing the schema, the dj_local_conf.json should have the following
configuration variables to modify data without affecting others' databases

  * ingest.behavior.database set to `[username]_ingestBehavior`
  * ingest.ephys.database set to `[username]_ingestEphys`
  * ccf.database set to `[username]_ccf`
  * ephys.database set to `[username]_ephys`
  * experiment.database set to `[username]_experiment`
  * lab.database set to `[username]_lab`
  * publication.database set to `[username]_publication`
  
For accessing the map database, replace `[username]` with `map`. Please note
that currently all users can write and Delete the databases.

For ingesting experiment data, the `ingest/behavior.py` and `ingest/ephys.py`
code requires the `rig_data_path` and the experimenter\'s username as found in
the `pipeline.lab.RigDataPath` and `pipeline.lab.User` tables to be specified in
the code (see below).

## Jupyter Notebook Documentation

Several [Jupyter Notebook](http://jupyter.org/) demonstrations are available in
the `notebook` portion of repository demonstrating usage of the pipeline.

See the jupyter notebook `notebooks/Overview.ipynb` for more details.

## Data Ingestion
With respect to the day-to-day research activity, ingestion of newly recorded data to the MAP pipeline comprises of 4 core components:
+ Behavior Ingest
    + Creation of ***session(s)*** 
    + Ingestion of the session-relevant behavioral data:
        + session meta information (e.g. subject, session's datetime, etc.)
        + trials and trial labels (e.g. type, response, photostim trial, etc.)
        + trial event (e.g. onsets of sample/delay/response period, onset of photostim, etc.)
+ Tracking Ingest
    + On a per-session basis, ingest tracking data (e.g. nose, tongue, jaw movements, from DLC)
+ Ephys Ingest
    + On a per-session basis, identify and ingest ***probe insertion*** information
    + Ingestion of clustering results per probe
    + Ingestion of Quality Control results (if any)
+ Histology Ingest
    + On a per-session basis, ingest electrode ccf location and probe track data
    
*Note: Behavior Ingest is required for Tracking/Ephys/Histology ingest to take place (can be ingested in any particular order)*

### Configure the data directory
Users performing the ingestion should configure the data path properly. Configuration of the data path is done in the ***dj_local_conf.json*** file in the following format:
```json
{
...
    "custom": 
    {
        "behavior_data_paths":
            [
                ["RRig", "C:/data/behavior_data_searchpath_1", 0],
                ["RRig2", "C:/data/behavior_data_searchpath_2", 1]
            ],
        "tracking_data_paths":
            [
                ["RRig", "C:/data/tracking_data_searchpath_1", 0],
                ["RRig2", "C:/data/tracking_data_searchpath_2", 1]
            ],
        "ephys_data_paths": ["C:/data/ephys_data_searchpath_1", "C:/data/ephys_data_searchpath_2", "C:/data/ephys_data_searchpath_3"],
        "histology_data_paths": ["C:/data/histology_data_searchpath_1"]
    }
}
```

### Running the data ingestion
Once the data paths are configured, executing the 4 ingestion routines above comes down to evoking the 4 corresponding commands:
>python scripts/mapshell.py ingest-behavior

>python scripts/mapshell.py ingest-tracking

>python scripts/mapshell.py ingest-ephys

>python scripts/mapshell.py ingest-histology

*Note: make sure to navigate yourself to the root of this project directory (e.g. `cd map-ephys`) if you don't already have the pipeline installed (with `pip install .` - see "Installation and Setup" above.* 

### Path Expectations

#### Behavior Files Ingestion

File paths conform to the pattern:

    dl7_TW_autoTrain_20180104_132813.mat

which is, more generally:

    {h2o}_*_*_{YYYYMMDD}_{HHMMSS}.mat
 
where:

  - `h2o` is the water restriction ID of the subject (see also `lab.Subject`)
  - `YYYYMMDD` and `HHMMSS` refer to the date and time of the session.
  
#### Ephys Files Ingestion
Ephys Files are typically structured in a directory convention as followed:

    {h2o}/{YYYYMMDD}/{probe_no}/ephys_file  
or

    {h20}/{h20}_{YYYYMMDD}_*/{h20}_{YYYYMMDD}_*_imec[0-9]/
    
Ephys Files can be:
+ JRClust v3 - e.g. "tw5ap_imec3_opt3_jrc.mat"
    + `{h2o}_*_jrc.mat`
+ JRClust v4 - e.g. "tw5ap_imec3_opt3.ap_res.mat"
    + `{h2o}_*.ap_res.mat`
+ Kilosort2 - *ks* folder with a set of ks output files (e.g. spike_times.npy, template.npy, etc.)
    
Some example ephys paths:
    
    dl40/20181022/1/dl40_g0_t50.imec.ap_imec3_opt3_jrc.mat
    SC022/20190303/1/SC022_g0_t0.imec0.ap_res.mat
    SC035/catgt_SC035_010720_g0/SC035_010720_g0_imec0/imec0_ks2
    SC035/catgt_SC035_011020_g0/SC035_011020_g0_imec2/imec2_ks2_orig
    
### Manual session-based ephys ingest
This section introduces additional manual routines for ephys ingest for any given behavior session after the main ephys ingestion (i.e. `mapshell.py ingest-ephys`) has been completed. 
There are 2 reasons for performing manual ephys ingestion after the `mapshell.py ingest-ephys` routine:    

1. Extending an ingested set of ephys results
    + This is needed when not all data were available during the 1st ingestion run. For instance, clustering results for probe 3 was not available at the time of the 1st ingestion - thus only 2 probe ephys data were ingested in a 3-probe recording session
    + This does not modify the already ingested ephys data, only extending it
2. Replacing an ingested set of ephys results
    + This is needed for a new version of curated clustering results to replace the ingested set of ephys results (or when quality controlled results become available)
    + There are 4 major steps taken in this routine:
        1. Search the specify data directory and verify the presence of new clustering results for the specified ***session***
        2. Copy the ingested ephys data over to the ***ArchiveUnit*** table, with all meta info tracked (this archived results will be stored externally on AWS S3 to reduce cost of storage)
        3. Delete the ingested data
        4. Re-ingest new version of the clustering results

Example execution:
```python
from pipeline.ingest import ephys as ephys_ingest
from pipeline import experiment

# obtain the session key for the session of interest

session_key = (experiment.Session & 'subject_id=471324' & 'session=2').fetch1('KEY')

# to extend ephys result:

ephys_ingest.extend_ephys_ingest(session_key)

# to archive and replace ephys result:

ephys_ingest.replace_ingested_clustering_results(session_key)
```

#####Setting up the external location for archiving previous ephys results

The `replace_ingested_clustering_results()` routine involves archiving previous ephys results in AWS S3 storage, thus users are required to configure this external storage (known in DataJoint as "store").
The configuration is as followed:

```json
{
...
    "stores": {
        "archive_store":
        {
            "protocol": "s3",
            "endpoint": "s3.amazonaws.com",
            "access_key": "s3_access_key",
            "secret_key": "s3_secret_key",
            "bucket": "map-cluster-archive",
            "location": "/cluster_archive",
            "stage": "/map_data/cluster_archive"
        }
    }
}    
```

*Note: if you don't have the ***access_key*** and ***secret_key*** to the AWS S3, contact your administrator for access request.
It is crucial that these keys are kept private and protected (a common mistake is committing your `dj_local_conf.json` to a public github repository)*

## Raw Recording File Publication and Retrieval

The map-ephys pipeline does *not* directly handle processing of raw reording
files into the second-stage processed data used in later stages, however, some
facility is provided for tracking raw data files and transferring them to/from
the ANL [\'petrel\'](https://www.alcf.anl.gov/petrel) facility using the [globus toolkit](http://toolkit.globus.org/toolkit/) and [Globus Python SDK](https://globus-sdk-python.readthedocs.io/en/stable/).

### Globus Configuration

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

please note that the `local_endpoint_subdir` should use the globus convention of
using forward slashes (`/`) for directory separation, whereas the
`local_endpont_local_path` should use whatever convention is used by the host
operating system running the map-ephys code. Please note that since the
backslash character (`\`) is treated specially within python strings, the
backslashes used in Windows-style paths should be specified twice as shown
above.

### Login to Globus

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

As can be seen from the above, calling `publication.GlobusStorageManager()` from
the mapshell interpreter will prompt the user to login in a web browser, and
paste an authrization code back to the script, which then completes the process
of requesting an access token. From there, the configuration is saved via the
DataJoint `dj.config.save` utility function.

This process only needs to be performed once to generate a valid
`dj_local_conf.json` file suitable for future sessions. Please note that after
login, the `dj_local_conf.json` file will contain a security sensitive access
token which should not be shared with anyone who should not have full access to
your globus account.

### Raw Recording Usage (Data Publisher)

Once the local globus endpoint has been configured, population and data transfer
of the raw data files can be performed using the `mapshell.py` utility function
`publish`, which simply wraps a call to `ArchivedRawEphysTrial.populate()`.

The population logic will look for files underneath the local storage path
matching project naming conventions for the ingested experimental sessions, and
if corresponding files are found, transfer them to petrel and log an available
entry in the publication schema. These records can then be used by users to
query and retrieve the raw data files (see below).

### Raw Recording Usage (Data Consumer)

Once the local globus endpoint has been configured, retrieval of raw data
files can be done interactively using DataJoint operations in combination
with the `.retrieve()` method of the `ArchivedRawEphysTrial` class.

For example, to fetch all raw recordings related to subject #407513,
the following DataJoint expression can be used:

    >>> (publication.ArchivedRawEphysTrial() & {'subject_id': 407513}).retrieve()

This will save the related files to the configured local globus storage path.
In the event of issues, be sure to check that the endpoint is configured as
writable. Also, to note, care should be taken not to retrieve files on the same
computer used to transmit them, lest the original files be overwritten;
Disabling the writable setting on machines used for data publication should 
help prevent this issue from occurring.

### Unit Tests

Unit tests for ingest are present in the `tests` directory and can be run using
the `nostests` command. These tests will *destructively* delete the actively
configured databases and perform data ingest and transfer tasks using the data
stored within the 'test_data' directory - as such, care should be taken not to
run the tests against the live database configuration settings.

