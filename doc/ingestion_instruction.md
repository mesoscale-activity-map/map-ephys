
# Data Ingestion Setup and Instructions

## Overview 

This section describes in detail the programming environment setup
and configuration to ingest data into this pipeline. This is
specifically relevant to the researchers that will contribute data
to this project, particularly the researchers at Janelia Research
Campus.

## Installation and Setup

Regular users who will be using the code for querying/analysis should
checkout the source code from git and install the package using python's `pip`
command. For example:

    $ git clone https://github.com/mesoscale-activity-map/map-ephys.git
    $ cd map-ephys
    $ pip install -e .

This will make the MAP pipeline package available to your python interpreter 
as 'pipeline'. For example, to import the MAP experiment module, users can
import pipeline.experiment:

    >>> from pipeline import experiment

and so on. Please see the respective documentation for more on pipeline 
code contents and DataJoint schemas.

Direct installation without a source checkout may be desired for non-interactive
machines - this can be done directly via pip:

    $ pip install git+https://github.com/mesoscale-activity-map/map-ephys/

## Mapshell: mapshell.py and pipeline/shell.py

The python module pipeline/shell.py contains utility functions to
trigger file ingest for behavior and ephys ingest, a facility to
launch a basic python environment with the various schema modules
loaded, and a utility to generate ERD diagrams from the current code,
and other administrative tasks.

These functions are also made available in the `mapshell.py` script
which can be used as a command line utility for common tasks.

This chapter is written with the stand-alone mapshell.py method in mind.
If you wish to use the shell as a module, please see shell.py's 'actions'
variable for the functions corresponding to the shell commands.

Account setup, test & usage synopsis using the shell is as follows:

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
    usage: mapshell.py [ingest-behavior|ingest-ephys|ingest-tracking|ingest-histology|auto-ingest|populate-psth|publish|export-recording|generate-report|sync-report|shell|erd|ccfload|automate-computation|automate-sync-and-cleanup|load-insertion-location|load-animal] <args>

The mapshell.py script and pipeline.shell module support
runtime-configurable logging via the environment variable MAP_LOGLEVEL
and dj.config['loglevel'].  Possible values and logging configuration
are processed by the 'pipeline.shell.logsetup' function, which will also
enable logging to the file specified in dj.config['custom']['logfile'],
if this setting is present.

## Data Ingestion

With respect to the day-to-day research activity, ingestion of newly
recorded data to the MAP pipeline is comprised of 4 main steps
discussed below. This section only discusses ingest steps; for
further details about the data loaded in each stage please see the
main pipeline documentation.

1) Behavior Ingest

   The behavior ingest phase loads behavior data into the pipeline.
   This includes:

     - Main Session record
     - Session meta information (e.g. subject, session's datetime, etc.)
     - Trials and trial labels (e.g. type, response, photostim trial, etc.)
     - Trial events (e.g. onsets of sample/delay/response period, 
       onset of photostim, etc.)
    
   ### Delay-Response Experiment
   To perform behavior ingest for the ***delay-response*** experiment, users can additionally configure the
   dj.config['custom'] variable 'behavior_data_paths' list. Each item
   in this list is itself a list containing the rig name, path to rig 
   data files, and a priority number. For example:

       >>> mypaths = [['Rig1', '/data/rig1', 0]]
       >>> dj.config['custom']['behavior_data_paths'] = mypaths
       >>> dj.config.save_local()
       
   or in ***dj_local_conf.json***:
       
   ```json
    "custom": {
            "behavior_data_paths":
            [
                ["Rig1", "/data/rig1", 0],
                ["RRig2", "/data/rig2", 1],
                ["Tower-3", "/data/tower3", 2]
            ]
    }
   ```

   After any necessary configuration, behavior ingest is run using
   the 'ingest-behavior' mapshell command:

       $ mapshell.py ingest-behavior

   This action will traverse the contents of the rig data paths, finding
   behavior session files and attempting to load them.
   
   ### Foraging Experiment
   To perform behavior ingest for the ***foraging*** experiment, users will need to configure the dj.config['custom'] 
   to contain the ***behavior_bpod*** variable, which specifies 2 fields:
    + ***meta_dir***: full path to the directory containing all of the meta data csv files for all subjects (e.g. FOR01.csv, FOR02.csv...)
    + ***project_paths***: full path to all of the bpod project data directories
    
   For example, in ***dj_local_conf.json***:
       
   ```json
    "custom": {
            "behavior_bpod": {
                "meta_dir": "path/to/Metadata",
                "project_paths": 
                [
                    "/path/to/bpod/Tower-1/Foraging",
                    "/path/to/bpod/Tower-2/Foraging",
                    "/path/to/bpod/Tower-2/Foraging_homecage"
                ]
            },
    }
   ```

   After any necessary configuration, behavior ingest is run using
   the 'ingest-foraging' mapshell command:

       $ mapshell.py ingest-foraging

   This action will traverse the contents of the rig data paths, finding
   behavior session files and attempting to load them.
  
2) Tracking Ingest

   The tracking ingest phase loads feature position data into the
   pipeline (e.g. nose, tongue, jaw movements, from DLC). The
   ingest should be run after behavior is ingested, and will attempt
   to load position tracking information for sessions where it does
   not already exist.

   If needed, users can additionally configure a dj.config['custom']
   variable to adjust local data paths for tracking:

       >>> mypaths = [['Rig1', '/data/rig1']]
       >>> dj.config['custom']['tracking_data_paths'] = mypaths
       >>> dj.config.save_local()
       
   or in ***dj_local_conf.json***:
       
   ```json
    "custom": {
        "tracking_data_paths":
            [
                ["RRig1", "/data/rig1"],
                ["RRig2", "/data/rig2"]
            ]
    }
   ```

   After any necessary configuration, behavior ingest is run using
   the 'ingest-tracking' mapshell command:

       $ mapshell.py ingest-tracking

   This action will jump to the expected location of tracking data for
   each session without tracking and attempt to load any tracking data which
   is available.

3) Electrophysiology Ingest

   The electrophysiology ingest phase loads processed spike data output from
   Jrclust3, Jrclust4, and kilosort2 on a per-probe basis. This includes:

     - Ingestion of ***probe insertion*** information
     - Ingestion of clustering results
     - Ingestion of quality control results (if available)

   If needed, users can additionally configure a dj.config['custom']
   variable to adjust local data paths for ephys data.
    
   In ***dj_local_conf.json***:
       
   ```json
    "custom": {
        "ephys_data_paths": ["/path/to/ephys", "/path/to/ephys2"]
    }
   ```
       
   After any necessary configuration, ephys ingest is run using
   the 'ingest-ephys' mapshell command:

       $ mapshell.py ingest-ephys

   This action will jump to the expected location of ephys data for
   each session without ephys data and attempt to load any ephys data which
   is available.

   Subsequent stages of the pipeline also require manual entry of
   probe insertion locations (ephys.ProbeInsertion.InsertionLocation)
   into the pipeline to assist in downstream computations. This can
   be done directly using DataJoint, or via an excel spreadsheet
   helper routine 'load-insertion-location'.  Please see the code
   for load-insertion-location for further details on the excel
   approach.

4) Histology Ingest

   The histology ingest phase loads processed histology details
   into the
   pipeline. These include:

     - electrode ccf location 
     - probe track data

   In addition to session behavior data, Histology data load is
   dependent on loading the reference CCF volume data. The reference
   volume used in the MAP pipeline is the Allen Institute CCF r3
   20 uM Dataset tiff stack. This is done manually once prior to
   ingest of any histology data, see section (5) below.

   If needed, users can additionally configure a dj.config['custom']
   variable to adjust local data paths for session histology data:

       >>> mypaths = ["/path/to/histology"]
       >>> dj.config['custom']['histology_data_paths'] = mypaths
       >>> dj.config.save_local()
   
   or in ***dj_local_conf.json***:
   
   ```json
    "custom": {
        "histology_data_paths": ["/path/to/histology"]
    }
   ```

   Histology paths should be layed out as follows:

     {histology_data_path}/{h2o}/histology/{files}

   Expected files are documented in the function:
   
     pipeline.ingest.histologyHistologyIngest._search_histology_files

   After any necessary configuration, histology ingest is run using
   the 'ingest-histology' mapshell command:

       $ mapshell.py ingest-histology

   This action will jump to the expected location of histology data for
   each session without ephys data and attempt to load any ephys data which
   is available.

5) CCF Ingest

   New ingestion of CCF is rarely needed, typically only applicable when a
   new CCF Annotation version is available and the `ccf.CCFBrainRegion` 
   and `ccf.CCFAnnotation` tables are required to be updated.

   To do CCF ingestion, set up `dj_local_conf.json` to contain:
   
      >>> from pipeline import ccf
      >>> dj.config['custom']['ccf_data_paths'] = {
          'version_name': 'CCF_2017',
          'region_csv': '/data/ccf/mousebrainontology_2.csv',
          'hexcode_csv': '/data/ccf/hexcode.csv',
          'annotation_nrrd: '/data/ccf/annotation_10.nrrd",
      }
      >>> ccf.CCFBrainRegion.load_regions()
      >>> ccf.CCFAnnotation.load_ccf_annotation()

   The ingest expects the Allen 10 uM input volume and downscales to 20 uM
   precision on ingest.

   Or, users can setup the `ccf_data_paths` in `dj_local_conf.json` 
   as above and run:
   
       $ mapshell.py ccfload
