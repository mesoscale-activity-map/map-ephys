# map-ephys
Mesoscale activity ephys ingest schema

For testing the schema, the dj_local_conf.json should have the following
configuration variables to modify data without affecting others' databases

  * "ingestBehavior.database": "[username]_ingestBehavior",
  * "ingestBehavior.database": "[username]_ingestEphys",
  * "ccf.database": "[username]_ccf",
  * "ephys.database": "[username]_ephys",
  * "experiment.database": "[username]_experiment",
  * "lab.database": "[username]_lab",

For accessing the map database, replace [username] with map. Please note all
users can write and Delete the databases.

ingestBehavior.py and ingestEphys.py require the rig_data_path and the username
to be specified in the code. Also, SessionDiscovery has the RigDataPath hard
coded for now. We will probably have a RigType key to track the training and
recording rigs.

See the notebooks, Overview.ipynb, for examples

## File path expectations of ingest logic

### Behavior Files

The behavior ingest logic searches the rig_data_paths for behavior files.  The
rig data paths are specified in specified dj.config.json as:

    "rig_data_paths": [
        ["RRig", "/Users/chris/src/djc/map/map-sharing/unittest/behavior", "0"]
    ],

if this variable is not configured, hardcoded values in the code matching the
experimental setup will be used.

File paths conform to the pattern:

    dl7/TW_autoTrain/Session Data/dl7_TW_autoTrain_20180104_132813.mat

which is, more generally:

    {h2o}/*/Session Data/{h2o}_{training protocol}_{YYYYMMDD}_{HHMMSS}.mat
 
### Ephys Files

The ephys ingest logic searches the ephys_data_paths for processed ephys
data files. These are specified in dj.config.json as:

    "ephys_data_paths": [["/Users/chris/src/djc/map/map-sharing/unittest/ephys", "0"]],

if this variable is not configured, hardcoded values in the code matching
the experimental setup will be used.

File paths conform to the pattern:

    \2017-10-21\tw5ap_imec3_opt3_jrc.mat

which is, more generally:

    \{YYYY}-{MM}-{DD}\{h2o}_ap_imec3_opt3_jrc.mat

Older files matched:
TODO Clarify - should code include actual logic to pick these up still?

    {h2o}_g0_t0.imec.ap_imec3_opt3_jrc.mat
    {h2o}_g0_t0.imec.ap_imec3_opt3_jrc.mat


### Unit Tests

TODO.
