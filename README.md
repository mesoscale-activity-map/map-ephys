# map-ephys
Mesoscale activity ephys ingest schema

For testing the schema, the dj_local_conf.json should have the following configuration variables to modify data without affecting  others's databases

"ingest.database": "[username]_ingest",
"ccf.database": "[username]_ccf",
"ephys.database": "[username]map_ephys",
"experiment.database": "[username]_map_experi",
"lab.database": "[username]_map",

ingest.py requires the rig_data_path and the username to be specified in the code. Also, SessionDiscoverty has the RigDataPath hard coded for now. We will probably have a RigType key to track the training and recording rigs.

See Overview.ipynb for the required tables to be inserted.

The example behavioral data file is 'tw5_TW_autoTrain_20171021_150914.mat'.
The example Ephys data file is 'tw5ap_imec3_opt3_jrc.mat'.
