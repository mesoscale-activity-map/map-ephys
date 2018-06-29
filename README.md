# map-ephys
Mesoscale activity ephys ingest schema

For testing the schema, the dj_local_conf.json should have the following configuration variables to modify data without affecting  others' databases

"ingestBehavior.database": "[username]_ingestBehavior",
"ingestBehavior.database": "[username]_ingestEphys",
"ccf.database": "[username]_ccf",
"ephys.database": "[username]_ephys",
"experiment.database": "[username]_experiment",
"lab.database": "[username]_lab",

For accessing the map database, replace [username] with map. Please note all users can write and Delete the databases.

ingestBehavior.py and ingestEphys.py require the rig_data_path and the username to be specified in the code. Also, SessionDiscovery has the RigDataPath hard coded for now. We will probably have a RigType key to track the training and recording rigs.

See the notebooks, Overview.ipynb, for examples

