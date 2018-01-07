# map-ephys
Mesoscale activity ephys ingest schema

For testing the schema, the dj_local_conf.json should have the following configuartion variables to modify data without affecting  others's databases

"ingest.database": "daveliu_ingest",
"ccf.database": "daveliu_ccf",
"ephys.database": "daveliu_map_ephys",
"experiment.database": "daveliu_map_experi",
"lab.database": "daveliu_map",

ingest.py requires the rig_data_path and the username to be specified.

See Ingest Behavior Data.ipynb for the required tables to be inserted. The example behavioral data file is 'tw5_TW_autoTrain_20171021_150914.mat'.

See Ingest Ephys Data for an example of the Ephys data ingestion. The example Ephys data file is 'real5ap_imec3_opt3_jrc.mat'.
