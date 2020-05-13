
## Test / Development configuration

For developing changes to the pipeline, the dj_local_conf.json
'custom' dictionary can be modified to contain the following
configuration variables to modify data without affecting others'
databases:

  * ingest.behavior.database
  * ingest.ephys.database
  * lab.database
  * experiment.database
  * ephys.database
  * psth.database
  * ccf.database
  * histology.database
  * report.database
  * publication.database

See also the 'get_schema_name' function in pipeline/__init__.py which is
used by the various MAP pipeline modules to adust DataJoint schema at runtime.  

## Unit Tests

Unit tests for ingest are present in the `tests` directory and can be run using
the `nostests` command. These tests will *destructively* delete the actively
configured databases and perform data ingest and transfer tasks using the data
stored within the 'test_data' directory - as such, care should be taken not to
run the tests against the live database configuration settings.

