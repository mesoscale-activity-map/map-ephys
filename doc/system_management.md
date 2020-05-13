
# MAP Pipeline System Management

This section is describes running and management of this pipeline,
including details for:

  - MAP Pipeline Configuration Repository
  - AWS resource management
  - Petrel/Globus storage management
  - MAP Pipeline database administration
  - MAP Pipeline JupyterHub administration
  - Setup and launching of workers for automated tasks
  - Setup and launching of the MAP Web-GUI

These are discussed in more detail in the subsections which follow.

## Map Pipeline Configuration Repository

A git repository containing configuration logic for automating most
of the system setup for the workers, JupyterHub, and MAP Web-GUI
is maintained by the project. Since some of the configuration is
security sensitive, the repository is kept private.  Please discuss
with other members for access if desired.

## AWS Resource Management

Currently, with the exception of Petrel/Globus storage, the key MAP
computational resources used for the central production pipeline
and user facing systems are deployed in Amazon Web Services (AWS).

This currently consists of 4 components:

  - 1x MAP Database Server (AWS MySQL RDS)
  - 1x MAP Computational Server (AWS EC2)
  - 1x MAP Report Storage (AWS S3)
  - 1x MAP Ephys Archive Storage (AWS S3)

These services are deployed in the us-east-1 zone and managed via
standard AWS tools in a dedicated MAP AWS account.


## Petrel/Globus Storage Management

Petrel is a research data storage system provided by ANL. The MAP project
has a storage allocation on Petrel which is made accessible via Globus
to the project. Aside from data contents, all administrative activities
related to Petrel and Petrel's Globus interface are handled directly by
ANL staff.

## MAP Pipeline Database Administration

The MAP project database server is the main component of the central
production pipeline accessible for project use. The database server 
is a AWS MySQL RDS instance deployed to the us-east-1 region. 

Project-specific configuration includes user management and database
parameter adjustment for DataJoint.  Core MySQL/RDS administration
is beyond the scope of this document, please see AWS and MySQL
documentation where applicable.

Project specific Users are configured with full permissions to
`username_` prefix schemas, and varying degrees of access to `map_`
schemas depending on the user's role within the project. In addition
to normal users, accounts are also enabled for the MAP pipeline
processing workers and Web GUI.

DataJoint specific parameters for MySQL are documented in the DataJoint
mysql-docker code repository: https://github.com/datajoint/mysql-docker

## MAP Pipeline JupyterHub Administration

The MAP pipeline JupyterHub environment runs on the MAP Computational
Server and is enabled using github authentication. The JupyterHub
per-user environment is built to contain DataJoint and the MAP Pipeline 
code and is manually rebuilt when new feature or updates are added. 

Specifics of managing JupyterHub are beyond the scope of this document,
please see JupyterHub documentation for general information and the
MAP pipeline configuration repository for the MAP-specific configuration
of JupyterHub and the per-user environment.

## MAP Pipeline Processing Workers

To ensure computed pipeline results are kept up to date, the map pipeline
contains a set of 'worker' scripts which are used to perform various 
background processing tasks. These tasks include:

    - Automatic computation of computed data table results
    - Automatic generation of figures for the Map Web GUI
    - Cleanup and pruning of outdated computations and figures

These jobs are currently run on the MAP Computational server.  The
production worker environment dj_local_config.json should be
configured with appropriate credentials and have a report store
enabled for storage of generated figures.  The precise server-specific
deployment logic is in the MAP pipeline configuration repository.

Generically, The logic for the worker code, along with configuration
for running the various worker tasks is available in the 'workers'
subdirectory of the map-epyhs project. The configuration uses
'Docker' and 'docker-compose' to build and run the processing
workers. Once these tools are installed, the background workers can
be configured by editing the '.env' file in the workers directory,
and then starting the workers using docker-compose.

For example:

   $ cat .env
   report_store_stage=/data/stage
   worker_count=4
   $ docker-compose up -d

## MAP Web GUI

The MAP Web GUI allows casual navigation of available sessions and related
plots and figures. The code for the web GUI resides in the 'map-WebGUI'
repository: https://github.com/mesoscale-activity-map/map-WebGUI.

The GUI architecture consists of 2 pieces:

  - Backend API server (provides frontend API with MAP Database Data)
  - Frontend/API server (javascript, provides UI with data)

The Backend API server is written in python and serves to provide the frontend
API server with MAP Database data. The Frontend/API server is written in
javascript and serves the frontend UI code as well as provides API endpoints
that the UI needs for its operation.

Currently, the Map WEB GUI runs on the MAP Computational server
using docker and docker-compose.  deployment specifics are contained
in the core map-WebGUI repository as well as the MAP pipeline
configuration repository.

