# Mesoscale Activity Project Pipeline

Mesoscale Activity Project Pipeline - [map-ephys](https://github.com/mesoscale-activity-map/map-ephys/)

## Documentation

Visit [here](doc/intro.md) for full documentation of this project


## The MAP Navigator

A website dedicated for data browsing and results visualization for MAP project can be accessed at
 [https://navigator.mesoscale-activity-map.org/](http://navigator.mesoscale-activity-map.org/)


## Access via Jupyter Hub

A MAP-dedicated Jupyter Hub can be accessed at
 [https://jupyter.mesoscale-activity-map.org/](https://jupyter.mesoscale-activity-map.org/)

Users can sign in using their GitHub account. Please contact chris@vathes.com for database credential if you don't already have one.

Once signed in, the foler `/work` is dedicated for each user to freely experiment with the pipeline. 
Guidelines, tutorial materials, example notebooks can be found under `/workshop` or `/map-ephys/notebook`


## Local Installation and Setup

Direct installation without a source checkout can be done directly via pip:

    $ pip install git+https://github.com/mesoscale-activity-map/map-ephys/

Regular users who will be using the code for querying/analysis should
checkout the source code from git and install the package using pythons's `pip`
command. For example:

    $ git clone https://github.com/mesoscale-activity-map/map-ephys.git
    $ cd map-ephys
    $ pip install -e .

This will make the MAP pipeline modules available to your python interpreter as
'pipeline'. 

For more details on system setup and administration, visit [here](doc/system_management.md)
