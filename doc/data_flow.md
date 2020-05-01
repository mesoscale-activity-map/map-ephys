# MAP Data Flow

This document captures the sequence of data import, preprocessing, analysis, and
 figure generation implemented in the MAP pipeline. 

## `dj.Lookup` data
 

## Data Ingestion

Roughly speaking, the data being imported into the `Imported` tables of this pipeline
 belong to four primary categories, correspondingly accompanied by four "ingestion schemas" and scripts:

+ Behavior
    + schema: `ingest_behavior`
    + ingestion [script](../pipeline/ingest/behavior.py)
+ Tracking
    + schema: `ingest_tracking`
    + ingestion [script](../pipeline/ingest/tracking.py)
+ Ephys
    + schema: `ingest_ephys`
    + ingestion [script](../pipeline/ingest/ephys.py)
+ Histology
    + schema: `ingest_histology`
    + ingestion [script](../pipeline/ingest/ephys.py)

### Behavior Ingestion

The ingestion of behavior data is performed by the `BehaviorIngest`'s `make()` function.
 Each `make()` call perform the import for one session, determined by browsing the data folder
 and following the folder/file naming convention specified in the [Ingestion Instruction](./ingestion_instruction.md).
 
`BehaviorIngest` performs ingestion on a ***per-session*** basis,
 populating data into the tables outlined in red in the diagram below.

![behavior_ingest](./pipeline_architecture/static/MAP_ingestion_diagram-behavior_ingest.svg)


### Tracking Ingestion

Similarly, `TrackingIngest` performs ingestion on a ***per-session*** basis,
 populating data into the tables outlined in red in the diagram below.

![tracking_ingest](./pipeline_architecture/static/MAP_ingestion_diagram-tracking_ingest.svg)

The association of whisker tracking time series and the tracked whisker(s) are to be done as follow up manual insertion step.


### Ephys Ingestion

`EphysIngest` performs ingestion on a ***per-session*** basis,
 search for all Probe and Ephys data associated with that session,
 and populate data into the tables outlined in red in the diagram below.

![ephys_ingest](./pipeline_architecture/static/MAP_ingestion_diagram-ephys_ingest.svg)

The detailed steps are as followed:
1. Search data directory for clustering folder per probe - e.g. kilosort or jrclust
2. Per probe, performs:
    1. Read neuropixels meta file (*_ap.meta) for probe type, probe's serial number, and electrode configuration
    2. Insert new `ephys.ProbeInsertion`
    3. Read clustering results for units, unit labels, spike times, waveforms, etc.
    4. Insert new `ephys.Unit` for each unit
    5. Read quality control data (metrics.csv) if available, insert new:
        +`ephys.UnitStat`
        +`ephys.ClusterMetric`
        +`ephys.WaveformMetric`
    6. Read ***bitcode*** data file for trial numbering, start times, and "go-cue" onset
    7. Extract spike times per unit per trial, time alignment to "go-cue", and insert new `ephys.Unit.TrialSpike`
3. Insertion details are to be subsequently manually inserted by experimenters:
    + `ephys.ProbeInsertion.InsertionLocation`
    + `ephys.ProbeInsertion.RecordableBrainRegion`

### Histology Ingestion

`HistologyIngest` performs ingestion on a ***per-probe*** basis, populating electrodes' CCF locations and probe tracks.

![histology_ingest](./pipeline_architecture/static/MAP_ingestion_diagram-histology_ingest.svg)



