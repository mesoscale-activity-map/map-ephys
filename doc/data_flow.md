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

