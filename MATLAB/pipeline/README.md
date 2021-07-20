# Working with the map-ephys DataJoint pipeline in MATLAB

The MATLAB scripts here provide a setup for users to interact 
with the map-ephys DataJoint pipeline using MATLAB

## Getting started

To get started, follow the steps below:
1. navigate to this folder (.../map-ephys/MATLAB/pipeline)
2. from MATLAB command prompt, run `init`
3. You're all set, now you can interact with the pipeline, see some example below

## Usage examples

All schemas have `v_` prepended ("v" for virtual)
You can use all schemas the same way you would in Python

    
    v_experiment.Session()
    v_tracking.Tracking()
    
    v_experiment.Session & 'rig = "RRig-MTL"'
    v_experiment.Session & 'rig = "RRig-MTL"' & v_tracking.Tracking
    
    mtl_sessions = fetch(v_experiment.Session & 'rig = "RRig-MTL"')
    
    mtl_session = mtl_sessions(1)
    v_tracking.TrackingTongueTracking & 'tracking_device = "Camera 4"' & mtl_session
    
    cam4_tongue = fetch(v_tracking.TrackingTongueTracking & 'tracking_device = "Camera 4"' & mtl_session, '*')
    
 