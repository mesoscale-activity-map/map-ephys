
'''
MAP Motion Tracking Schema
'''

import datajoint as dj

from . import experiment


schema = dj.schema(dj.config.get('tracking.database', 'map_tracking'))
[experiment]  # NOQA flake8


@schema
class TrackingDevice(dj.Lookup):
    definition = """
    tracking_device:                    varchar(20)     # device type/function
    ---
    sampling_rate:                      decimal(8, 4)   # sampling rate (Hz)
    tracking_device_description:        varchar(100)    # device description
    """
    contents = [
       ('Camera 0, side', 300, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
       ('Camera 1, bottom', 300, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)')]


@schema
class Tracking(dj.Imported):
    '''
    Video feature tracking.
    Position values in px; camera location is fixed & real-world position
    can be computed from px values.
    '''

    definition = """
    -> experiment.SessionTrial
    -> TrackingDevice
    """

    class NoseTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        nose_x:                 float   # nose x location (px)
        nose_y:                 float   # nose y location (px)
        nose_likelyhood:        float   # nose location likelyhood
        """

    class TongueTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        tongue_x:               float   # tongue x location (px)
        tongue_y:               float   # tongue y location (px)
        tongue_likelyhood:      float   # tongue location likelyhood
        """

    class JawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        jaw_x:                  float   # jaw x location (px)
        jaw_y:                  float   # jaw y location (px)
        jaw_likelyhood:         float   # jaw location likelyhood
        """
