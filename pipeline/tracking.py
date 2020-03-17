
'''
MAP Motion Tracking Schema
'''

import datajoint as dj

from . import experiment
from . import get_schema_name

schema = dj.schema(get_schema_name('tracking'))
[experiment]  # NOQA flake8


@schema
class TrackingDevice(dj.Lookup):
    definition = """
    tracking_device:                    varchar(20)     # device type/function
    ---
    tracking_position:                  varchar(20)     # device position
    sampling_rate:                      decimal(8, 4)   # sampling rate (Hz)
    tracking_device_description:        varchar(100)    # device description
    """
    contents = [
       ('Camera 0', 'side', 1/0.0034, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
       ('Camera 1', 'bottom', 1/0.0034, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)'),
        ('Camera 2', 'body', 1/0.01, 'Chameleon3 CM3-U3-13Y3M-CS (FLIR)')]


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
    ---
    tracking_samples:           int             # number of events (possibly frame number, relative to the start of the trial)
    """

    class NoseTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        nose_x:                 longblob        # nose x location (px)
        nose_y:                 longblob        # nose y location (px)
        nose_likelihood:        longblob        # nose location likelihood
        """

    class TongueTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        tongue_x:               longblob        # tongue x location (px)
        tongue_y:               longblob        # tongue y location (px)
        tongue_likelihood:      longblob        # tongue location likelihood
        """

    class JawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        jaw_x:                  longblob        # jaw x location (px)
        jaw_y:                  longblob        # jaw y location (px)
        jaw_likelihood:         longblob        # jaw location likelihood
        """

    class LeftPawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        left_paw_x:             longblob        # left paw x location (px)
        left_paw_y:             longblob        # left paw y location (px)
        left_paw_likelihood:    longblob        # left paw location likelihood
        """

    class RightPawTracking(dj.Part):
        definition = """
        -> Tracking
        ---
        right_paw_x:            longblob        # right paw x location (px)
        right_paw_y:            longblob        # right_paw y location (px)
        right_paw_likelihood:   longblob        # right_paw location likelihood
        """


# ---------- HELPER FUNCTIONS --------------

tracking_feature_tables = {'NoseTracking': Tracking.NoseTracking,
                           'TongueTracking': Tracking.TongueTracking,
                           'JawTracking': Tracking.JawTracking,
                           'LeftPawTracking': Tracking.LeftPawTracking,
                           'RightPawTracking': Tracking.RightPawTracking}
