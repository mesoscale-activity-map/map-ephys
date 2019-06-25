import datajoint as dj

from . import lab, experiment, ccf, ephys
from . import get_schema_name
[lab, experiment, ccf, ephys]  # schema imports only

schema = dj.schema(get_schema_name('histology'))


@schema
class CCFToMRITransformation(dj.Imported):
    definition = """  # once per project - a mapping between CCF coords and MRI coords (e.g. average MRI from 10 brains)
    project_name: varchar(32)  # e.g. MAP
    """

    class Landmark(dj.Part):
        definition = """
        -> master
        landmark_id: int
        ---
        landmark_name='': varchar(32)
        mri_x: float  # (um)
        mri_y: float  # (um)
        mri_z: float  # (um)
        -> ccf.CCF
        """


@schema
class RawToCCFTransformation(dj.Imported):
    definition = """
    -> lab.Subject
    """

    class Landmark(dj.Part):
        definition = """
        -> master
        landmark_name='': varchar(8)
        ---
        raw_x: float  # (um)
        raw_y: float  # (um)
        raw_z: float  # (um)
        -> ccf.CCF
        """


@schema
class ElectrodeCCFPosition(dj.Manual):
    definition = """
    -> ephys.ProbeInsertion
    """

    class ElectrodePosition(dj.Part):
        definition = """
        -> lab.ElectrodeConfig.Electrode
        -> ccf.CCF
        """

    class ElectrodePositionError(dj.Part):
        definition = """
        -> lab.ElectrodeConfig.Electrode
        -> ccf.CCFLabel
        x   :  int   # (um)
        y   :  int   # (um)
        z   :  int   # (um)
        """


@schema
class LabeledProbeTrack(dj.Manual):
    definition = """
    -> ephys.ProbeInsertion
    ---
    labeling_date=NULL:         date
    dye_color=NULL:             varchar(32)
    """

    class Point(dj.Part):
        definition = """
        -> master
        -> ccf.CCF
        """

    class PointError(dj.Part):
        definition = """
        -> master
        -> ccf.CCFLabel
        x   :  int   # (um)
        y   :  int   # (um)
        z   :  int   # (um)
        """


@schema
class EphysCharacteristic(dj.Imported):
    definition = """
    -> ephys.ProbeInsertion
    -> lab.ElectrodeConfig.Electrode
    ---
    lfp_power: float
    waveform_amplitude: float
    waveform_width: float
    firing_rate: float
    percentage_change: float
    """
