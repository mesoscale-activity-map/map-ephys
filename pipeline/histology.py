import datajoint as dj

from . import lab, experiment, ccf, ephys
from . import get_schema_name
[lab, experiment, ccf, ephys]  # schema imports only

schema = dj.schema(get_schema_name('histology'))


@schema
class CCFToMRITransformation(dj.Imported):
    definition = """  # one per project - a mapping between CCF coords and MRI coords (e.g. average MRI from 10 brains)
    project_name: varchar(32)  # e.g. MAP
    """

    class Landmark(dj.Part):
        definition = """
        -> master
        landmark_id: int
        ---
        landmark_name='': varchar(32)
        mri_x: float  # (mm)
        mri_y: float  # (mm)
        mri_z: float  # (mm)
        ccf_x: float  # (um)
        ccf_y: float  # (um)
        ccf_z: float  # (um)
        """


@schema
class SubjectToCCFTransformation(dj.Imported):
    definition = """  # one per subject
    -> lab.Subject
    """

    class Landmark(dj.Part):
        definition = """
        -> master
        landmark_name:          char(8)         # pt-N from landmark file.
        ---
        subj_x:                 float           # (a.u.)
        subj_y:                 float           # (a.u.)
        subj_z:                 float           # (a.u.)
        ccf_x:                  float           # (um)
        ccf_y:                  float           # (um)
        ccf_z:                  float           # (um)
        """


@schema
class ElectrodeCCFPosition(dj.Manual):
    definition = """
    -> ephys.ProbeInsertion
    """

    class ElectrodePosition(dj.Part):
        definition = """
        -> master
        -> lab.Probe.Electrode
        -> ccf.CCF
        mri_x: float  # (mm)
        mri_y: float  # (mm)
        mri_z: float  # (mm)
        """

    class ElectrodePositionError(dj.Part):
        definition = """
        -> master
        -> lab.Probe.Electrode
        -> ccf.CCFLabel
        ccf_x: int   # (um)
        ccf_y: int   # (um)
        ccf_z: int   # (um)
        mri_x: float  # (mm)
        mri_y: float  # (mm)
        mri_z: float  # (mm)
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
        order: int
        ---
        ccf_x: float  # (um)
        ccf_y: float  # (um)
        ccf_z: float  # (um)    
        """


@schema
class EphysCharacteristic(dj.Imported):
    definition = """
    -> ephys.ProbeInsertion
    -> lab.ElectrodeConfig.Electrode
    ---
    lfp_theta_power: float
    lfp_beta_power: float
    lfp_gama_power: float
    waveform_amplitude: float
    waveform_width: float
    mua: float
    photstim_effect: float
    """
