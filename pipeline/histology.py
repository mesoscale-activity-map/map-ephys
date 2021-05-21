import datajoint as dj
import numpy as np

from . import lab, experiment, ccf, ephys, get_schema_name
from pipeline.plot import unit_characteristic_plot

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
        -> lab.ElectrodeConfig.Electrode
        -> ccf.CCF
        ---
        mri_x=null: float  # (mm)
        mri_y=null: float  # (mm)
        mri_z=null: float  # (mm)
        """

    class ElectrodePositionError(dj.Part):
        definition = """
        -> master
        -> lab.ElectrodeConfig.Electrode
        -> ccf.CCFLabel
        ccf_x: int   # (um)
        ccf_y: int   # (um)
        ccf_z: int   # (um)
        ---
        mri_x=null: float  # (mm)
        mri_y=null: float  # (mm)
        mri_z=null: float  # (mm)
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
        shank: int
        ---
        ccf_x: float  # (um)
        ccf_y: float  # (um)
        ccf_z: float  # (um)    
        """


@schema
class InterpolatedShankTrack(dj.Computed):
    definition = """
    -> ElectrodeCCFPosition
    shank: int
    """

    class Point(dj.Part):
        definition = """
        -> master
        -> ccf.CCF  
        """

    def make(self, key):
        probe_insertion = ephys.ProbeInsertion & key

        shanks = probe_insertion.aggr(lab.ElectrodeConfig.Electrode * lab.ProbeType.Electrode,
                                      shanks='GROUP_CONCAT(DISTINCT shank SEPARATOR ", ")').fetch1('shanks')
        shanks = np.array(shanks.split(', ')).astype(int)

        shank_points = []
        for shank in shanks:
            _, shank_ccfs = retrieve_pseudocoronal_slice(probe_insertion, shank)
            shank_points.extend([{**key, 'shank': shank, 'ccf_label_id': 0,
                                  'ccf_x': ml, 'ccf_y': dv, 'ccf_z': ap}
                                 for ml, dv, ap in shank_ccfs])

        self.insert({**key, 'shank': shank} for shank in shanks)
        self.Point.insert(shank_points)


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


@schema
class ArchivedElectrodeHistology(dj.Manual):
    definition = """
    -> ephys.ProbeInsertion
    archival_time: datetime  # time of archiving
    ---
    archival_note='': varchar(2000)  # user notes about this particular Electrode CCF being archived
    archival_hash: varchar(32)        # hash of Electrode CCF position, prevent duplicated archiving
    unique index (archival_hash)
    """

    class ElectrodePosition(dj.Part):
        definition = """
        -> master
        -> lab.ElectrodeConfig.Electrode
        -> ccf.CCF
        ---
        mri_x=null  : float  # (mm)
        mri_y=null  : float  # (mm)
        mri_z=null  : float  # (mm)
        """

    class ElectrodePositionError(dj.Part):
        definition = """
        -> master
        -> lab.ElectrodeConfig.Electrode
        -> ccf.CCFLabel
        ccf_x       : int   # (um)
        ccf_y       : int   # (um)
        ccf_z       : int   # (um)
        ---
        mri_x=null  : float  # (mm)
        mri_y=null  : float  # (mm)
        mri_z=null  : float  # (mm)
        """

    class LabeledProbeTrack(dj.Part):
        definition = """
        -> master
        ---
        labeling_date=NULL:         date
        dye_color=NULL:             varchar(32)
        """

    class ProbeTrackPoint(dj.Part):
        definition = """
        -> master.LabeledProbeTrack
        order: int
        shank: int
        ---
        ccf_x: float  # (um)
        ccf_y: float  # (um)
        ccf_z: float  # (um)    
        """


# ====================== HELPER METHODS ======================


def retrieve_pseudocoronal_slice(probe_insertion, shank_no=1):
    """
    For each shank, retrieve the pseudocoronal slice of the brain that the shank traverses
    This function returns a tuple of 2 things:
    1. an array of (CCF_DV, CCF_ML, CCF_AP, color_codes) for all points in the pseudocoronal slice
    2. an array of (CCF_DV, CCF_ML, CCF_AP) for all points on the interpolated track of the shank
    """
    probe_insertion = probe_insertion.proj()

    # ---- Electrode sites ----
    annotated_electrodes = (lab.ElectrodeConfig.Electrode * lab.ProbeType.Electrode
                            * ephys.ProbeInsertion
                            * ElectrodeCCFPosition.ElectrodePosition
                            & probe_insertion & {'shank': shank_no})

    electrode_coords = np.array(list(zip(*annotated_electrodes.fetch(
        'ccf_z', 'ccf_y', 'ccf_x', order_by='ccf_y'))))  # (AP, DV, ML)
    probe_track_coords = np.array(list(zip(*(LabeledProbeTrack.Point
                                             & probe_insertion & {'shank': shank_no}).fetch(
        'ccf_z', 'ccf_y', 'ccf_x', order_by='ccf_y'))))
    coords = np.vstack([electrode_coords, probe_track_coords])

    # ---- linear fit of probe in DV-AP axis ----
    X = np.asmatrix(np.hstack((np.ones((coords.shape[0], 1)), coords[:, 1][:, np.newaxis])))  # DV

    y = np.asmatrix(coords[:, 0]).T  # AP
    XtX = X.T * X
    Xty = X.T * y
    ap_fit = np.linalg.solve(XtX, Xty)

    y = np.asmatrix(coords[:, 2]).T  # ML
    XtX = X.T * X
    Xty = X.T * y
    ml_fit = np.linalg.solve(XtX, Xty)

    # ---- predict x coordinates ----
    voxel_res = ccf.get_ccf_vox_res()
    lr_max, dv_max, _ = ccf.get_ccf_xyz_max()

    dv_coords = np.arange(0, dv_max, voxel_res)

    X2 = np.asmatrix(np.hstack((np.ones((len(dv_coords), 1)), dv_coords[:, np.newaxis])))

    ap_coords = np.array(X2 * ap_fit).flatten().astype(np.int)
    ml_coords = np.array(X2 * ml_fit).flatten().astype(np.int)

    # round to the nearest voxel resolution
    ap_coords = (voxel_res * np.round(ap_coords / voxel_res)).astype(np.int)
    ml_coords = (voxel_res * np.round(ml_coords / voxel_res)).astype(np.int)

    # ---- extract pseudoconoral plane from DB ----
    q_ccf = ccf.CCFAnnotation * ccf.CCFBrainRegion.proj(..., annotation='region_name')
    dv_pts, lr_pts, ap_pts, color_codes = (
            q_ccf & [{'ccf_y': dv, 'ccf_z': ap} for ap, dv in zip(ap_coords, dv_coords)]).fetch(
        'ccf_y', 'ccf_x', 'ccf_z', 'color_code', order_by='ccf_y')

    # ---- CCF coords for voxels on the interpolated probe/shank track ----
    coronal_point_cloud = set(zip(lr_pts, dv_pts, ap_pts))  # ML, DV, AP
    shank_ccfs = np.vstack([(ml, dv, ap)
                            for dv, ap, ml in zip(dv_coords, ap_coords, ml_coords)
                            if (ml, dv, ap) in coronal_point_cloud])

    return np.vstack([dv_pts, lr_pts, ap_pts, color_codes]).T, shank_ccfs
