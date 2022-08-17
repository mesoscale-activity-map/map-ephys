import datajoint as dj
import numpy as np
import pandas as pd

from . import lab, experiment, ccf, ephys, get_schema_name, create_schema_settings

[lab, experiment, ccf, ephys]  # schema imports only

schema = dj.schema(get_schema_name('histology'), **create_schema_settings)


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
class InterpolatedElectrodeCCF(dj.Computed):
    definition = """
    -> ElectrodeCCFPosition
    """

    class ElectrodePosition(dj.Part):
        definition = """
        -> master
        -> ElectrodeCCFPosition.ElectrodePosition
        """

    class ElectrodePositionError(dj.Part):
        definition = """
        -> master
        -> ElectrodeCCFPosition.ElectrodePositionError
        """

    @property
    def key_source(self):
        """
        Only Insertions with incomplete ElectrodeCCFPosition
        """
        all_electrodes = ElectrodeCCFPosition.aggr(ephys.ProbeInsertion * lab.ElectrodeConfig.Electrode,
                                                   elec_count='count(electrode)')
        ccf_electrodes = ElectrodeCCFPosition.aggr(ephys.ProbeInsertion * ElectrodeCCFPosition.ElectrodePosition,
                                                   ccf_count='count(electrode)')
        ccf_error_electrodes = ElectrodeCCFPosition.aggr(ephys.ProbeInsertion * ElectrodeCCFPosition.ElectrodePositionError,
                                                         ccf_err_count='count(electrode)')
        return ElectrodeCCFPosition & (all_electrodes * ccf_electrodes * ccf_error_electrodes
                                       & 'elec_count > ccf_count + ccf_err_count')

    def make(self, key):
        from scipy import interpolate

        ccf_res = ccf.CCFLabel.CCF_R3_20UM_RESOLUTION  # 20um voxel size
        ccf_label_id = ccf.CCFLabel.CCF_R3_20UM_ID

        electrode_ccf_query = (lab.ProbeType.Electrode
                               * lab.ElectrodeConfig.Electrode
                               * ephys.ProbeInsertion & key).join(
            ElectrodeCCFPosition.ElectrodePosition, left=True).proj(
            'shank', 'shank_col', 'shank_row',
            x='IFNULL(ccf_x, -1)', y='IFNULL(ccf_y, -1)', z='IFNULL(ccf_z, -1)')

        electrode_ccf = (dj.U('electrode_group', 'electrode',
                              'shank', 'shank_col', 'shank_row', 'x', 'y', 'z')
                         & electrode_ccf_query).fetch(format='frame').reset_index()
        electrode_ccf.set_index('electrode', inplace=True)
        electrode_ccf.replace(-1, np.nan, inplace=True)

        # per-shank, per-col interpolation of missing electrode site
        interp_electrodes = []
        for shank in set(electrode_ccf.shank):
            for shank_col in set(electrode_ccf.shank_col):
                col_ind = electrode_ccf.query(f'shank == {shank} & shank_col == {shank_col}').index
                is_nan = electrode_ccf['x'][col_ind].isna()
                interp_electrode = electrode_ccf.loc[col_ind[is_nan]]
                for c in ('x', 'y', 'z'):
                    # build interpolation function
                    interp_f = interpolate.interp1d(electrode_ccf[c][col_ind][~is_nan].index,
                                                    electrode_ccf[c][col_ind][~is_nan],
                                                    kind='linear', fill_value='extrapolate')
                    # interpolate missing data and round to CCF voxel size
                    interp_coord = interp_f(electrode_ccf[c][col_ind].index)
                    interp_coord = ccf_res * np.around(interp_coord / ccf_res)
                    interp_electrode[c] = interp_coord[is_nan]
                interp_electrodes.append(interp_electrode)
        interp_electrodes = pd.concat(interp_electrodes)

        # insert
        econfig_key = (ephys.ProbeInsertion * lab.ElectrodeConfig & key).fetch1('KEY')
        self.insert1(key)

        for _, r in interp_electrodes.iterrows():
            filled_position = {**econfig_key,
                               'electrode_group': r.electrode_group,
                               'electrode': r.name,
                               'ccf_label_id': ccf_label_id,
                               'ccf_x': r.x, 'ccf_y': r.y, 'ccf_z': r.z}

            try:
                ElectrodeCCFPosition.ElectrodePosition.insert1(filled_position, allow_direct_insert=True)
            except dj.errors.IntegrityError:
                if not (ElectrodeCCFPosition.ElectrodePositionError & {**econfig_key,
                                                                       'electrode_group': r.electrode_group,
                                                                       'electrode': r.name,
                                                                       'ccf_label_id': ccf_label_id}):
                    ElectrodeCCFPosition.ElectrodePositionError.insert1(filled_position, allow_direct_insert=True)
                    self.ElectrodePositionError.insert1(filled_position)
            else:
                self.ElectrodePosition.insert1(filled_position)


@schema
class InterpolatedShankTrack(dj.Computed):
    definition = """
    -> ElectrodeCCFPosition
    shank: int
    """

    class Point(dj.Part):
        definition = """  # CCF coordinates of all points on this interpolated shank track
        -> master
        -> ccf.CCF  
        """

    class BrainSurfacePoint(dj.Part):
        definition = """  # CCF coordinates of the brain surface intersection point with this shank
        -> master
        ---
        -> ccf.CCF  
        """

    class DeepestElectrodePoint(dj.Part):
        definition = """  # CCF coordinates of the most ventral recording electrode site (deepest in the brain)
        -> master
        ---
        -> ccf.CCF  
        """

    def make(self, key):
        probe_insertion = ephys.ProbeInsertion & key

        shanks = probe_insertion.aggr(lab.ElectrodeConfig.Electrode * lab.ProbeType.Electrode,
                                      shanks='GROUP_CONCAT(DISTINCT shank SEPARATOR ", ")').fetch1('shanks')
        shanks = np.array(shanks.split(', ')).astype(int)

        shank_points, brain_surface_points, last_electrode_points = [], [], []
        for shank in shanks:
            # shank points
            _, shank_ccfs = retrieve_pseudocoronal_slice(probe_insertion, shank)
            points = [{**key, 'shank': shank, 'ccf_label_id': 0,
                       'ccf_x': ml, 'ccf_y': dv, 'ccf_z': ap} for ml, dv, ap in shank_ccfs]
            shank_points.extend(points)
            # brain surface site
            brain_surface_points.append(points[0])
            # last electrode site
            last_electrode_site = (
                    lab.ElectrodeConfig.Electrode * lab.ProbeType.Electrode
                    * ephys.ProbeInsertion * ElectrodeCCFPosition.ElectrodePosition
                    & key & {'shank': shank}).fetch(
                'ccf_x', 'ccf_y', 'ccf_z', order_by='ccf_y DESC', limit=1)
            last_electrode_site = np.array([*last_electrode_site]).squeeze()
            last_electrode_points.append({**key, 'shank': shank, 'ccf_label_id': 0,
                                          'ccf_x': last_electrode_site[0],
                                          'ccf_y': last_electrode_site[1],
                                          'ccf_z': last_electrode_site[2]})

        self.insert({**key, 'shank': shank} for shank in shanks)
        self.Point.insert(shank_points)
        self.BrainSurfacePoint.insert(brain_surface_points)
        self.DeepestElectrodePoint.insert(last_electrode_points)


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
    if len(probe_track_coords):
        coords = np.vstack([electrode_coords, probe_track_coords])
    else:
        coords = electrode_coords

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
    voxel_res = ccf.CCFLabel.CCF_R3_20UM_RESOLUTION
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
