import logging

import numpy as np
import pandas as pd
import datajoint as dj
import pathlib
import scipy.io as scio

import nrrd

from . import InsertBuffer
from . import get_schema_name

schema = dj.schema(get_schema_name('ccf'))

log = logging.getLogger(__name__)


@schema
class CCFLabel(dj.Lookup):
    definition = """
    # CCF Dataset Information
    ccf_label_id:       int             # Local CCF ID
    ---
    ccf_version:        int             # Allen CCF Version
    ccf_resolution:     int             # Voxel Resolution (uM)
    ccf_description:    varchar(255)    # CCFLabel Description
    """
    CCF_R3_20UM_ID = 0
    CCF_R3_20UM_VERSION = 3
    CCF_R3_20UM_RESOLUTION = 20
    CCF_R3_20UM_DESC = 'Allen Institute Mouse CCF, Rev. 3, 20uM Resolution'

    contents = [(CCF_R3_20UM_ID,
                 CCF_R3_20UM_VERSION,
                 CCF_R3_20UM_RESOLUTION,
                 CCF_R3_20UM_DESC)]


@schema
class CCF(dj.Lookup):
    definition = """
    # Common Coordinate Framework
    -> CCFLabel
    ccf_x   :  int   # (um)
    ccf_y   :  int   # (um)
    ccf_z   :  int   # (um)
    """


@schema
class AnnotationVersion(dj.Lookup):
    definition = """ 
    annotation_version: varchar(32)  # e.g. CCF_2017
    ---
    annotation_desc='': varchar(255)
    """
    contents = (('CCF_2017', ''),)


@schema
class CCFBrainRegion(dj.Lookup):
    definition = """
    -> AnnotationVersion
    region_name : varchar(128)
    ---
    region_id: int
    color_code: varchar(6) # hexcode of the color code of this region
    """

    @classmethod
    def load_regions(cls):
        version_name = dj.config['custom']['ccf_data_paths']['version_name']
        regions = get_ontology_regions()
        cls.insert([dict(annotation_version=version_name,
                         region_id=region_id,
                         region_name=r.region_name,
                         color_code=r.hexcode) for region_id, r in regions.iterrows()],
                   skip_duplicates=True)


@schema
class CCFAnnotation(dj.Manual):
    definition = """
    -> CCF
    -> CCFBrainRegion.proj(annotation='region_name')
    """

    @classmethod
    def load_ccf_annotation(cls):
        """
        Load the CCF r3 10 uM NRRD Dataset scaled to 20um.

        Expects dj.config:

            "custom": {
                "ccf_data_paths": {
                    "version_name": "CCF_2017",
                    "annotation_nrrd": "annotation_10.nrrd",
                    "region_csv": "mousebrainontology_2.csv",
                    "hexcode_csv": "hexcode.csv"
                }
            }

        see also: 

        http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017

        """

        ccf_vol_res = CCFLabel.CCF_R3_20UM_RESOLUTION
        vol_res = 10                         # volume resolution from the nrrd
        ds_factor = ccf_vol_res / vol_res    # down-sample factor
        assert int(ds_factor) == ds_factor   # futureproofing paranoia
        ds_factor = int(ds_factor)           # int(down-sample factor)
        scale_factor = vol_res * ds_factor   # voxel to uM scale factor

        log.info('CCFAnnotation.load_ccf_annotation(): start')

        version_name = dj.config['custom']['ccf_data_paths']['version_name']
        stack_path = dj.config['custom']['ccf_data_paths']['annotation_nrrd']

        stack, hdr = nrrd.read(stack_path)  # AP (ccf_z), DV (ccf_y), ML (ccf_x)

        # downsample by a factor of "ds_factor" (all 3 dimensions)
        stack = stack[::ds_factor, ::ds_factor, ::ds_factor]

        log.info('.. loaded stack of shape {} from {}'
                 .format(stack.shape, stack_path))

        # iterate over ccf ontology region id/name records,
        regions = get_ontology_regions()
        chunksz, ib_args = 50000, {'skip_duplicates': True,
                                   'allow_direct_insert': True}

        for idx, (region_id, r) in enumerate(regions.iterrows()):

            region_id = int(region_id)

            log.info('.. loading region {} ({}/{}) ({})'
                     .format(region_id, idx, len(regions), r.region_name))

            # extracting filled volumes from stack in scaled [[x,y,z]] shape,
            vol = (np.array(np.where(stack == region_id)).T[:, [2, 1, 0]]
                   * scale_factor)

            if not vol.shape[0]:
                log.info('.. region {} volume: shape {} - skipping'
                         .format(region_id, vol.shape))
                continue

            log.info('.. region {} volume: shape {}'.format(
                region_id, vol.shape))

            with dj.conn().transaction:
                with InsertBuffer(CCF, chunksz, **ib_args) as buf:
                    for vox in vol:
                        buf.insert1((CCFLabel.CCF_R3_20UM_ID, *vox))

                with InsertBuffer(cls, chunksz, **ib_args) as buf:
                    for vox in vol:
                        buf.insert1({'ccf_label_id': CCFLabel.CCF_R3_20UM_ID,
                                     'ccf_x': vox[0], 'ccf_y': vox[1], 'ccf_z': vox[2],
                                     'annotation_version': version_name,
                                     'annotation': r.region_name})

        log.info('.. done.')


@schema
class AnnotatedBrainSurface(dj.Manual):
    definition = """  # iso-surface of annotated brain in CCF coordinate frame (e.g. Annotation_new_10_ds222_16bit.mat)
    annotated_brain_name: varchar(100)  # e.g. Annotation_new_10_ds222_16bit_isosurf.mat
    ---
    vertices: longblob  # (px)
    faces: longblob
    """

    @classmethod
    def load_matlab_mesh(self, mesh_fp):
        mesh_fp = pathlib.Path(mesh_fp).resolve()
        if not mesh_fp.exists():
            raise FileNotFoundError('Unable to locate {}'.format(mesh_fp))
        mesh = scio.loadmat(mesh_fp, struct_as_record=False, squeeze_me=True)['mesh']
        self.insert1(dict(annotated_brain_name=mesh_fp.stem,
                          vertices=mesh.vertices,
                          faces=mesh.faces - 1),  #  0-base index
                     allow_direct_insert=True)


# ========= HELPER METHODS ======


def get_ontology_regions():
    regions = pd.read_csv(dj.config['custom']['ccf_data_paths']['region_csv'], header=None, index_col=0)
    regions.columns = ['region_name']
    hexcode = pd.read_csv(dj.config['custom']['ccf_data_paths']['hexcode_csv'], header=None, index_col=0)
    hexcode.columns = ['hexcode']

    return pd.concat([regions, hexcode], axis=1)

