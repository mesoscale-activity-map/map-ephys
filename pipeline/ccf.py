import csv
import logging

import numpy as np
import pandas as pd
import datajoint as dj
import pathlib
import scipy.io as scio

from tifffile import imread

from . import InsertBuffer
from .reference import ccf_ontology
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
    CCF_R3_20UM_DESC = 'Allen Institute Mouse CCF, Rev. 3, 20uM Resolution'
    CCF_R3_20UM_TYPE = 'CCF_R3_20UM'

    contents = [(CCF_R3_20UM_ID, 3, 20, CCF_R3_20UM_DESC)]


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
class AnnotationType(dj.Lookup):
    definition = """
    annotation_type  : varchar(16)
    """
    contents = ((CCFLabel.CCF_R3_20UM_TYPE,),)


@schema
class CCFAnnotation(dj.Manual):
    definition = """
    -> CCF
    -> AnnotationType
    ---
    ontology_region_id: int  
    annotation  : varchar(1024)
    index (annotation)
    color_code: varchar(6) # hexcode of the color code of this region
    """

    @classmethod
    def get_ccf_r3_20um_ontology_regions(cls):
        return [c for c in csv.reader(ccf_ontology.splitlines())
                if len(c) == 3]

    @classmethod
    def load_ccf_r3_20um(cls):
        """
        Load the CCF r3 20 uM Dataset.
        Requires that dj.config['ccf.r3_20um_path'] be set to the location
        of the CCF Annotation tif stack.
        """
        # TODO: scaling
        log.info('CCFAnnotation.load_ccf_r3_20um(): start')

        self = cls()  # Instantiate self,
        stack_path = dj.config['custom']['ccf.r3_20um_path']
        stack = imread(stack_path)  # load reference stack

        log.info('.. loaded stack of shape {} from {}'
                 .format(stack.shape, stack_path))

        # iterate over ccf ontology region id/name records,
        regions = self.get_ccf_r3_20um_ontology_regions()
        region, nregions = 0, len(regions)
        chunksz, ib_args = 50000, {'skip_duplicates': True,
                                   'allow_direct_insert': True}

        for region_id, region_name, color_hexcode in regions:

            region += 1
            region_id = int(region_id)

            log.info('.. loading region {} ({}/{}) ({})'
                     .format(region_id, region, nregions, region_name))

            # extracting filled volumes from stack in scaled [[x,y,z]] shape,
            vol = np.array(np.where(stack == region_id)).T[:, [2, 1, 0]] * 20

            if not vol.shape[0]:
                log.info('.. region {} volume: shape {} - skipping'
                         .format(region_id, vol.shape))
                continue

            log.info('.. region {} volume: shape {}'.format(region_id, vol.shape))

            with dj.conn().transaction:
                with InsertBuffer(CCF, chunksz, **ib_args) as buf:
                    for vox in vol:
                        buf.insert1((CCFLabel.CCF_R3_20UM_ID, *vox))
                        buf.flush()

                with InsertBuffer(cls, chunksz, **ib_args) as buf:
                    for vox in vol:
                        buf.insert1({'ccf_label_id': CCFLabel.CCF_R3_20UM_ID,
                                     'ccf_x': vox[0], 'ccf_y': vox[1], 'ccf_z': vox[2],
                                     'annotation_type': CCFLabel.CCF_R3_20UM_TYPE,
                                     'ontology_region_id': region_id,
                                     'annotation': region_name,
                                     'color_code': color_hexcode})
                        buf.flush()

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
