

import csv
import logging

import numpy as np
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
    annotation  : varchar(1024)
    index (annotation)
    """

    @classmethod
    def get_ccf_r3_20um_ontology_regions(cls):
        return [c for c in csv.reader(ccf_ontology.splitlines())
                if len(c) == 2]

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
        stack = imread(stack_path)  # load reference stack,

        log.info('.. loaded stack of shape {} from {}'
                 .format(stack.shape, stack_path))

        # iterate over ccf ontology region id/name records,
        regions = self.get_ccf_r3_20um_ontology_regions()
        region, nregions = 0, len(regions)
        chunksz, ib_args = 50000, {'skip_duplicates': True,
                                   'allow_direct_insert': True}

        for num, txt in regions:

            region += 1
            num = int(num)

            log.info('.. loading region {} ({}/{}) ({})'
                     .format(num, region, nregions, txt))

            # extracting filled volumes from stack in scaled [[x,y,z]] shape,
            vol = np.array(np.where(stack == num)).T[:, [2, 1, 0]] * 20

            if not vol.shape[0]:
                log.info('.. region {} volume: shape {} - skipping'
                         .format(num, vol.shape))
                continue

            log.info('.. region {} volume: shape {}'.format(num, vol.shape))

            with dj.conn().transaction:
                with InsertBuffer(CCF, chunksz, **ib_args) as buf:
                    for vox in vol:
                        buf.insert1((CCFLabel.CCF_R3_20UM_ID, *vox))
                        buf.flush()

                with InsertBuffer(cls, chunksz, **ib_args) as buf:
                    for vox in vol:
                        buf.insert1((CCFLabel.CCF_R3_20UM_ID, *vox,
                                     CCFLabel.CCF_R3_20UM_TYPE, txt))
                        buf.flush()

        log.info('.. done.')


@schema
class AnnotatedBrainSurface(dj.Manual):
    definition = """  # iso-surface of annotated brain in CCF coordinate frame
    annotated_brain_name: varchar(100)  # e.g. Annotation_new_10_ds222_16bit
    ---
    vertices: longblob  # (px)
    faces: longblob
    """

    @classmethod
    def load_matlab_mesh(self, mesh_fp):
        mesh_fp = pathlib.Path(mesh_fp).resolve()
        assert mesh_fp.exists()
        mesh = scio.loadmat(mesh_fp, struct_as_record = False, squeeze_me = True)['mesh']
        self.insert1(dict(annotated_brain_name=mesh_fp.stem,
                          vertices=mesh.vertices,
                          faces=mesh.faces - 1),  #  0-base index
                     allow_direct_insert=True)
