#! /usr/bin/env python

import logging

import datajoint as dj
import numpy as np
import pathlib
from datetime import datetime
from tifffile import imread

from pipeline import ccf
from pipeline.fixes import FixHistory

log = logging.getLogger(__name__)


def add_ontology_region_id_and_hexcode():
    """
    Update to the ccf.CCFAnnotation table, udpating values for 2 new attributes:
        + ontology_region_id
        + color_code
    """

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}
    FixHistory.insert1(fix_hist_key)

    stack_path = dj.config['custom']['ccf.r3_20um_path']
    stack = imread(stack_path)  # load reference stack

    log.info('.. loaded stack of shape {} from {}'
             .format(stack.shape, stack_path))

    # iterate over ccf ontology region id/name records,
    regions = ccf.CCFAnnotation().get_ccf_r3_20um_ontology_regions()
    region, nregions = 0, len(regions)

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
            for vox in vol:
                key = {'ccf_label_id': ccf.CCFLabel.CCF_R3_20UM_ID,
                       'ccf_x': vox[0], 'ccf_y': vox[1], 'ccf_z': vox[2],
                       'annotation_type': ccf.CCFLabel.CCF_R3_20UM_TYPE}
                assert (ccf.CCFAnnotation & key).fetch1('annotation') == region_name
                (ccf.CCFAnnotation & key)._update('ontology_region_id', region_id)
                (ccf.CCFAnnotation & key)._update('color_code', color_hexcode)

    log.info('.. done.')


if __name__ == '__main__':
    add_ontology_region_id_and_hexcode()
