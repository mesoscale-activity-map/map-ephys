#! /usr/bin/env python

import logging

import datajoint as dj
import re
import numpy as np
import pathlib
from tqdm import tqdm
from datetime import datetime

from pipeline import lab, experiment, ephys, report
from pipeline.fixes import schema, FixHistory

log = logging.getLogger(__name__)


@schema
class FixedAmpWfUnit(dj.Manual):
    """
    This table accompanies fix_0009, keeping track of the units that have had a scaling factor of 3.01 applied.
    This table can be dropped once confident that all units have been corrected
    """
    definition = """ # This table accompanies fix_0009
    -> FixHistory
    -> ephys.Unit
    ---
    fixed: bool
    scale: float
    """


def apply_amplitude_scaling(insertion_keys={}):
    """
    This fix is identical to that of fix_0007 - apply an amplitude scaling (3.01) to npx 2.0 probe units
    The difference is that this fix only apply the scaling to mean waveform, and not unit_amp
    """

    amp_scale = 3.01

    npx2_inserts = ephys.ProbeInsertion & insertion_keys & 'probe_type LIKE "neuropixels 2.0%"'

    units2fix = ephys.Unit * ephys.ClusteringLabel & npx2_inserts.proj() & 'quality_control = 1'
    units2fix = units2fix - (FixedAmpWfUnit & 'fixed=1')  # exclude those that were already fixed

    if not units2fix:
        return

    # safety check, no jrclust results
    assert len(units2fix & 'clustering_method LIKE "jrclust%"') == 0

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}
    FixHistory.insert1(fix_hist_key)

    for unit in tqdm(units2fix.proj('waveform').fetch(as_dict=True)):
        wf = unit.pop('waveform')
        with dj.conn().transaction:
            (ephys.Unit & unit)._update('waveform', wf * amp_scale)
            FixedAmpWfUnit.insert1({**fix_hist_key, **unit, 'fixed': True, 'scale': amp_scale})


if __name__ == '__main__':
    apply_amplitude_scaling()
