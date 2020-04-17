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
class FixedAmpUnit(dj.Manual):
    """
    This table accompanies fix_0007, keeping track of the units that have had a scaling factor of 3.01 applied.
    This table can be dropped once confident that all units have been corrected - ideally right after
    executing the one-time fix_0007
    """
    definition = """ # This table accompanies fix_0007
    -> FixHistory
    -> ephys.Unit
    ---
    fixed: bool
    scale: float
    """


def apply_amplitude_scaling(insertion_keys={}):
    """
    This is a one-time operation only - April 2020
    Kilosort2 results from neuropixels probe 2.0 requires an additionally scaling factor of 3.01 applied
    to the unit amplitude and mean waveform.
    Future version of quality control pipeline will apply this scaling.
    """

    amp_scale = 3.01

    npx2_inserts = ephys.ProbeInsertion & insertion_keys & 'probe_type LIKE "neuropixels 2.0%"'

    units2fix = ephys.Unit * ephys.ClusteringLabel & npx2_inserts.proj() & 'quality_control = 1'
    units2fix = units2fix - (FixedAmpUnit & 'fixed=1')  # exclude those that were already fixed

    if not units2fix:
        return

    # safety check, no jrclust results
    assert len(units2fix & 'clustering_method LIKE "jrclust%"') == 0

    fix_key = {'fix_name': pathlib.Path(__file__).name,
               'fix_timestamp': datetime.now()}
    FixHistory.insert1(fix_key)

    for unit in tqdm(units2fix.proj('unit_amp', 'waveform').fetch(as_dict=True)):
        amp = unit.pop('unit_amp')
        wf = unit.pop('waveform')
        with dj.conn().transaction:
            (ephys.Unit & unit)._update('unit_amp', amp * amp_scale)
            (ephys.Unit & unit)._update('waveform', wf * amp_scale)
            FixedAmpUnit.insert1({**fix_key, **unit, 'fixed': True, 'scale': amp_scale})

    # delete cluster_quality figures and remake figures with updated unit_amp
    with dj.config(safemode=False):
        (report.ProbeLevelReport & npx2_inserts).delete()


if __name__ == '__main__':
    apply_amplitude_scaling()
