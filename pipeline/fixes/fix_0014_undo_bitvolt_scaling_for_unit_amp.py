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
from pipeline.ingest.ephys import npx_bit_volts

log = logging.getLogger(__name__)


@schema
class UndoBitVoltScalingAmpUnit(dj.Manual):
    """
    This table keeps track of all the units undergone undoing of the bitvolt scaling
     for quality control clustering results (read with metrics.csv)
    """
    definition = """ # This table accompanies fix_0014
    -> FixHistory
    -> ephys.Unit
    ---
    fixed: bool
    scale: float
    """


def undo_bitvolt_scaling(insertion_keys={}):
    """
    This is a one-time operation only - Oct 2020
    """

    units2fix = ephys.Unit * ephys.ClusteringLabel & insertion_keys & 'quality_control = 1'  # only on QC results
    units2fix = units2fix - (UndoBitVoltScalingAmpUnit & 'fixed=1')  # exclude those that were already fixed

    if not units2fix:
        return

    # safety check, no jrclust results
    assert len(units2fix & 'clustering_method LIKE "jrclust%"') == 0

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}
    FixHistory.insert1(fix_hist_key)

    for unit in tqdm(units2fix.proj('unit_amp').fetch(as_dict=True)):
        probe_type = (ephys.ProbeInsertion & unit).fetch1('probe_type')
        bit_volts = npx_bit_volts[re.match('neuropixels (\d.0)', probe_type).group()]

        amp = unit.pop('unit_amp')
        with dj.conn().transaction:
            (ephys.Unit & unit)._update('unit_amp', amp * 1/bit_volts)
            UndoBitVoltScalingAmpUnit.insert1({**fix_hist_key, **unit, 'fixed': True, 'scale': 1/bit_volts})

    # delete cluster_quality figures and remake figures with updated unit_amp
    with dj.config(safemode=False):
        (report.ProbeLevelReport & units2fix).delete()


if __name__ == '__main__':
    undo_bitvolt_scaling()
