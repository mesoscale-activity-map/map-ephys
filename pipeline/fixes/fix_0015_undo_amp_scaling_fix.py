#! /usr/bin/env python

import logging

import datajoint as dj
import pathlib
from tqdm import tqdm
from datetime import datetime

from pipeline import ephys, report
from pipeline.fixes import schema, FixHistory
from pipeline.fixes.fix_0007_amplitude_scaling_for_npx2_probes import FixedAmpUnit

log = logging.getLogger(__name__)

"""
This fix is to undo the amplitude scaling from fix_0007.
The 3.01 scaling fix done on fix_0007 is unnecessary and need to be reverted.

Full sequence of events: `FixHistory.fetch(order_by='fix_timestamp')`
1. fix_0007 was applied - 3.01 scaling applied to unit_amp and unit_waveform
2. fix_0008 was applied - unit_waveform reingested
3. fix_0009 was applied - 3.01 scaling applied to unit_waveform only (as they were reingested)
4. fix_0008 was applied again - reingest unit_waveform for more sessions

Here, the units underwent step-4 (fix_0008 after fix_0009) should have had the 3.01 scaling applied to them as well, 
as they were reingested, however this did not happen!
Hence, to undo all of this, the fix should undo amplitude scaling only for `unit_amp` for all units in fix_0007
"""


@schema
class UndoFixedAmpUnit(dj.Manual):
    """
    This table accompanies fix_0015, undo the amplitude scaling from fix_0007.
    The 3.01 scaling fix done on fix_0007 is unnecessary and need to be reverted.
    """
    definition = """ # This table accompanies fix_0015
    -> FixHistory
    -> ephys.Unit
    ---
    fixed: bool
    scale: float
    """


def undo_amplitude_scaling():

    amp_scale = 1 / 3.01

    units2fix = ephys.Unit & FixedAmpUnit  # only fix those units that underwent fix_0007
    units2fix = units2fix - (UndoFixedAmpUnit & 'fixed=1')  # exclude those that were already fixed

    if not units2fix:
        return

    # safety check, no jrclust results and no npx 1.0
    assert len(units2fix & 'clustering_method LIKE "jrclust%"') == 0
    assert len(units2fix.proj() * ephys.ProbeInsertion & 'probe_type LIKE "neuropixels 1.0%"') == 0

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}
    FixHistory.insert1(fix_hist_key)

    for unit in tqdm(units2fix.proj('unit_amp').fetch(as_dict=True)):
        amp = unit.pop('unit_amp')
        with dj.conn().transaction:
            (ephys.Unit & unit)._update('unit_amp', amp * amp_scale)
            FixedAmpUnit.insert1({**fix_hist_key, **unit, 'fixed': True, 'scale': amp_scale})

    # delete cluster_quality figures and remake figures with updated unit_amp
    with dj.config(safemode=False):
        (report.ProbeLevelReport & units2fix).delete()


if __name__ == '__main__':
    undo_amplitude_scaling()
