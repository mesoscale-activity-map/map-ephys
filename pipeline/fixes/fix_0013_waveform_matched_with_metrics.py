#! /usr/bin/env python

import os
import logging
import pathlib
from datetime import datetime
from itertools import repeat
import numpy as np
import datajoint as dj

from pipeline import lab, experiment, ephys

from pipeline.ingest import ProbeInsertionError, ClusterMetricError
from pipeline.ingest.ephys import (get_ephys_paths, cluster_loader_map,
                                   _get_sess_dir, _match_probe_to_ephys)
from pipeline.fixes import schema, FixHistory


os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

log = logging.getLogger(__name__)


@schema
class FixQCsessionWaveform(dj.Manual):
    """
    This table accompanies fix_0013, keeping track of the units with waveform fixed.
    This tracking is not extremely important, as a rerun of this fix_0013 will still yield the correct results
    """
    definition = """ # This table accompanies fix_0013
    -> FixHistory
    -> ephys.Unit
    ---
    fixed: bool
    """


def update_waveforms(session_keys={}):
    """
    For results with quality control, updating unit-waveform
     where a unit's waveform read from waveform.npy needs to be filtered down by
        i) units in metrics.csv
        ii) channels in "channel_map"
    Applicable to only kilosort2 clustering results with quality control
    """
    sessions_2_update = (experiment.Session & (ephys.Unit * ephys.ClusteringLabel
                                               & 'quality_control = 1'
                                               & 'clustering_method = "kilosort2"') & session_keys)
    sessions_2_update = sessions_2_update - FixQCsessionWaveform

    if not sessions_2_update:
        return

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}

    for key in sessions_2_update.fetch('KEY'):
        success = _update_one_session(key)
        if success:
            FixHistory.insert1(fix_hist_key, skip_duplicates=True)
            FixQCsessionWaveform.insert([{**fix_hist_key, **ukey, 'fixed': 1} for ukey in (ephys.Unit & key).fetch('KEY')])


def _update_one_session(key):
    log.info('\n======================================================')
    log.info('Waveform update for key: {k}'.format(k=key))

    #
    # Find Ephys Recording
    #
    key = (experiment.Session & key).fetch1()
    sinfo = ((lab.WaterRestriction
              * lab.Subject.proj()
              * experiment.Session.proj(..., '-session_time')) & key).fetch1()

    rigpaths = get_ephys_paths()
    h2o = sinfo['water_restriction_number']

    sess_time = (datetime.min + key['session_time']).time()
    sess_datetime = datetime.combine(key['session_date'], sess_time)

    for rigpath in rigpaths:
        dpath, dglob = _get_sess_dir(rigpath, h2o, sess_datetime)
        if dpath is not None:
            break

    if dpath is not None:
        log.info('Found session folder: {}'.format(dpath))
    else:
        log.warning('Error - No session folder found for {}/{}. Skipping...'.format(h2o, key['session_date']))
        return False

    try:
        clustering_files = _match_probe_to_ephys(h2o, dpath, dglob)
    except FileNotFoundError as e:
        log.warning(str(e) + '. Skipping...')
        return False

    with ephys.Unit.connection.transaction:
        for probe_no, (f, cluster_method, npx_meta) in clustering_files.items():
            try:
                log.info('------ Start loading clustering results for probe: {} ------'.format(probe_no))
                loader = cluster_loader_map[cluster_method]
                dj.conn().ping()
                _wf_update(loader(sinfo, *f), probe_no, npx_meta, rigpath)
            except (ProbeInsertionError, ClusterMetricError, FileNotFoundError) as e:
                dj.conn().cancel_transaction()  # either successful fix of all probes, or none at all
                if isinstance(e, ProbeInsertionError):
                    log.warning('Probe Insertion Error: \n{}. \nSkipping...'.format(str(e)))
                else:
                    log.warning('Error: {}'.format(str(e)))
                return False

        with dj.config(safemode=False):
            (ephys.UnitCellType & key).delete()

    return True


def _wf_update(data, probe, npx_meta, rigpath):

    sinfo = data['sinfo']
    skey = data['skey']
    method = data['method']
    units = data['units']
    unit_wav = data['unit_wav']  # (unit x channel x sample)
    vmax_unit_site = data['vmax_unit_site']
    trial_start = data['trial_start']
    trial_go = data['trial_go']
    clustering_label = data['clustering_label']

    log.info('-- Start insertions for probe: {} - Clustering method: {} - Label: {}'.format(probe, method,
                                                                                            clustering_label))
    assert method == 'kilosort2'
    assert len(trial_start) == len(trial_go)

    # probe insertion key
    insertion_key = {'subject_id': sinfo['subject_id'],
                     'session': sinfo['session'],
                     'insertion_number': probe}

    # _update() on waveform
    for i, u in enumerate(set(units)):
        wf_chn_idx = np.where(data['ks_channel_map'] == vmax_unit_site[i])[0][0]
        wf = unit_wav[i][wf_chn_idx]
        (ephys.Unit & {**skey, **insertion_key, 'clustering_method': method,
                       'unit': u})._update('waveform', wf)


if __name__ == '__main__':
    update_waveforms()
