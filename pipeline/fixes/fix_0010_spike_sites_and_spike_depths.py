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
                                   get_sess_dir, match_probe_to_ephys, gen_probe_insert)
from pipeline.fixes import schema, FixHistory


os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

log = logging.getLogger(__name__)


@schema
class AddSpikeSitesAndDepths(dj.Manual):
    """
    This table accompanies fix_0010, keeping track of the units with spike sites and spike depths added.
    This tracking is not extremely important, as a rerun of this fix_0010 will still yield the correct results
    """
    definition = """ # This table accompanies fix_0010
    -> FixHistory
    -> ephys.Unit
    ---
    fixed: bool
    """


def update_spike_sites_and_depths(session_keys={}):
    """
    Updating unit-waveform, where a unit's waveform is updated to be from the peak-channel
        and not from the 1-st channel as before
    Applicable to only kilosort2 clustering results and not jrclust
    """
    sessions_2_update = experiment.Session & ephys.Unit & session_keys
    sessions_2_update = sessions_2_update - AddSpikeSitesAndDepths

    if not sessions_2_update:
        return

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}

    for key in sessions_2_update.fetch('KEY'):
        success = _update_one_session(key)
        if success:
            FixHistory.insert1(fix_hist_key, skip_duplicates=True)
            AddSpikeSitesAndDepths.insert([{**fix_hist_key, **ukey, 'fixed': 1} for ukey in (ephys.Unit & key).fetch('KEY')])


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
        dpath, dglob = get_sess_dir(rigpath, h2o, sess_datetime)
        if dpath is not None:
            break

    if dpath is not None:
        log.info('Found session folder: {}'.format(dpath))
    else:
        log.warning('Error - No session folder found for {}/{}. Skipping...'.format(h2o, key['session_date']))
        return False

    try:
        clustering_files = match_probe_to_ephys(h2o, dpath, dglob)
    except FileNotFoundError as e:
        log.warning(str(e) + '. Skipping...')
        return False

    with ephys.Unit.connection.transaction:
        for probe_no, (f, cluster_method, npx_meta) in clustering_files.items():
            try:
                log.info('------ Start loading clustering results for probe: {} ------'.format(probe_no))
                loader = cluster_loader_map[cluster_method]
                dj.conn().ping()
                _add_spike_sites_and_depths(loader(sinfo, *f), probe_no, npx_meta, rigpath)
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


def _add_spike_sites_and_depths(data, probe, npx_meta, rigpath):

    sinfo = data['sinfo']
    skey = data['skey']
    method = data['method']
    spikes = data['spikes']
    units = data['units']
    spike_sites = data['spike_sites']
    spike_depths = data['spike_depths']

    clustering_label = data['clustering_label']

    log.info('-- Start insertions for probe: {} - Clustering method: {} - Label: {}'.format(probe, method,
                                                                                            clustering_label))

    # probe insertion key
    insertion_key, e_config_key = gen_probe_insert(sinfo, probe, npx_meta, probe_insertion_exists=True)

    # remove noise clusters
    if method in ['jrclust_v3', 'jrclust_v4']:
        units, spikes, spike_sites, spike_depths = (v[i] for v, i in zip(
            (units, spikes, spike_sites, spike_depths), repeat((units > 0))))

    q_electrodes = lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode & e_config_key
    site2electrode_map = {}
    for recorded_site in np.unique(spike_sites):
        shank, shank_col, shank_row, _ = npx_meta.shankmap['data'][recorded_site - 1]  # subtract 1 because npx_meta shankmap is 0-indexed
        site2electrode_map[recorded_site] = (q_electrodes
                                             & {'shank': shank + 1,  # this is a 1-indexed pipeline
                                                'shank_col': shank_col + 1,
                                                'shank_row': shank_row + 1}).fetch1('KEY')

    spike_sites = np.array([site2electrode_map[s]['electrode'] for s in spike_sites])
    unit_spike_sites = np.array([spike_sites[np.where(units == u)] for u in set(units)])
    unit_spike_depths = np.array([spike_depths[np.where(units == u)] for u in set(units)])

    # _update() on spike_sites and spike_depths
    for i, u in enumerate(set(units)):
        (ephys.Unit & {**skey, **insertion_key, 'clustering_method': method,
                       'unit': u})._update('spike_sites', unit_spike_sites[i])
        (ephys.Unit & {**skey, **insertion_key, 'clustering_method': method,
                       'unit': u})._update('spike_depths', unit_spike_depths[i])


if __name__ == '__main__':
    update_spike_sites_and_depths()
