import math
from collections import defaultdict
import numpy as np
import scipy.io as scio
import json
import pathlib
from tqdm import tqdm
from datetime import datetime
from pipeline import lab
from pipeline import experiment
from pipeline import ephys
from pipeline import histology
from pipeline import psth

'''

Notes:

  - export includes behavior for trials without ephys data. how to handle?

    if exclude, this means trial indices will be non-contiguous w/r/t database
    if include, this means .mat cell arrays will vary by shape and need
    handling locally.

  - Photostim Data (task_stimulation):

    - Experimental data doesn't contain actual start/end/power times;
      Start is captured per trial with power/duration modelled as session
      parameters. This implies that power+off time in export data are
      synthetic.

'''


def mkfilename(insert_key):
    '''
    create a filename for the given insertion key.
    filename will be of the format map-export_h2o_YYYY-MM-DD_SN_PN.mat

    where:

      - h2o: water restriction number
      - YYYY-MM-DD: session recording date
      - SN: session number for this subject
      - PN: probe number for this session

    '''

    fvars = ((lab.WaterRestriction * experiment.Session
              * ephys.ProbeInsertion) & insert_key).fetch1()

    return 'map-export_{}_{}_{}_{}.mat'.format(
        fvars['water_restriction_number'], str(fvars['session_date']),
        fvars['session'], fvars['insertion_number'])


def export_recording(insert_key, filepath=None):
    '''
    Export a 'recording' (probe specific data + related events) to a file.

    Parameters:

      - insert_key: an ephys.ProbeInsertion.primary_key
        currently: {'subject_id', 'session', 'insertion_number'})

      - filepath: an optional output file path string. If not provided,
        files will be created in the current directory using the 'mkfilename'
        function.
    '''

    if not filepath:
        filepath = mkfilename(insert_key)

    print('exporting {} to {}'.format(insert_key, filepath))

    print('fetching spike/behavior data')

    insertion = (ephys.ProbeInsertion.InsertionLocation & insert_key).fetch1()
    units = (ephys.Unit & insert_key).fetch()

    trial_spikes = (ephys.Unit.TrialSpikes & insert_key).fetch(order_by='trial asc')

    behav = (experiment.BehaviorTrial & insert_key).fetch(order_by='trial asc')

    trials = behav['trial']

    exports = ['neuron_single_units', 'neuron_unit_info', 'behavior_report',
               'behavior_early_report', 'behavior_lick_times',
               'task_trial_type', 'task_stimulation', 'task_cue_time']

    edata = {k: None for k in exports}

    print('reshaping/processing for export')

    # neuron_single_units
    # -------------------

    # [[u0t0.spikes, ..., u0tN.spikes], ..., [uNt0.spikes, ..., uNtN.spikes]]
    print('... neuron_single_units:', end='')

    _su = defaultdict(list)

    ts = trial_spikes[['unit', 'trial', 'spike_times']]

    for u, t in ((u, t) for t in trials for u in units['unit']):
        ud = ts[np.logical_and(ts['unit'] == u, ts['trial'] == t)]
        if ud:
            _su[u].append(ud['spike_times'][0])
        else:
            _su[u].append(np.array([]))

    ndarray_object = np.empty((len(_su.keys()), 1), dtype=np.object)
    for idx, i in enumerate(sorted(_su.keys())):
        ndarray_object[idx, 0] = np.array(_su[i], ndmin=2).T

    edata['neuron_single_units'] = ndarray_object

    print('ok.')

    # neuron_unit_info
    # ----------------
    #
    # [[depth_in_um, cell_type, recording_location] ...]
    print('... neuron_unit_info:', end='')

    dv = insertion['dv_location'] if insertion['dv_location'] else np.nan
    loc = insertion['brain_location_name']

    types = (ephys.UnitCellType & insert_key).fetch()

    _ui = []
    for u in units:

        if u['unit'] in types['unit']:
            typ = types[np.where(types['unit'] == u['unit'])][0]
            _ui.append([u['unit_posy'] + dv, typ['cell_type'], loc])
        else:
            _ui.append([u['unit_posy'] + dv, 'unknown', loc])

    edata['neuron_unit_info'] = np.array(_ui, dtype='O')

    print('ok.')

    # behavior_report
    # ---------------
    print('... behavior_report:', end='')

    behavior_report_map = {'hit': 1, 'miss': 0, 'ignore': 0}  # XXX: ignore ok?
    edata['behavior_report'] = np.array([
        behavior_report_map[i] for i in behav['outcome']])

    print('ok.')

    # behavior_early_report
    # ---------------------
    print('... behavior_early_report:', end='')

    early_report_map = {'early': 1, 'no early': 0}
    edata['behavior_early_report'] = np.array([
        early_report_map[i] for i in behav['early_lick']])

    print('ok.')

    # behavior_touch_times
    # --------------------

    behavior_touch_times = None  # NOQA no data (see ActionEventType())

    # behavior_lick_times
    # -------------------
    print('... behavior_lick_times:', end='')

    _lt = []
    licks = (experiment.ActionEvent() & insert_key
             & "action_event_type in ('left lick', 'right lick')").fetch()

    for t in trials:

        _lt.append([float(i) for i in   # decimal -> float
                    licks[licks['trial'] == t]['action_event_time']]
                   if t in licks['trial'] else [])

    edata['behavior_lick_times'] = np.array(_lt)

    behavior_whisker_angle = None  # NOQA no data
    behavior_whisker_dist2pol = None  # NOQA no data

    print('ok.')

    # task_trial_type
    # ---------------
    print('... task_trial_type:', end='')

    task_trial_type_map = {'left': 'l', 'right': 'r'}
    edata['task_trial_type'] = np.array([
        task_trial_type_map[i] for i in behav['trial_instruction']], dtype='O')

    print('ok.')

    # task_stimulation
    # ----------------
    print('... task_stimulation:', end='')

    _ts = []  # [[power, type, on-time, off-time], ...]

    photostim = (experiment.Photostim() & insert_key).fetch()

    photostim_map = {}
    photostim_dat = {}
    photostim_keys = ['left_alm', 'right_alm', 'both_alm']
    photostim_vals = [1, 2, 6]

    # XXX: we don't detect duplicate presence of photostim_keys in data
    for fk, rk in zip(photostim_keys, photostim_vals):

        i = np.where(photostim['brain_location_name'] == fk)[0][0]
        j = photostim[i]['photo_stim']
        photostim_map[j] = rk
        photostim_dat[j] = photostim[i]

    photostim_ev = (experiment.PhotostimEvent & insert_key).fetch()

    for t in trials:

        if t in photostim_ev['trial']:

            ev = photostim_ev[np.where(photostim_ev['trial'] == t)]
            ps = photostim_map[ev['photo_stim'][0]]
            pd = photostim_dat[ev['photo_stim'][0]]

            _ts.append([float(ev['power']), ps,
                        float(ev['photostim_event_time']),
                        float(ev['photostim_event_time'] + pd['duration'])])

        else:
            _ts.append([0, math.nan, math.nan, math.nan])

    edata['task_stimulation'] = np.array(_ts)

    print('ok.')

    # task_pole_time
    # --------------

    task_pole_time = None  # NOQA no data

    # task_cue_time
    # -------------

    print('... task_cue_time:', end='')

    _tct = (experiment.TrialEvent()
            & {**insert_key, 'trial_event_type': 'go'}).fetch(
                'trial_event_time')

    edata['task_cue_time'] = np.array([float(i) for i in _tct])

    print('ok.')

    # savemat
    # -------

    print('... saving to {}:'.format(filepath), end='')

    scio.savemat(filepath, edata)

    print('ok.')


def write_to_activity_viewer_json(probe_insertion, filepath=None, per_period=False):
    probe_insertion = probe_insertion.proj()
    key = (probe_insertion * lab.WaterRestriction * experiment.Session).proj('session_date', 'water_restriction_number').fetch1()
    uid = f'{key["subject_id"]}({key["water_restriction_number"]})/{datetime.strftime(key["session_date"], "%m-%d-%Y")}({key["session"]})/{key["insertion_number"]}'

    units = (ephys.UnitStat * ephys.Unit * lab.ElectrodeConfig.Electrode
             * histology.ElectrodeCCFPosition.ElectrodePosition
             & probe_insertion & 'unit_quality != "all"').fetch(
        'unit', 'ccf_x', 'ccf_y', 'ccf_z', 'avg_firing_rate', order_by='unit')

    if len(units[0]) == 0:
        print('The units in the specified ProbeInsertion do not have CCF data yet')
        return

    penetration_group = {'id': uid, 'points': []}

    for unit, x, y, z, spk_rate in tqdm(zip(*units)):
        contra_frate, ipsi_frate = (psth.PeriodSelectivity & probe_insertion
                                    & f'unit={unit}' & 'period in ("sample", "delay", "response")').fetch(
            'contra_firing_rate', 'ipsi_firing_rate')

        # (red: #FF0000), (blue: #0000FF)
        if per_period:
            sel_color = ['#FF0000' if i_rate > c_rate else '#0000FF' for c_rate, i_rate in zip(contra_frate, ipsi_frate)]
            radius = [np.mean([c_rate, i_rate]) for c_rate, i_rate in zip(contra_frate, ipsi_frate)]
        else:
            sel_color = ['#FF0000' if ipsi_frate.mean() > contra_frate.mean() else '#0000FF']
            radius = [spk_rate]

        unit_dict = {'id': unit, 'x': x, 'y': y, 'z': z, 'alpha': 0.8,
                     'color': {'t': list(range(len(sel_color))), 'vals': sel_color},
                     'radius': {'t': list(range(len(radius))), 'vals': radius}}

        penetration_group['points'].append(unit_dict)

    if filepath:
        path = pathlib.Path(filepath)
        with open(path, 'w') as fp:
            json.dump(penetration_group, fp, default=str)

    return penetration_group
