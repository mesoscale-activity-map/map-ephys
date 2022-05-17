import pathlib
import numpy as np
import json
from datetime import datetime
from dateutil.tz import tzlocal
from datajoint.errors import DataJointError
import pynwb
from pynwb import NWBFile, NWBHDF5IO

from pipeline import lab, experiment, tracking, ephys, histology, psth, ccf
from pipeline.report import get_wr_sessdatetime

# Some constants to work with
zero_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # no precise time available


def datajoint_to_nwb(session_key):
    """
    Generate one NWBFile object representing all data
     coming from the specified "session_key" (representing one session)
    """
    water_res_num, sess_datetime = get_wr_sessdatetime(session_key)

    session_identifier = f'{water_res_num}_{sess_datetime}_s{session_key["session"]}'

    experiment_description = (experiment.TaskProtocol
                              & (experiment.BehaviorTrial & session_key)).fetch1(
        'task_protocol_description')

    try:
        session_descr = (experiment.SessionComment & session_key).fetch1('session_comment')
    except DataJointError:
        session_descr = ''

    nwbfile = NWBFile(identifier=session_identifier,
                      session_description=session_descr,
                      session_start_time=datetime.strptime(sess_datetime, '%Y%m%d_%H%M%S'),
                      file_create_date=datetime.now(tzlocal()),
                      experimenter=list((experiment.Session & session_key).fetch('username')),
                      data_collection='',
                      institution='Janelia Research Campus',
                      experiment_description=experiment_description,
                      related_publications='',
                      keywords=[])

    # ==================================== SUBJECT ==================================
    subject = (lab.Subject & session_key).fetch1()
    nwbfile.subject = pynwb.file.Subject(
        subject_id=str(subject['subject_id']),
        date_of_birth=datetime.combine(subject['date_of_birth'], zero_time) if subject['date_of_birth'] else None,
        sex=subject['sex'])

    # ==================================== EPHYS ==================================
    # add additional columns to the electrodes table
    electrodes_query = lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode
    for additional_attribute in ['shank', 'shank_col', 'shank_row']:
        nwbfile.add_electrode_column(
            name=electrodes_query.heading.attributes[additional_attribute].name,
            description=electrodes_query.heading.attributes[additional_attribute].comment)

    # add additional columns to the units table
    units_query = (ephys.ProbeInsertion.RecordingSystemSetup
                   * ephys.Unit * ephys.UnitStat
                   * ephys.ClusterMetric * ephys.WaveformMetric
                   & session_key)

    units_omitted_attributes = ['subject_id', 'session', 'insertion_number',
                                'clustering_method', 'unit', 'unit_uid', 'probe_type',
                                'epoch_name_quality_metrics', 'epoch_name_waveform_metrics',
                                'electrode_config_name', 'electrode_group',
                                'electrode', 'waveform']

    for attr in units_query.heading.names:
        if attr in units_omitted_attributes:
            continue
        nwbfile.add_unit_column(
            name=units_query.heading.attributes[attr].name,
            description=units_query.heading.attributes[attr].comment)

    # iterate through curated clusterings and export units data
    for insert_key in (ephys.ProbeInsertion & session_key).fetch('KEY'):
        # ---- Probe Insertion Location ----
        if ephys.ProbeInsertion.InsertionLocation & insert_key:
            insert_location = {
                k: str(v) for k, v in (ephys.ProbeInsertion.InsertionLocation
                                       & insert_key).aggr(
                    ephys.ProbeInsertion.RecordableBrainRegion.proj(
                        ..., brain_region='CONCAT(hemisphere, " ", brain_area)'),
                    ..., brain_regions='GROUP_CONCAT(brain_region SEPARATOR ", ")').fetch1().items()
                if k not in ephys.ProbeInsertion.primary_key}
            insert_location = json.dumps(insert_location)
        else:
            insert_location = 'N/A'

        # ---- Electrode Configuration ----
        electrode_config = (lab.Probe * lab.ProbeType * lab.ElectrodeConfig
                            * ephys.ProbeInsertion & insert_key).fetch1()
        ephys_device_name = f'{electrode_config["probe"]} ({electrode_config["probe_type"]})'
        ephys_device = (nwbfile.get_device(ephys_device_name)
                        if ephys_device_name in nwbfile.devices
                        else nwbfile.create_device(name=ephys_device_name))

        electrode_group = nwbfile.create_electrode_group(
            name=f'{electrode_config["probe"]} {electrode_config["electrode_config_name"]}',
            description=json.dumps(electrode_config, default=str),
            device=ephys_device,
            location=insert_location)

        electrode_query = (lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode
                           & electrode_config)
        electrode_ccf = {e: {'x': float(x), 'y': float(y), 'z': float(z)} for e, x, y, z in zip(
            *(histology.ElectrodeCCFPosition.ElectrodePosition
              & electrode_config).fetch(
                'electrode', 'ccf_x', 'ccf_y', 'ccf_z'))}

        for electrode in electrode_query.fetch(as_dict=True):
            nwbfile.add_electrode(
                id=electrode['electrode'], group=electrode_group,
                filtering='', imp=-1.,
                **electrode_ccf.get(electrode['electrode'], {'x': np.nan, 'y': np.nan, 'z': np.nan}),
                rel_x=electrode['x_coord'], rel_y=electrode['y_coord'], rel_z=np.nan,
                shank=electrode['shank'], shank_col=electrode['shank_col'], shank_row=electrode['shank_row'],
                location=electrode_group.location)

        electrode_df = nwbfile.electrodes.to_dataframe()
        electrode_ind = electrode_df.index[electrode_df.group_name == electrode_group.name]
        # ---- Units ----
        unit_query = units_query & insert_key
        for unit in unit_query.fetch(as_dict=True):
            # make an electrode table region (which electrode(s) is this unit coming from)
            unit['id'] = unit.pop('unit')
            unit['electrodes'] = np.where(electrode_ind == unit.pop('electrode'))[0]
            unit['electrode_group'] = electrode_group
            unit['waveform_mean'] = unit.pop('waveform')
            unit['waveform_sd'] = np.full(1, np.nan)

            for attr in list(unit.keys()):
                if attr in units_omitted_attributes:
                    unit.pop(attr)
                elif unit[attr] is None:
                    unit[attr] = np.nan

            nwbfile.add_unit(**unit)

    # =============================== PHOTO-STIMULATION ===============================
    stim_sites = {}
    photostim_query = (experiment.Photostim & (experiment.PhotostimTrial & session_key))
    if photostim_query:
        for photostim_key in photostim_query.fetch('KEY'):
            photostim = (experiment.Photostim * lab.PhotostimDevice.proj('excitation_wavelength') & photostim_key).fetch1()
            stim_device = (nwbfile.get_device(photostim['photostim_device'])
                        if photostim['photostim_device'] in nwbfile.devices
                        else nwbfile.create_device(name=photostim['photostim_device']))

            stim_site = pynwb.ogen.OptogeneticStimulusSite(
                name=f'{photostim["photostim_device"]}_{photostim["photo_stim"]}',
                device=stim_device,
                excitation_lambda=float(photostim['excitation_wavelength']),
                location=json.dumps([{k: v for k, v in stim_locs.items()
                                    if k not in experiment.Photostim.primary_key}
                                    for stim_locs in (experiment.Photostim.PhotostimLocation
                                                    & photostim_key).fetch(as_dict=True)], default=str),
                description=f'excitation_duration: {photostim["duration"]}')
            nwbfile.add_ogen_site(stim_site)
            stim_sites[photostim['photo_stim']] = stim_site 

    # =============================== TRACKING =============================== 
    if tracking.Tracking & session_key:
        behav_acq = pynwb.behavior.BehavioralTimeSeries(name='BehavioralTimeSeries')
        nwbfile.add_acquisition(behav_acq)

        tracking_devices = (tracking.TrackingDevice & (tracking.Tracking & session_key)).fetch(as_dict=True)

        for trk_device in tracking_devices:
            trk_device_name = trk_device['tracking_device'].replace(' ', '') + '_' + trk_device['tracking_position']
            trk_fs = float(trk_device['sampling_rate'])
            for feature, feature_tbl in tracking.Tracking().tracking_features.items():
                ft_attrs = [n for n in feature_tbl.heading.names if n not in feature_tbl.primary_key]
                if feature_tbl & trk_device & session_key:
                    if feature == 'WhiskerTracking':
                        additional_conditions = [{'whisker_name': n} for n in
                                                 set((feature_tbl & trk_device & session_key).fetch(
                                                     'whisker_name'))]
                    else:
                        additional_conditions = [{}]
                    for r in additional_conditions:
                        samples, start_time, *position_data = (experiment.SessionTrial
                                                               * tracking.Tracking
                                                               * feature_tbl
                                                               & session_key
                                                               & r).fetch(
                            'tracking_samples', 'start_time', *ft_attrs, order_by='trial')

                        tracking_timestamps = np.hstack([np.arange(nsample) / trk_fs + float(trial_start_time)
                                                         for nsample, trial_start_time in zip(samples, start_time)])
                        position_data = np.vstack([np.hstack(d) for d in position_data])

                        behav_ts_name = f'{trk_device_name}_{feature}' + (f'_{r["whisker_name"]}' if r else '')

                        behav_acq.create_timeseries(name=behav_ts_name,
                                                    data=position_data,
                                                    timestamps=tracking_timestamps,
                                                    description=f'Time series for {feature} position: {tuple(ft_attrs)}',
                                                    unit='a.u.',
                                                    conversion=1.0)

    # =============================== BEHAVIOR TRIALS ===============================
    # ---- TrialSet ----
    q_photostim = (experiment.PhotostimEvent
                   * experiment.Photostim & session_key).proj(
        'photostim_event_time', 'power', 'duration')
    q_trial = experiment.SessionTrial * experiment.BehaviorTrial & session_key
    q_trial = q_trial.aggr(
        q_photostim, ...,
        photostim_onset='IFNULL(GROUP_CONCAT(photostim_event_time SEPARATOR ", "), "N/A")',
        photostim_power='IFNULL(GROUP_CONCAT(power SEPARATOR ", "), "N/A")',
        photostim_duration='IFNULL(GROUP_CONCAT(duration SEPARATOR ", "), "N/A")',
        keep_all_rows=True)

    skip_adding_columns = experiment.Session.primary_key 

    if q_trial:
        # Get trial descriptors from TrialSet.Trial and TrialStimInfo
        trial_columns = {tag: {'name': tag,
                               'description': q_trial.heading.attributes[tag].comment}
                         for tag in q_trial.heading.names
                         if tag not in skip_adding_columns + ['start_time', 'stop_time']}

        # Add new table columns to nwb trial-table
        for column in trial_columns.values():
            nwbfile.add_trial_column(**column)

        # Add entries to the trial-table
        for trial in q_trial.fetch(as_dict=True):
            trial['start_time'], trial['stop_time'] = float(trial['start_time']), float(trial['stop_time'])
            nwbfile.add_trial(**{k: v for k, v in trial.items() if k not in skip_adding_columns})

    # =============================== BEHAVIOR TRIALS' EVENTS ===============================

    behavioral_event = pynwb.behavior.BehavioralEvents(name='BehavioralEvents')
    nwbfile.add_acquisition(behavioral_event)

    # ---- behavior events

    q_trial_event = (experiment.TrialEvent * experiment.SessionTrial & session_key).proj(
        'trial_event_type',
        event_start='trial_event_time + start_time',
        event_stop='trial_event_time + start_time + duration')

    for trial_event_type in (experiment.TrialEventType & q_trial_event).fetch('trial_event_type'):
        trial, event_starts, event_stops = (q_trial_event
                                            & {'trial_event_type': trial_event_type}).fetch(
            'trial', 'event_start', 'event_stop', order_by='trial')

        behavioral_event.create_timeseries(
            name=trial_event_type + '_start_times',
            unit='a.u.', conversion=1.0,
            data=np.full_like(event_starts.astype(float), 1),
            timestamps=event_starts.astype(float))

        behavioral_event.create_timeseries(
            name=trial_event_type + '_stop_times',
            unit='a.u.', conversion=1.0,
            data=np.full_like(event_stops.astype(float), 1),
            timestamps=event_stops.astype(float))

    # ---- action events

    q_action_event = (experiment.ActionEvent * experiment.SessionTrial & session_key).proj(
        'action_event_type',
        event_time='action_event_time + start_time')

    for action_event_type in (experiment.ActionEventType & q_action_event).fetch('action_event_type'):
        trial, event_starts = (q_action_event
                               & {'action_event_type': action_event_type}).fetch(
            'trial', 'event_time', order_by='trial')

        behavioral_event.create_timeseries(
            name=action_event_type.replace(' ', '_') + '_times',
            unit='a.u.', conversion=1.0,
            data=np.full_like(event_starts.astype(float), 1),
            timestamps=event_starts.astype(float))

    # ---- photostim events ----

    q_photostim_event = (experiment.PhotostimEvent
                         * experiment.Photostim.proj('duration')
                         * experiment.SessionTrial
                         & session_key).proj(
        'trial', 'power', 'photostim_event_time',
        event_start='photostim_event_time + start_time',
        event_stop='photostim_event_time + start_time + duration')

    trials, event_starts, event_stops, powers, photo_stim = q_photostim_event.fetch(
        'trial', 'event_start', 'event_stop', 'power', 'photo_stim', order_by='trial')

    behavioral_event.create_timeseries(
        name='photostim_start_times', unit='mW', conversion=1.0,
        description='Timestamps of the photo-stimulation and the corresponding powers (in mW) being applied',
        data=powers.astype(float),
        timestamps=event_starts.astype(float),
        control=photo_stim.astype('uint8'), control_description=stim_sites)
    behavioral_event.create_timeseries(
        name='photostim_stop_times', unit='mW', conversion=1.0,
        description='Timestamps of the photo-stimulation being switched off',
        data=np.full_like(event_starts.astype(float), 0),
        timestamps=event_stops.astype(float),
        control=photo_stim.astype('uint8'), control_description=stim_sites)

    return nwbfile


def export_recording(session_keys, output_dir='./', overwrite=False):
    if not isinstance(session_keys, list):
        session_keys = [session_keys]

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for session_key in session_keys:
        nwbfile = datajoint_to_nwb(session_key)
        # Write to .nwb
        save_file_name = ''.join([nwbfile.identifier, '.nwb'])
        output_fp = (output_dir / save_file_name).absolute()
        if overwrite or not output_fp.exists():
            with NWBHDF5IO(output_fp.as_posix(), mode='w') as io:
                io.write(nwbfile)
                print(f'\tWrite NWB 2.0 file: {save_file_name}')
