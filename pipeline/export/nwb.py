import datajoint as dj
import pathlib
import numpy as np
import json
from datetime import datetime
from dateutil.tz import tzlocal
from decimal import Decimal
import pynwb
from pynwb import NWBFile, NWBHDF5IO

from pipeline import lab, experiment, tracking, ephys, histology, psth, ccf
from pipeline.util import _get_clustering_method
from pipeline.report import get_wr_sessdate

# Some constants to work with
zero_time = datetime.strptime('00:00:00', '%H:%M:%S').time()  # no precise time available

def datajoint_to_nwb(session_key):
    """
    Generate one NWBFile object representing all data
     coming from the specified "session_key" (representing one session)
    """
    session_key = (experiment.Session.proj(
        session_datetime="cast(concat(session_date, ' ', session_time) as datetime)")
                   & session_key).fetch1()

    session_identifier = f'{session_key["subject_id"]}' \
                         f'_{session_key["session"]}' \
                         f'_{session_key["session_datetime"].strftime("%Y%m%d_%H%M%S")}'

    experiment_description = (experiment.TaskProtocol
                              & (experiment.BehaviorTrial & session_key)).fetch1(
        'task_protocol_description')

    try:
        session_descr = (experiment.SessionComment & session_key).fetch1('session_comment')
    except:
        session_descr = ' '

    nwbfile = NWBFile(identifier=session_identifier,
                      session_description=(session_descr),
                      session_start_time=session_key['session_datetime'],
                      file_create_date=datetime.now(tzlocal()),
                      experimenter=list((experiment.Session & session_key).fetch('username')),
                      data_collection='',
                      institution='Janelia Research Campus',
                      experiment_description=experiment_description,
                      related_publications='',
                      keywords=[])

     # ---- Subject ----
    subject = (lab.Subject & session_key).fetch1()
    nwbfile.subject = pynwb.file.Subject(
        subject_id=str(subject['subject_id']),
        date_of_birth=datetime.combine(subject['date_of_birth'], zero_time) if subject['date_of_birth'] else None,
        sex=subject['sex'])

    # add additional columns to the electrodes table
    # electrodes_query = lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode
    # for additional_attribute in ['shank', 'shank_col', 'shank_row']:
    #     nwbfile.add_electrode_column(
    #         name=electrodes_query.heading.attributes[additional_attribute].name,
    #         description=electrodes_query.heading.attributes[additional_attribute].comment)

    # nwbfile.add_electrode_column(
    #     name="id_in_probe", description="electrode id within the probe",
    # )
    # # add additional columns to the units table
    # units_query = (ephys.ProbeInsertion.RecordingSystemSetup
    #                * ephys.Unit * ephys.UnitStat
    #                * ephys.ClusterMetric * ephys.WaveformMetric
    #                & session_key)

    # units_omitted_attributes = ['subject_id', 'session', 'insertion_number',
    #                             'clustering_method', 'unit', 'unit_uid', 'probe_type',
    #                             'epoch_name_quality_metrics', 'epoch_name_waveform_metrics',
    #                             'electrode_config_name', 'electrode_group',
    #                             'electrode', 'waveform']

    # for attr in units_query.heading.names:
    #     if attr in units_omitted_attributes:
    #         continue
    #     nwbfile.add_unit_column(
    #         name=units_query.heading.attributes[attr].name,
    #         description=units_query.heading.attributes[attr].comment)

    # # iterate through curated clusterings and export units data
    # for insert_key in (ephys.ProbeInsertion & session_key).fetch('KEY'):
    #     # ---- Probe Insertion Location ----
    #     if ephys.ProbeInsertion.InsertionLocation & insert_key:
    #         insert_location = {
    #             k: str(v) for k, v in (ephys.ProbeInsertion.InsertionLocation
    #                                    & insert_key).aggr(
    #                 ephys.ProbeInsertion.RecordableBrainRegion.proj(
    #                     ..., brain_region='CONCAT(hemisphere, " ", brain_area)'),
    #                 ..., brain_regions='GROUP_CONCAT(brain_region SEPARATOR ", ")').fetch1().items()
    #             if k not in ephys.ProbeInsertion.primary_key}
    #         insert_location = json.dumps(insert_location)
    #     else:
    #         insert_location = 'N/A'

    #     # ---- Electrode Configuration ----
    #     electrode_config = (lab.Probe * lab.ProbeType * lab.ElectrodeConfig
    #                         * ephys.ProbeInsertion & insert_key).fetch1()
    #     ephys_device_name = f'{electrode_config["probe"]} ({electrode_config["probe_type"]})'
    #     ephys_device = (nwbfile.get_device(ephys_device_name)
    #                     if ephys_device_name in nwbfile.devices
    #                     else nwbfile.create_device(name=ephys_device_name))

    #     electrode_group = nwbfile.create_electrode_group(
    #         name=f'{electrode_config["probe"]} {electrode_config["electrode_config_name"]}',
    #         description=json.dumps(electrode_config, default=str),
    #         device=ephys_device,
    #         location=insert_location)

    #     electrode_query = (lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode
    #                        & electrode_config)
    #     for electrode in electrode_query.fetch(as_dict=True):
    #         nwbfile.add_electrode(
    #             id_in_probe=electrode['electrode'], group=electrode_group,
    #             filtering='', imp=-1.,
    #             x=np.nan, y=np.nan, z=np.nan,
    #             rel_x=electrode['x_coord'], rel_y=electrode['y_coord'], rel_z=np.nan,
    #             shank=electrode['shank'], shank_col=electrode['shank_col'], shank_row=electrode['shank_row'],
    #             location=electrode_group.location)

    #     electrode_df = nwbfile.electrodes.to_dataframe()
    #     electrode_ind = electrode_df.index[electrode_df.group_name == electrode_group.name]
    #     # ---- Units ----
    #     unit_query = units_query & insert_key
    #     for unit in unit_query.fetch(as_dict=True):
    #         # make an electrode table region (which electrode(s) is this unit coming from)
    #         unit['id'] = unit.pop('unit')
    #         unit['electrodes'] = np.where(electrode_ind == unit.pop('electrode'))[0]
    #         unit['electrode_group'] = electrode_group
    #         unit['waveform_mean'] = unit.pop('waveform')
    #         unit['waveform_sd'] = np.full(1, np.nan)

    #         for attr in list(unit.keys()):
    #             if attr in units_omitted_attributes:
    #                 unit.pop(attr)
    #             elif unit[attr] is None:
    #                 unit[attr] = np.nan

    #         nwbfile.add_unit(**unit)
    #         pass
        
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
    # add tracking device 
    tables = [(tracking.Tracking.NoseTracking,'nose'),(tracking.Tracking.TongueTracking,'tongue'),
            (tracking.Tracking.JawTracking,'jaw'),(tracking.Tracking.LeftPawTracking,'left_paw'),
            (tracking.Tracking.RightPawTracking,'right_paw'),(tracking.Tracking.LickPortTracking,'lickport'),
            (tracking.Tracking.WhiskerTracking,'whisker')]
    
    time_stamps, event_start_times = [], []

    if tracking.Tracking & session_key:
        behav_acq = pynwb.behavior.BehavioralTimeSeries(name='BehavioralTimeSeries')
        nwbfile.add_acquisition(behav_acq)
        for table in tables:
            if table[0] & session_key:
                for tracking_device in (tracking.TrackingDevice & (table[0] & session_key)).fetch('KEY'):
                    trial_event_id, trial_event_times, x, y,likelihood, tracking_samples = ((tracking.Tracking 
                                                                                            * table[0]
                                                                                            * experiment.TrialEvent)
                                                                                            & tracking_device
                                                                                            & session_key).fetch(
                                                                                                                'trial_event_id', 'trial_event_time',f'{table[1]}_x', 
                                                                                                                f'{table[1]}_y',f'{table[1]}_likelihood', 'tracking_samples',
                                                                                                                'duration') 
                    # calculating start time for each trial_event_id relative to session
                    all_start_times = (experiment.TrialEvent & session_key).fetch('duration')
                    time_1 = (experiment.TrialEvent & session_key).fetch('trial_event_time', limit=1)
                    all_start_times = np.insert(all_start_times, 0, time_1[0])
                    all_start_times = np.cumsum(all_start_times)
                    all_start_times = np.delete(all_start_times,-1)
                    all_start_times = all_start_times.astype(float)

                    # getting the tracking time start times from all_durations
                    for idx, time in enumerate(all_start_times):
                        for trial in trial_event_id:
                            if trial == idx:
                                event_start_times = np.append(event_start_times, time)

                    # time stamps for each trial_event
                    for idx, time in enumerate(event_start_times):
                       while idx+1 <= len(event_start_times):
                            time_stamp = np.linspace(time, event_start_times[idx+1], tracking_samples[idx])
                            time_stamps = np.append(time_stamps, time_stamp)

                    behav_acq.create_timeseries(name=f'BehavioralTimeSeries_{table[0]}_x_y_data', 
                                                data=np.c_[x, y],
                                                timestamps=np.hstack(time_stamps),
                                                description='video description',
                                                unit='a.u.', 
                                                conversion=1.0)
                    behav_acq.create_timeseries(name=f'BehavioralTimeSeries_{table[0]}_likelihood', 
                            data=likelihood,
                            timestamps=np.hstack(time_stamps),
                            description='video description',
                            unit='a.u.', 
                            conversion=1.0)


    # =============================== BEHAVIOR TRIALS ===============================

    # =============== TrialSet ====================

    q_photostim = ((experiment.TrialEvent & 'trial_event_type="go"').proj('trial_event_time') 
                    * experiment.PhotostimEvent.proj(photostim_event_time='photostim_event_time',power='power') 
                    * experiment.Photostim.proj(stim_dur='duration') 
                    & session_key).proj(
                        'power','photostim_event_time','stim_dur',
                         stim_time='ROUND(trial_event_time - photostim_event_time, 2)')

    q_trial = experiment.SessionTrial * experiment.BehaviorTrial * experiment.TrialNote & session_key
    q_trial_aggr = q_trial.aggr(q_photostim, ..., photostim_onset='IFNULL(GROUP_CONCAT(stim_time SEPARATOR ", "), "N/A")',
                            photostim_power='IFNULL(GROUP_CONCAT(power SEPARATOR ", "), "N/A")',
                            photostim_duration='IFNULL(GROUP_CONCAT(stim_dur SEPARATOR ", "), "N/A")', keep_all_rows=True)

    skip_adding_columns = experiment.Session.primary_key 

    if q_trial_aggr:
            # Get trial descriptors from TrialSet.Trial and TrialStimInfo
        trial_columns = {tag: {'name': tag,
                                'description': q_trial_aggr.heading.attributes[tag].comment}
                            for tag in q_trial_aggr.heading.names
                            if tag not in skip_adding_columns + ['start_time','stop_time']}

        # Add new table columns to nwb trial-table
        for c in trial_columns.values():
            nwbfile.add_trial_column(**c)

        # Add entries to the trial-table
        for trial in q_trial_aggr.fetch(as_dict=True):
            trial['start_time'], trial['stop_time'] = float(trial['start_time']), float(trial['stop_time'])
            [trial.pop(k) for k in skip_adding_columns]
            nwbfile.add_trial(**trial)

# # # =============================== TRIAL EVENTS ==========================
            
    behav_event = pynwb.behavior.BehavioralEvents(name='BehavioralEvents')
    nwbfile.add_acquisition(behav_event)

    all_session_times = experiment.TrialEvent * experiment.SessionTrial  & session_key  

    # ---- photostim events ----
    q_photostim_event = (experiment.TrialEvent * experiment.PhotostimEvent & session_key).proj(
        event_start='trial_event_time', event_stop = '(trial_event_time+duration)', 
        power= 'power', photostim='photo_stim')

    if q_photostim_event:
        trial_event_id, event_starts, event_stops, powers, photo_stim = q_photostim_event.fetch(
            'trial_event_id','event_start', 'event_stop', 'power', 'photostim', order_by='trial')
        
        all_times = [event_starts[0]]

        for time in all_session_times.fetch('duration'):
            all_times.append(time)
            all_times_arr = np.cumsum(all_times)

        all_times_arr = np.delete(all_times_arr, -1)
        trial_start_times = []

        for idx, times in enumerate(all_times_arr):
            for trial in trial_event_id:
                if trial == idx:
                    trial_start_times.append(times)

                
        behav_event.create_timeseries(name='photostim_start_times', unit='mW', conversion=1.0,
                                    description='Timestamps of the photo-stimulation and the corresponding powers (in mW) being applied',
                                    data=powers.astype(float),
                                    timestamps=trial_start_times,
                                    control=photo_stim.astype('uint8'), control_description=stim_sites)
        behav_event.create_timeseries(name='photostim_stop_times', unit='mW', conversion=1.0,
                                    description = 'Timestamps of the photo-stimulation being switched off',
                                    data=np.full_like(event_starts.astype(float), 0),
                                    timestamps=event_stops.astype(Decimal) + trial_start_times,
                                    control=photo_stim.astype('uint8'), control_description=stim_sites)

    # ---- behavior events ----

    q_behavior_trials = ((experiment.TrialEvent * experiment.BehaviorTrial) - experiment.PhotostimEvent 
                        & session_key).proj(event_start='trial_event_time', event_stop = '(trial_event_time+duration)',
                            outcome='outcome',trial_instruction='trial_instruction')

    if q_behavior_trials:
        trial_event_id, event_starts, event_stops, powers, photo_stim = q_behavior_trials.fetch(
            'trial_event_id','event_start', 'event_stop', 'outcome', 'trial_instruction', order_by='trial')
 
        all_times = [event_starts[0]]

        for time in all_session_times.fetch('duration'):
            all_times.append(time)
            all_times_arr = np.cumsum(all_times)

        all_times_arr = np.delete(all_times_arr, -1)

        trial_start_times = []

        for idx, times in enumerate(all_times_arr):
            for trial in trial_event_id:
                if trial == idx:
                    trial_start_times.append(times)

        behav_event.create_timeseries(name='behavior_event_start_times', unit='a.u', conversion=1.0,
                                    description='Behavior event start times',
                                    data=np.full_like(event_starts.astype(float), 0),
                                    timestamps=trial_start_times)
        behav_event.create_timeseries(name='behavior_event_stop_times', unit='a.u', conversion=1.0,
                                    description = 'Behavior event stop times',
                                    data=np.full_like(event_stops.astype(float), 0),
                                    timestamps=event_stops.astype(Decimal) + trial_start_times)
        
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
