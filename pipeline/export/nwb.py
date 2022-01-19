import datajoint as dj
import pathlib
import numpy as np
import json
from datetime import datetime
from dateutil.tz import tzlocal
from decimal import Decimal
from datajoint.errors import DataJointError
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
    except DataJointError:
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

    if tracking.Tracking & session_key:
        behav_acq = pynwb.behavior.BehavioralTimeSeries(name='BehavioralTimeSeries')
        nwbfile.add_acquisition(behav_acq)

        for table, name in tables:
            if table & session_key:
                x, y, likelihood, samples, start_time, sampling_rate= (experiment.SessionTrial 
                                                                    * tracking.Tracking 
                                                                    * table 
                                                                    * tracking.TrackingDevice.proj('sampling_rate') 
                                                                    & session_key).fetch(f'{name}_x',f'{name}_y', 
                                                                                            f'{name}_likelihood', 
                                                                                            'tracking_samples','start_time',
                                                                                            'sampling_rate',order_by='trial')

                shifted_time_stamps = [np.arange(sample) / fs + trial_start_time for sample, fs, trial_start_time in zip(samples, sampling_rate, start_time)]

                behav_acq.create_timeseries(name=f'BehavioralTimeSeries_{name}', 
                                            data=np.vstack([np.hstack(x),np.hstack(y), np.hstack(likelihood)]),
                                            timestamps=np.hstack(shifted_time_stamps),
                                            description=f'Time series for {name} position: x, y, likelihood',
                                            unit='a.u.', 
                                            conversion=1.0)

        # =============================== BEHAVIOR TRIALS ===============================

        # =============== TrialSet ====================

    q_photostim = (experiment.PhotostimEvent 
                * experiment.Photostim  & session_key).proj('photostim_event_time', 'power','duration')
    q_trial = experiment.SessionTrial * experiment.BehaviorTrial * experiment.TrialNote & session_key
    q_trial = q_trial.aggr(q_photostim, ..., photostim_onset='IFNULL(GROUP_CONCAT(photostim_event_time SEPARATOR ", "), "N/A")',
                            photostim_power='IFNULL(GROUP_CONCAT(power SEPARATOR ", "), "N/A")',
                            photostim_duration='IFNULL(GROUP_CONCAT(duration SEPARATOR ", "), "N/A")', keep_all_rows=True)
    q_trial
    
    skip_adding_columns = experiment.Session.primary_key 

    if q_trial:
            # Get trial descriptors from TrialSet.Trial and TrialStimInfo
        trial_columns = {tag: {'name': tag,
                                'description': q_trial.heading.attributes[tag].comment}
                            for tag in q_trial.heading.names
                            if tag not in skip_adding_columns + ['start_time','stop_time']}

        # Add new table columns to nwb trial-table
        for column in trial_columns.values():
            nwbfile.add_trial_column(**column)

        # Add entries to the trial-table
        for trial in q_trial.fetch(as_dict=True):
            trial['start_time'], trial['stop_time'] = float(trial['start_time']), float(trial['stop_time'])
            nwbfile.add_trial(**{k:v for k,v in trial.items() if k not in skip_adding_columns})

# # # # =============================== TRIAL EVENTS ==========================
            
    behavioral_event = pynwb.behavior.BehavioralEvents(name='BehavioralEvents')
    nwbfile.add_acquisition(behavioral_event)

    # ---- behavior events

    q_trial_event = (experiment.TrialEvent & session_key).proj('trial_event_type','trial_event_time',event_stop='trial_event_time + duration')

    for trial_event_type in (experiment.TrialEventType & q_trial_event).fetch('trial_event_type'):
        trial, event_starts, event_stops = (q_trial_event & {'trial_event_type':trial_event_type}).fetch('trial',
                                                                                                        'trial_event_time','event_stop', order_by='trial')

        trial_start_times = (experiment.SessionTrial * q_trial_event & {'trial_event_type':trial_event_type}).fetch('start_time', order_by='trial')

        behavioral_event.create_timeseries(name=trial_event_type + '_start_times', unit='a.u.', conversion=1.0,
                                data=np.full_like(event_starts.astype(float), 1),
                                timestamps=event_starts.astype(float) + trial_start_times.astype(float))

        behavioral_event.create_timeseries(name=trial_event_type + '_stop_times', unit='a.u.', conversion=1.0,
                            data=np.full_like(event_stops.astype(float), 1),
                            timestamps=event_stops.astype(float) + trial_start_times.astype(float))

    # ---- photostim events ----

    q_photostim_event = (experiment.PhotostimEvent * experiment.Photostim.proj('duration') & session_key).proj('trial','power','photostim_event_time', event_stop='photostim_event_time+duration',)
    trials, event_starts, event_stops, powers, photo_stim = q_photostim_event.fetch(
        'trial', 'photostim_event_time', 'event_stop', 'power', 'photo_stim', order_by='trial')
    trial_starts = (experiment.SessionTrial() & q_photostim_event).fetch('start_time')
    behavioral_event.create_timeseries(name='photostim_start_times', unit='mW', conversion=1.0,
                                description='Timestamps of the photo-stimulation and the corresponding powers (in mW) being applied',
                                data=powers.astype(float),
                                timestamps=event_starts.astype(float) + trial_starts.astype(float),
                                control=photo_stim.astype('uint8'), control_description=stim_sites)
    behavioral_event.create_timeseries(name='photostim_stop_times', unit='mW', conversion=1.0,
                                description = 'Timestamps of the photo-stimulation being switched off',
                                data=np.full_like(event_starts.astype(float), 0),
                                timestamps=event_stops.astype(float) + trial_starts.astype(float),
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
