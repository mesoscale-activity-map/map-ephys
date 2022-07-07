import logging
import pathlib
from datetime import datetime
import re
import numpy as np
import datajoint as dj

from ... import get_schema_name, dict_value_to_hash
from .spike_sorter_loader import SpikeGLXMeta


log = logging.getLogger(__name__)


lab = dj.create_virtual_module('lab', get_schema_name('lab'))
experiment = dj.create_virtual_module('experiment', get_schema_name('experiment'))
ephys = dj.create_virtual_module('ephys', get_schema_name('ephys'))


def gen_probe_insert(sinfo, probe, npx_meta, probe_insertion_exists=False):
    '''
    generate probe insertion for session / probe - for neuropixels recording

    Arguments:

      - sinfo: lab.WaterRestriction * lab.Subject * experiment.Session
      - probe: probe id

    '''

    part_no = npx_meta.probe_SN

    e_config_key = gen_electrode_config(npx_meta)

    # ------ ProbeInsertion ------
    insertion_key = {'subject_id': sinfo['subject_id'],
                     'session': sinfo['session'],
                     'insertion_number': probe}

    if probe_insertion_exists:
        if insertion_key not in ephys.ProbeInsertion.proj():
            raise RuntimeError(f'ProbeInsertion key not present. Expecting: {insertion_key}')
    else:
        # add probe insertion
        log.info('.. creating probe insertion: {}'.format(insertion_key))

        lab.Probe.insert1({'probe': part_no, 'probe_type': e_config_key['probe_type']}, skip_duplicates=True)

        ephys.ProbeInsertion.insert1({**insertion_key,  **e_config_key, 'probe': part_no})

        ephys.ProbeInsertion.RecordingSystemSetup.insert1({**insertion_key, 'sampling_rate': npx_meta.meta['imSampRate']})

    return insertion_key, e_config_key


def gen_electrode_config(npx_meta):
    """
    Generate and insert (if needed) an ElectrodeConfiguration based on the specified neuropixels meta information
    """

    if re.search('(1.0|2.0)', npx_meta.probe_model):
        eg_members = []
        probe_type = {'probe_type': npx_meta.probe_model}
        q_electrodes = lab.ProbeType.Electrode & probe_type
        for shank, shank_col, shank_row, is_used in npx_meta.shankmap['data']:
            electrode = (q_electrodes & {'shank': shank + 1,  # shank is 1-indexed in this pipeline
                                         'shank_col': shank_col + 1,
                                         'shank_row': shank_row + 1}).fetch1('KEY')
            eg_members.append({**electrode, 'is_used': is_used, 'electrode_group': 0})
    else:
        raise NotImplementedError('Processing for neuropixels probe model {} not yet implemented'.format(
            npx_meta.probe_model))

    # ---- compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode) ----
    ec_hash = dict_value_to_hash({k['electrode']: k for k in eg_members})

    el_list = sorted([k['electrode'] for k in eg_members])
    el_jumps = [-1] + np.where(np.diff(el_list) > 1)[0].tolist() + [len(el_list) - 1]
    ec_name = '; '.join([f'{el_list[s + 1]}-{el_list[e]}' for s, e in zip(el_jumps[:-1], el_jumps[1:])])

    e_config = {**probe_type, 'electrode_config_name': ec_name}

    # ---- make new ElectrodeConfig if needed ----
    if not (lab.ElectrodeConfig & {'electrode_config_hash': ec_hash}):

        log.info('.. Probe type: {} - creating lab.ElectrodeConfig: {}'.format(npx_meta.probe_model, ec_name))

        lab.ElectrodeConfig.insert1({**e_config, 'electrode_config_hash': ec_hash})

        lab.ElectrodeConfig.ElectrodeGroup.insert1({**e_config, 'electrode_group': 0})

        lab.ElectrodeConfig.Electrode.insert({**e_config, **m} for m in eg_members)

    return e_config


# ======== Helpers for directory navigation ========

def get_ephys_paths():
    """
    retrieve ephys paths from dj.config
    config should be in dj.config of the format:

      dj.config = {
        ...,
        'custom': {
          'ephys_data_paths': ['/path/string', '/path2/string']
        }
        ...
      }
    """
    return dj.config.get('custom', {}).get('ephys_data_paths', None)


def get_sess_dir(session_key):
    session_info = (experiment.Session & session_key).fetch1()
    sinfo = ((lab.WaterRestriction
              * lab.Subject.proj()
              * experiment.Session.proj(..., '-session_time')) & session_info).fetch1()

    rigpaths = get_ephys_paths()
    h2o = sinfo['water_restriction_number']

    sess_time = (datetime.min + session_info['session_time']).time()
    sess_datetime = datetime.combine(session_info['session_date'], sess_time)

    for rigpath in rigpaths:
        rigpath = pathlib.Path(rigpath)
        subject_dir = rigpath / h2o
        dpath = None  # session directory
        dglob = None  # probe directory pattern
        if (subject_dir / sess_datetime.date().strftime('%Y%m%d')).exists():
            dpath = subject_dir / sess_datetime.date().strftime('%Y%m%d')
            dglob = '[0-9]/{}'
        elif (subject_dir / 'catgt_{}_g0'.format(
                sess_datetime.date().strftime('%Y%m%d'))).exists():
            dpath = subject_dir / 'catgt_{}_g0'.format(
                sess_datetime.date().strftime('%Y%m%d'))
            dglob = '{}_*_imec[0-9]'.format(
                sess_datetime.date().strftime('%Y%m%d')) + '/{}'
        else:
            date_strings = [sess_datetime.date().strftime('%m%d%y'), sess_datetime.date().strftime('%Y%m%d')]
            for date_string in date_strings:
                sess_dirs = list(subject_dir.glob('*{}*{}_*'.format(h2o, date_string)))
                for sess_dir in sess_dirs:
                    try:
                        npx_meta = SpikeGLXMeta(next(sess_dir.rglob('{}_*tcat*.ap.meta'.format(h2o))))
                    except StopIteration:
                        continue
                    # ensuring time difference between behavior-start and ephys-start is no more than 2 minutes - this is to handle multiple sessions in a day
                    start_time_difference = abs((npx_meta.recording_time - sess_datetime).total_seconds())
                    if start_time_difference <= 120:
                        dpath = sess_dir
                        dglob = '{}*{}_*_imec[0-9]'.format(h2o, date_string) + '/{}'  # probe directory pattern
                        break
                    else:
                        log.info('Found {} - difference in behavior and ephys start-time: {} seconds (more than 2 minutes). Skipping...'.format(sess_dir, start_time_difference))

        if dpath is not None:
            break
    else:
        dpath, dglob, rigpath = None, None, None

    if dpath is not None:
        log.info('Found session folder: {}'.format(dpath))
    else:
        log.warning('Error - No session folder found for {}/{}'.format(h2o, session_info['session_date']))

    return dpath, dglob, rigpath


def match_probe_to_ephys(h2o, dpath, dglob):
    """
    Based on the identified spike sorted file(s), match the probe number (i.e. 1, 2, 3) to the cluster filepath, loader, and npx_meta
    Return a dict, e.g.:
    {
     1: (cluster_fp, cluster_method, npx_meta),
     2: (cluster_fp, cluster_method, npx_meta),
     3: (cluster_fp, cluster_method, npx_meta),
    }
    """
    # npx ap.meta: '{}_*.imec.ap.meta'.format(h2o)
    npx_meta_files = list(dpath.glob(dglob.format('*tcat*.ap.meta')))  # search for "tcat" ap.meta files only
    if not npx_meta_files:
        raise FileNotFoundError('Error - no ap.meta files at {}'.format(dpath))

    jrclustv3spec = '{}_*_jrc.mat'.format(h2o)
    jrclustv4spec = '{}_*.ap_res.mat'.format(h2o)
    ks2specs = ('mean_waveforms.npy', 'spike_times.npy')  # prioritize QC output, then orig

    clustered_probes = {}
    for meta_file in npx_meta_files:
        probe_dir = meta_file.parent
        probe_number = re.search('(imec)?\d{1}$', probe_dir.name).group()
        probe_number = int(probe_number.replace('imec', '')) + 1 if 'imec' in probe_number else int(probe_number)

        # JRClust v3
        v3files = [((f, ), 'jrclust_v3') for f in probe_dir.glob(jrclustv3spec)]
        # JRClust v4
        v4files = [((f, ), 'jrclust_v4') for f in probe_dir.glob(jrclustv4spec)]
        # Kilosort
        ks2spec = ks2specs[0] if len(list(probe_dir.rglob(ks2specs[0]))) > 0 else ks2specs[1]
        ks2files = [((f.parent, probe_dir), 'kilosort2') for f in probe_dir.rglob(ks2spec)]

        if len(ks2files) > 1:
            raise ValueError('Multiple Kilosort outputs found at: {}'.format([str(x[0]) for x in ks2files]))

        clustering_results = v4files + v3files + ks2files

        if len(clustering_results) < 1:
            raise FileNotFoundError('Error - No clustering results found at {}'.format(probe_dir))
        elif len(clustering_results) > 1:
            log.warning('Found multiple clustering results at {probe_dir}.'
                        ' Prioritize JRC4 > JRC3 > KS2'.format(probe_dir))

        fp, loader = clustering_results[0]
        clustered_probes[probe_number] = (fp, loader, SpikeGLXMeta(meta_file))

    return clustered_probes
