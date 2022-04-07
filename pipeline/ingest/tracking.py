
import os
import logging
import pathlib
from glob import glob
from datetime import datetime
import uuid

import numpy as np
import datajoint as dj

from pipeline import lab
from pipeline import tracking
from pipeline import experiment
from pipeline.ingest import behavior as behavior_ingest
from collections import defaultdict
from .. import get_schema_name

schema = dj.schema(get_schema_name('ingest_tracking'))

log = logging.getLogger(__name__)

[behavior_ingest]  # NOQA schema only use


def get_tracking_paths():
    """
    retrieve behavior rig paths from dj.config
    config should be in dj.config of the format:

      dj.config = {
        ...,
        'custom': {
        "tracking_data_paths":
            [
                ["RRig", "/path/string"]
            ]
        }
        ...
      }
    """
    return dj.config.get('custom', {}).get('tracking_data_paths', None)


@schema
class TrackingIngest(dj.Imported):
    definition = """
    -> experiment.Session
    """

    class TrackingFile(dj.Part):
        definition = '''
        -> TrackingIngest
        -> experiment.SessionTrial
        -> tracking.TrackingDevice
        ---
        tracking_file:          varchar(255)            # tracking file subpath
        '''

    key_source = experiment.Session - tracking.Tracking

    camera_position_mapper = {'side': ('side', 'side_face'),
                              'bottom': ('bottom', 'bottom_face'),
                              'body': ('body', 'side_body')}

    def make(self, key, tracking_exists=False):
        '''
        TrackingIngest .make() function
        To rerun the tracking ingestion for an existing session to add new tracking data for new devices:
            TrackingIngest().make(session_key, tracking_exists=True)
        '''
        log.info('\n==================================================================')
        log.info('TrackingIngest().make(): key: {k}'.format(k=key))

        session = (lab.WaterRestriction.proj('water_restriction_number') * experiment.Session.proj(
            ..., session_datetime="cast(concat(session_date, ' ', session_time) as datetime)") & key).fetch1()

        h2o = session['water_restriction_number']
        trials = (experiment.SessionTrial & session).fetch('trial')
        session_rig = (experiment.Session & key).fetch1('rig')

        log.info('\tgot session: {} - {} trials - {}'.format(session, len(trials), session_rig))

        # Camera 3, 4, 5 are for multi-target-licking task - with RRig-MTL
        camera_restriction = ('tracking_device in ("Camera 3", "Camera 4", "Camera 5")'
                              if session_rig == 'RRig-MTL'
                              else 'tracking_device in ("Camera 0", "Camera 1", "Camera 2")')

        tracking_files = []
        for device in (tracking.TrackingDevice & camera_restriction).fetch(as_dict=True):
            tdev = device['tracking_device']
            cam_pos = device['tracking_position']

            if tracking_exists and (tracking.Tracking & key & device):
                log.info('\nSession tracking exists for: {} ({}). Skipping...'.format(tdev, cam_pos))
                continue

            log.info('\n-------------------------------------')
            log.info('\nSearching for session tracking data directory for: {} ({})'.format(tdev, cam_pos))

            for tracking_path in get_tracking_paths():
                tracking_root_dir = tracking_path[-1]
                try:
                    if session_rig == 'RRig-MTL':
                        tracking_sess_dir, sdate_sml = _get_MTL_sess_tracking_dir(
                            tracking_root_dir, session, cam_pos)
                    else:
                        tracking_sess_dir, sdate_sml = _get_sess_tracking_dir(
                            tracking_root_dir, session)
                except FileNotFoundError as e:
                    log.warning('{} - skipping'.format(str(e)))
                    continue
                else:
                    break
            else:
                log.warning('\tNo tracking directory found for {} ({}) - skipping...'.format(tdev, cam_pos))
                continue

            campath = None
            tpos = None
            for tpos_candidate in self.camera_position_mapper[cam_pos]:
                camtrial_fn = '{}_{}_{}.txt'.format(h2o, sdate_sml, tpos_candidate)
                log.info('Trying camera position trial map: {}'.format(tracking_sess_dir / camtrial_fn))
                if (tracking_sess_dir / camtrial_fn).exists():
                    campath = tracking_sess_dir / camtrial_fn
                    tpos = tpos_candidate
                    log.info('Matched! Using "{}"'.format(tpos))
                    break

            csv_file_ending = '' if session_rig == 'RRig-MTL' else '-*'  # csv filename containing '-0000' or not

            if campath is None:
                log.info('Video-Trial mapper file (.txt) not found - Using one-to-one trial mapping')
                tmap = {tr - (1 if session_rig == 'RRig-MTL' else 0): tr for tr in trials}  # one-to-one map
                for tpos_candidate in self.camera_position_mapper[cam_pos]:
                    camtrial_fn = '{}*_{}_[0-9]*{}.csv'.format(h2o, tpos_candidate, csv_file_ending)
                    log.info('Trying camera position trial map: {}'.format(tracking_sess_dir / camtrial_fn))
                    if list(tracking_sess_dir.glob(camtrial_fn)):
                        tpos = tpos_candidate
                        log.info('Matched! Using "{}"'.format(tpos))
                        break
            else:
                tmap = self.load_campath(campath)  # file:trial

            if tpos is None:
                log.warning('No tracking data for camera: {}... skipping'.format(cam_pos))
                continue

            n_tmap = len(tmap)

            # sanity check
            assert len(trials) >= n_tmap, '{} tracking trials found but only {} behavior trials available'.format(n_tmap, len(trials))

            log.info('loading tracking data for {} trials'.format(n_tmap))

            i = 0
            for t in tmap:  # load tracking for trial
                if tmap[t] not in trials:
                    log.warning('nonexistant trial {}.. skipping'.format(t))
                    continue

                i += 1

                # ex: dl59_side_1(-0000).csv
                tracking_trial_filename = '{}*_{}_{}{}.csv'.format(h2o, tpos, t, csv_file_ending)
                tracking_trial_filepath = list(tracking_sess_dir.glob(tracking_trial_filename))

                if not tracking_trial_filepath or len(tracking_trial_filepath) > 1:
                    log.debug('file mismatch: file: {} trial: {} ({})'.format(
                        t, tmap[t], tracking_trial_filepath))
                    continue

                if i % 50 == 0:
                    log.info('item {}/{}, trial #{} ({:.2f}%)'
                             .format(i, n_tmap, t, (i/n_tmap)*100))
                else:
                    log.debug('item {}/{}, trial #{} ({:.2f}%)'
                              .format(i, n_tmap, t, (i/n_tmap)*100))

                tracking_trial_filepath = tracking_trial_filepath[-1]
                try:
                    trk = self.load_tracking(tracking_trial_filepath)
                except Exception as e:
                    log.warning('Error loading .csv: {}\n{}'.format(
                        tracking_trial_filepath, str(e)))
                    raise e

                recs = {}
                rec_base = dict(key, trial=tmap[t], tracking_device=tdev)

                for k in trk:
                    if k == 'samples':
                        recs['tracking'] = {
                            **rec_base,
                            'tracking_samples': len(trk['samples']['ts']),
                        }
                    else:
                        rec = dict(rec_base)

                        for attr in trk[k]:
                            rec_key = '{}_{}'.format(k, attr)
                            rec[rec_key] = np.array(trk[k][attr])

                        recs[k] = rec

                tracking.Tracking.insert1(
                    recs['tracking'], allow_direct_insert=True)

                if 'nose' in recs:
                    tracking.Tracking.NoseTracking.insert1(
                        recs['nose'], allow_direct_insert=True)

                if 'tongue' in recs:
                    tracking.Tracking.TongueTracking.insert1(
                        recs['tongue'], allow_direct_insert=True)

                if 'jaw' in recs:
                    tracking.Tracking.JawTracking.insert1(
                        recs['jaw'], allow_direct_insert=True)

                if 'paw_left' in recs:
                    fmap = {'paw_left_x': 'left_paw_x',  # remap field names
                            'paw_left_y': 'left_paw_y',
                            'paw_left_likelihood': 'left_paw_likelihood'}

                    tracking.Tracking.LeftPawTracking.insert1({
                        **{k: v for k, v in recs['paw_left'].items()
                           if k not in fmap},
                        **{fmap[k]: v for k, v in recs['paw_left'].items()
                           if k in fmap}}, allow_direct_insert=True)

                if 'paw_right' in recs:
                    fmap = {'paw_right_x': 'right_paw_x',  # remap field names
                            'paw_right_y': 'right_paw_y',
                            'paw_right_likelihood': 'right_paw_likelihood'}

                    tracking.Tracking.RightPawTracking.insert1({
                        **{k: v for k, v in recs['paw_right'].items()
                           if k not in fmap},
                        **{fmap[k]: v for k, v in recs['paw_right'].items()
                           if k in fmap}}, allow_direct_insert=True)

                if 'lickport' in recs:
                    tracking.Tracking.LickPortTracking.insert1(
                        recs['lickport'], allow_direct_insert=True)

                # special handling for whisker(s)
                whisker_keys = [k for k in recs if 'whisker' in k]
                tracking.Tracking.WhiskerTracking.insert([
                    {**recs[k], 'whisker_name': k} for k in whisker_keys],
                    allow_direct_insert=True)

                tracking_files.append({
                    **key, 'trial': tmap[t], 'tracking_device': tdev,
                    'tracking_file': tracking_trial_filepath.relative_to(tracking_root_dir).as_posix()})

            log.info('... completed {}/{} items.'.format(i, n_tmap))

        log.info('\n---------------------')
        if tracking_files:
            if not tracking_exists:
                self.insert1(key)
            self.TrackingFile.insert(tracking_files)

            log.info('Tracking ingestion completed: {k}'.format(k=key))

    @staticmethod
    def load_campath(campath):
        ''' load camera position file-to-trial map '''
        log.debug("load_campath(): {}".format(campath))
        with open(campath, 'r') as f:
            return {int(k): int(v) for i in f
                    for k, v in (i.strip().split('\t'),)}

    def load_tracking(self, trkpath):
        log.debug('load_tracking() {}'.format(trkpath))
        '''
        load actual tracking data.

        example format:

        scorer,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000
        bodyparts,nose,nose,nose,tongue,tongue,tongue,jaw,jaw,jaw
        coords,x,y,likelihood,x,y,likelihood,x,y,likelihood
        0,418.48327827453613,257.231650352478,1.0,426.47182297706604,263.82502603530884,1.796432684386673e-06,226.12365770339966,395.8081398010254,1.0

        results are of the form:

          {'feature': {'attr': [val, ...]}}

        where feature is e.g. 'nose', 'attr' is e.g. 'x'.

        the special 'feature'/'attr' pair "samples"/"ts" is used to store
        the first column/sample timestamp for each row in the input file.
        '''
        res = defaultdict(lambda: defaultdict(list))

        with open(trkpath, 'r') as f:
            f.readline()  # discard 1st line
            parts, fields = f.readline(), f.readline()
            parts = parts.rstrip().split(',')
            fields = fields.rstrip().split(',')

            for l in f:
                if l.strip():
                    lv = l.rstrip().split(',')
                    for i, v in enumerate(lv):
                        v = float(v)
                        if i == 0:
                            res['samples']['ts'].append(v)
                        else:
                            res[parts[i]][fields[i]].append(v)

        return res


# ======== Helpers for directory navigation ========

def _get_same_day_session_order(session):
    """
    Given the session information, return the ordering of that session for that animal in the day
    - e.g. 1st or 2nd or 3rd ... session
    :param session: session dictionary (from experiment.Session.fetch1())
    :return: (int) 1-based session ordering
    """
    day_sessions = (experiment.Session & {'subject_id': session['subject_id'],
                                          'session_date': session['session_datetime'].date()})
    ordered_sess_numbers, _ = day_sessions.fetch(
        'session', 'session_time', order_by='session_time')  # TODO: remove 'session_time' when datajoint fix completed
    _, session_nth, _ = np.intersect1d(ordered_sess_numbers, session['session'],
                                       assume_unique=True, return_indices=True)
    return session_nth[0] + 1  # 1-based indexing


def _get_sess_tracking_dir(tracking_path, session):
    """
    Given the session information and a tracking root data directory,
    search and return:
     i) the directory containing the session's tracking data
     ii) the format of the session directory name

    Two directory structure conventions supported:
    1. tracking_path / h2o / h2o_mmddyy / *.csv(s)
    2. tracking_path / h2o / YYYYmmdd / tracking / *.csv(s)
    """
    tracking_path = pathlib.Path(tracking_path)
    h2o = session['water_restriction_number']
    sess_date = session['session_datetime'].date()

    if (tracking_path / h2o).exists():
        log.info('Checking for tracking data at: {}'.format(tracking_path / h2o))
    else:
        raise FileNotFoundError('{} not found'.format(tracking_path / h2o))

    session_nth = _get_same_day_session_order(session)
    session_nth_str = '_{}'.format(session_nth) if session_nth > 1 else ''

    sess_dirname = '{}_{}'.format(h2o, sess_date.strftime('%m%d%y')) + session_nth_str
    dir = tracking_path / h2o / sess_dirname

    legacy_sess_dirname = sess_date.strftime('%Y%m%d') + session_nth_str
    legacy_dir = tracking_path / h2o / legacy_sess_dirname / 'tracking'

    if dir.exists():
        log.info('Found {}'.format(dir.relative_to(tracking_path)))
        return dir, sess_date.strftime('%m%d%y') + session_nth_str
    elif legacy_dir.exists():
        log.info('Found {}'.format(legacy_dir.relative_to(tracking_path)))
        return legacy_dir, sess_date.strftime('%Y%m%d') + session_nth_str
    else:
        raise FileNotFoundError('Neither ({}) nor ({}) found'.format(
            dir.relative_to(tracking_path), legacy_dir.relative_to(tracking_path)))


def _get_MTL_sess_tracking_dir(tracking_path, session, camera_position):
    """
    Specialized directory searching method for "multi-target-licking" tracking data (recorded from RRig-MTL at Baylor)
    Given the session information, a tracking root data directory, and the camera position
    search and return:
     i) the directory containing the session's tracking data
     ii) the format of the session directory name

    The directory structure conventions supported:
        tracking_path / camera_position / h2o / YYYY_mm_dd / *.csv(s)
    """
    tracking_path = pathlib.Path(tracking_path)
    h2o = session['water_restriction_number']
    sess_date = session['session_datetime'].date()

    session_nth = _get_same_day_session_order(session)
    session_nth_str = '_{}'.format(session_nth) if session_nth > 1 else ''

    for tpos_candidate in TrackingIngest.camera_position_mapper[camera_position]:
        sess_dirname = sess_date.strftime('%Y_%m_%d') + session_nth_str
        dir = tracking_path / tpos_candidate / h2o / sess_dirname

        if dir.exists() and list(dir.glob('*{}*.csv'.format(tpos_candidate))):
            return dir, sess_date.strftime('%Y_%m_%d') + session_nth_str

    raise FileNotFoundError(
        'Multi-target-licking tracking data dir ({}) not found'.format(dir.relative_to(tracking_path)))
