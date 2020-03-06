
import os
import logging
import pathlib
from glob import glob
from datetime import datetime

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
    '''
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

    '''
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

    def make(self, key):
        '''
        TrackingIngest .make() function
        '''
        log.info('\n======================================================')
        log.info('TrackingIngest().make(): key: {k}'.format(k=key))

        h2o = (lab.WaterRestriction & key).fetch1('water_restriction_number')
        session = (experiment.Session & key).fetch1()
        trials = (experiment.SessionTrial & session).fetch('trial')

        log.info('got session: {} ({} trials)'.format(session, len(trials)))
        log.info('\n-----------')

        sess_time = (datetime.min + session['session_time']).time()
        sess_datetime = datetime.combine(session['session_date'], sess_time)

        paths = get_tracking_paths()
        devices = tracking.TrackingDevice.fetch(as_dict=True)

        # paths like: <root>/<h2o>/YYYY-MM-DD/tracking
        tracking_files = []
        for p, d in ((p, d) for d in devices for p in paths):

            tdev = d['tracking_device']
            cam_pos = d['tracking_position']
            tdat = p[-1]

            try:
                tpath, sdate_sml = _get_sess_tracking_dir(tdat, h2o, sess_datetime)
            except FileNotFoundError as e:
                log.warning('{} - skipping'.format(str(e)))
                continue

            campath = None
            for tpos_name in self.camera_position_mapper[cam_pos]:
                camtrial_fn = '{}_{}_{}.txt'.format(h2o, sdate_sml, tpos_name)
                log.info('trying camera position trial map: {}'.format(tpath / camtrial_fn))
                if (tpath / camtrial_fn).exists():
                    campath = tpath / camtrial_fn
                    tpos = tpos_name
                    break

            if campath is None:
                log.info('Video-Trial mapper file (.txt) not found - Skipping...')
                continue

            tmap = self.load_campath(campath)  # file:trial

            n_tmap = len(tmap)
            log.info('loading tracking data for {} trials'.format(n_tmap))

            i = 0
            for t in tmap:  # load tracking for trial
                if tmap[t] not in trials:
                    log.warning('nonexistant trial {}.. skipping'.format(t))
                    continue

                i += 1
                if i % 50 == 0:
                    log.info('item {}/{}, trial #{} ({:.2f}%)'
                             .format(i, n_tmap, t, (i/n_tmap)*100))
                else:
                    log.debug('item {}/{}, trial #{} ({:.2f}%)'
                              .format(i, n_tmap, t, (i/n_tmap)*100))

                # ex: dl59_side_1-0000.csv / h2o_position_tn-0000.csv
                tfile = '{}*_{}_{}-*.csv'.format(h2o, tpos, t)
                tfull = list(tpath.glob(tfile))

                if not tfull or len(tfull) > 1:
                    log.info('file mismatch: file: {} trial: {} ({})'.format(
                        t, tmap[t], tfull))
                    continue

                tfull = tfull[-1]
                trk = self.load_tracking(tfull)

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

                tracking_files.append({**key, 'trial': tmap[t], 'tracking_device': tdev,
                                       'tracking_file': str(tfull.relative_to(tdat))})

            log.info('... completed {}/{} items.'.format(i, n_tmap))

        if tracking_files:
            self.insert1(key)
            self.TrackingFile.insert(tracking_files)

            log.info('... done.')

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
                lv = l.rstrip().split(',')
                for i, v in enumerate(lv):
                    v = float(v)
                    if i == 0:
                        res['samples']['ts'].append(v)
                    else:
                        res[parts[i]][fields[i]].append(v)

        return res


# ======== Helpers for directory navigation ========

def _get_sess_tracking_dir(tracking_path, h2o, sess_datetime):
    tracking_path = pathlib.Path(tracking_path)

    if (tracking_path / h2o).exists():
        log.info('Checking for tracking data at: {}'.format(tracking_path / h2o))
    else:
        raise FileNotFoundError('{} not found'.format(tracking_path / h2o))

    dir_format = tracking_path / h2o / '{}_{}'.format(h2o, sess_datetime.date().strftime('%m%d%y'))
    legacy_dir_format = tracking_path / h2o / sess_datetime.date().strftime('%Y%m%d') / 'tracking'

    if dir_format.exists():
        return dir_format, sess_datetime.date().strftime('%m%d%y')
    elif legacy_dir_format.exists():
        return legacy_dir_format, sess_datetime.date().strftime('%Y%m%d')
    else:
        raise FileNotFoundError('Neither ({}) nor ({}) found'.format(dir_format.relative_to(tracking_path),
                                                                     legacy_dir_format.relative_to(tracking_path)))
