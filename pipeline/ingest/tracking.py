
import os
import logging
import pathlib
from glob import glob

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


@schema
class TrackingDataPath(dj.Lookup):
    # ephys data storage location(s)
    definition = """
    -> lab.Rig
    tracking_data_path:         varchar(255)            # rig data path
    """

    @property
    def contents(self):
        if 'tracking_data_paths' in dj.config['custom']:  # for local testing
            return dj.config['custom']['tracking_data_paths']

        return [('RRig', r'H:\\data\MAP',)]


@schema
class TrackingIngest(dj.Imported):
    definition = """
    -> behavior_ingest.BehaviorIngest
    """

    class TrackingFile(dj.Part):
        definition = '''
        -> TrackingIngest
        -> experiment.SessionTrial
        ---
        tracking_file:          varchar(255)            # tracking file subpath
        '''

    def make(self, key):
        '''
        TrackingIngest .make() function
        '''
        log.info('TrackingIngest().make(): key: {k}'.format(k=key))

        h2o = (lab.WaterRestriction() & key).fetch1('water_restriction_number')
        session = (experiment.Session() & key).fetch1()
        trials = (experiment.SessionTrial() & session).fetch('trial')

        log.info('got session: {} ({} trials)'.format(session, len(trials)))

        sdate = session['session_date']
        sdate_iso = sdate.isoformat()  # YYYY-MM-DD
        sdate_sml = "{}{:02d}{:02d}".format(sdate.year, sdate.month, sdate.day)

        paths = TrackingDataPath.fetch(as_dict=True)
        devices = tracking.TrackingDevice().fetch(as_dict=True)

        # paths like: <root>/<h2o>/YYYY-MM-DD/tracking
        for p, d in ((p, d) for d in devices for p in paths):

            tdev = d['tracking_device']
            tpos = d['tracking_position']
            tdat = p['tracking_data_path']

            log.info('checking {} for tracking data'.format(tdat))

            tpath = pathlib.Path(tdat, h2o, sdate_iso, 'tracking')

            if not tpath.exists():
                log.warning('tracking path {} n/a - skipping'.format(tpath))
                continue

            camtrial = '{}_{}_{}.txt'.format(h2o, sdate_sml, tpos)
            campath = tpath / camtrial

            log.info('trying camera position trial map: {}'.format(campath))

            if not campath.exists():
                log.info('skipping {} - does not exist'.format(campath))
                continue

            tmap = self.load_campath(campath)

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
                tfile = '{}_{}_{}-*.csv'.format(h2o, tpos, t)
                tfull = list(tpath.glob(tfile))

                if not tfull or len(tfull) > 1:
                    log.info('tracking file {} mismatch'.format(tfull))
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

                tracking.Tracking.NoseTracking.insert1(
                    recs['nose'], allow_direct_insert=True)

                tracking.Tracking.TongueTracking.insert1(
                    recs['tongue'], allow_direct_insert=True)

                tracking.Tracking.JawTracking.insert1(
                    recs['jaw'], allow_direct_insert=True)

            log.info('... completed {}/{} items.'.format(i, n_tmap))
            log.info('... saving load record')

            self.insert1(key)

            log.info('... done.')

    def load_campath(self, campath):
        ''' load camera position-to-trial map '''
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
