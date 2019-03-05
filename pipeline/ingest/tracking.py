# tracking_data_path removed from main; should go here += file traversal

'''
dl56_side_3-0000.csv : h2o_<camera>_3_XXXX (ignore XXXX)
file paths: root/h2o/date/'string of tracking'/file
(same as ephys, 'tracking' instead of '1','2',etc)
  - values are detected positions and video frame number
  - paths root/h2o/date/tracking/*.csv
  - file, 'dl59_20181207_side.txt' tracking record no -> trial no
'''

import os
import logging
from glob import glob

import datajoint as dj

from pipeline import lab
from pipeline import tracking
from pipeline import experiment
from pipeline.ingest import behavior as ingest_behavior

from code import interact  # HACK debug
from collections import ChainMap  # HACK debug
from collections import defaultdict

log = logging.getLogger(__name__)


schema = dj.schema(dj.config.get(
    'ingest.tracking.database',
    '{}_ingestTracking'.format(dj.config['database.user'])))


@schema
class TrackingDataPath(dj.Lookup):
    # ephys data storage location(s)
    definition = """
    -> lab.Rig
    tracking_data_path:         varchar(255)            # rig data path
    """

    @property
    def contents(self):
        if 'tracking_data_paths' in dj.config:  # for local testing
            return dj.config['tracking_data_paths']

        return [('RRig', r'H:\\data\MAP',)]


@schema
class TrackingIngest(dj.Imported):
    definition = """
    -> ingest_behavior.BehaviorIngest
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

        qs:
        - really using multiple rig paths?
        - camera number vs name/label: splitting would be better if OK
        - tracking files: always -0000.csv?
        - 1321914 points in one session.. sooo probably should blobify
          how to structure?
        '''
        log.info('TrackingIngest().make(): key: {k}'.format(k=key))

        h2o = (lab.WaterRestriction() & key).fetch1('water_restriction_number')
        session = (experiment.Session() & key).fetch1()
        trials = (experiment.SessionTrial() & session).fetch(as_dict=True)

        sdate = session['session_date']
        sdate_iso = sdate.isoformat()  # YYYY-MM-DD
        sdate_sml = "{}{:02d}{:02d}".format(sdate.year, sdate.month, sdate.day)

        # todo: split name vs position? confirm
        devices = tracking.TrackingDevice().fetch('tracking_device')

        # paths like: <root>/<h2o>/YYYY-MM-DD/tracking
        for p in TrackingDataPath.fetch('tracking_data_path'):

            print('key', key)  # subjectid sesssionno
            print('trying {}'.format(p))
            print('got session: {}'.format(session))
            print('got trials: {}'.format(trials))

            tpath = os.path.join(p, h2o, sdate_iso, 'tracking')

            print('trying tracking path: {}'.format(tpath))

            if not os.path.exists(tpath):
                log.info('skipping {} - does not exist'.format(tpath))

            # todo: for c in devices:
            # interact('muhrepl', local=dict(ChainMap(locals(), globals())))
            camtrial = '{}_{}_side.txt'.format(h2o, sdate_sml)
            campath = os.path.join(tpath, camtrial)
            if os.path.exists(campath):
                tmap = self.load_campath(campath)
                for t in tmap:
                    log.info('t in tmap: {}'.format(t))
                    # dl59_side_27-0000.csv / h2o_cname_t-0000.csv
                    tfile = '{}_side_{}-0000.csv'.format(h2o, t)
                    tfull = os.path.join(tpath, tfile)
                    if os.path.exists(tfull):
                        trk = self.load_tracking(tfull)
                        # rekey based on tmap
                        # insert.

    def load_campath(self, campath):
        ''' load camera position-to-trial map '''
        log.debug("load_campath(): {}".format(campath))
        with open(campath, 'r') as f:
            return {int(k): int(v) for i in f
                    for k, v in (i.strip().split('\t'),)}

    def load_tracking(self, trkpath):
        log.info('load_tracking() {}'.format(trkpath))
        '''
        file format:
        scorer,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000,DeepCut_resnet50_licking-sideAug10shuffle1_1030000
        bodyparts,nose,nose,nose,tongue,tongue,tongue,jaw,jaw,jaw
        coords,x,y,likelihood,x,y,likelihood,x,y,likelihood
        0,418.48327827453613,257.231650352478,1.0,426.47182297706604,263.82502603530884,1.796432684386673e-06,226.12365770339966,395.8081398010254,1.0
        '''
        # res['nose']['x'] = 1; res['nose']['ts'] = 1.00, etc
        res = defaultdict(lambda: defaultdict(list))

        with open(trkpath, 'r') as f:
            _, parts, fields = f.readline(), f.readline(), f.readline()
            for l in f:
                lv = l.split(',')
                for i, v in enumerate(lv):
                    if i == 0:
                        ts = v
                    else:
                        res[parts[i]]['ts'] = ts  # todo: * Hz?
                        res[parts[i]][fields[i]] = v

        return res
