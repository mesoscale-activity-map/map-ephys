
import logging
import pathlib
import csv
import re

import numpy as np
import scipy.io as scio
import datajoint as dj

from pipeline import lab
from pipeline import ephys
from pipeline import experiment
from pipeline import ccf
from pipeline import histology
from pipeline.ingest import behavior as behavior_ingest

from .. import get_schema_name

schema = dj.schema(get_schema_name('ingest_histology'))

log = logging.getLogger(__name__)


def get_histology_paths():
    """
    retrieve histology paths from dj.config
    config should be in dj.config of the format:

      dj.config = {
        ...,
        'custom': {
          'histology_data_paths': ['/path/string', '/path2/string']
        }
        ...
      }
    """
    return dj.config.get('custom', {}).get('histology_data_paths', None)


@schema
class HistologyIngest(dj.Imported):
    definition = """
    -> ephys.ProbeInsertion
    """

    class HistologyFile(dj.Part):
        definition = """
        -> master
        histology_file:                 varchar(255)    # rig file subpath
        """

    class LandmarkFile(dj.Part):
        definition = """
        -> master
        landmark_file:                  varchar(255)    # rig file subpath
        """

    # ephys.ProbeInsertion without ElectrodeCCFPosition and LabeledProbeTrack
    key_source = (ephys.ProbeInsertion
                  - histology.ElectrodeCCFPosition
                  - histology.LabeledProbeTrack)

    def make(self, key):
        '''
        HistologyIngest .make() function
        Expecting histology filename format to be:
        + landmarks_{water_res_number}_{session_date}_{session_time}_{probe_no}_{shank_no}.csv
        + landmarks_{water_res_number}_{session_date}_{session_time}_{probe_no}_{shank_no}_siteInfo.mat
        '''
        # TODO: check the length of the `site.ont.name` variable,
        #   and only ingest the sites with an ontology associated to it.

        log.info('\n======================================================')
        log.info('HistologyIngest().make(): key: {}'.format(key))

        self.session = (experiment.Session * lab.WaterRestriction.proj('water_restriction_number') & key).fetch1()

        rigpaths = get_histology_paths()

        self.water = self.session['water_restriction_number']
        self.probe = key['insertion_number']
        self.session_date_str = self.session['session_date'].strftime('%Y%m%d')

        # electrode configuration
        self.egroup = (ephys.ProbeInsertion * lab.ElectrodeConfig.ElectrodeGroup & key).fetch1('KEY')
        self.shanks = np.unique((lab.ElectrodeConfig.Electrode * lab.ProbeType.Electrode & self.egroup).fetch('shank'))

        # behavior_file
        if not (behavior_ingest.BehaviorIngest.BehaviorFile & key):
            log.warning('Missing BehaviorFile for session: {}. Skipping...'.format(self.session))
            return

        self.q_behavior_file = (behavior_ingest.BehaviorIngest.BehaviorFile & key)

        self.directory = None
        for rigpath in rigpaths:
            directory = pathlib.Path(rigpath, self.water, 'histology')
            if directory.exists():
                self.directory = directory
                break

        if self.directory is None:
            log.warning('Histology folder for animal: {} not found. Skipping...'.format(self.water))
            return

        # ingest histology
        prb_ingested, trk_ingested = False, False

        try:
            prb_ingested = self._load_histology_ccf()
        except FileNotFoundError as e:
            log.warning('Error: {}'.format(str(e)))
        except HistologyFileError as e:
            log.warning('Error: {}'.format(str(e)))

        if not prb_ingested:
            dj.conn().cancel_transaction()
            return

        try:
            trk_ingested = self._load_histology_track()
        except FileNotFoundError as e:
            log.warning('Error: {}'.format(str(e)))
            log.warning('Error: No histology with probe track. Skipping...')
        except HistologyFileError as e:
            log.warning('Error: {}'.format(str(e)))

        if not trk_ingested:
            dj.conn().cancel_transaction()
            return

        self.insert1(key)

    def _load_histology_ccf(self):

        sz = 20   # 20um voxel size

        log.info('... probe {} position ingest.'.format(self.probe))
        probefiles, shanks = self._search_histology_files('landmark_file')

        log.info('... found probe {} histology file(s) {}'.format(
            self.probe, probefiles))

        for probepath, shank_no in zip(probefiles, shanks):
            hist = scio.loadmat(probepath, struct_as_record=False, squeeze_me=True)['site']

            # probe CCF 3D positions
            pos_xyz = np.vstack([hist.pos.x, hist.pos.y, hist.pos.z,
                                 hist.warp.x, hist.warp.y, hist.warp.z]).T * sz

            # probe CCF regions
            ont_ids = np.where(np.isnan(hist.ont.id), 0, hist.ont.id)

            probe_electrodes = (ephys.ProbeInsertion.proj() * lab.ProbeType.Electrode & self.egroup
                                & {'shank': shank_no}).fetch(as_dict=True, order_by='electrode asc')

            if len(ont_ids) < len(probe_electrodes):
                raise HistologyFileError('Expecting at minimum {} electrodes - found {}'.format(
                    len(probe_electrodes), len(ont_ids)))

            ont_ids = ont_ids[:len(probe_electrodes)]
            pos_xyz = pos_xyz[:len(probe_electrodes), :]

            inserted_electrodes = (ephys.ProbeInsertion.proj() * lab.ElectrodeConfig.Electrode.proj()
                                   * lab.ProbeType.Electrode.proj('shank')
                                   & self.egroup & {'shank': shank_no}).fetch('electrode', order_by='electrode asc')

            recs = ({**electrode, **self.egroup, 'ccf_label_id': ccf.CCFLabel.CCF_R3_20UM_ID,
                     'ccf_x': ccf_x, 'ccf_y': ccf_y, 'ccf_z': ccf_z, 'mri_x': mri_x, 'mri_y': mri_y, 'mri_z': mri_z}
                    for electrode, (ccf_x, ccf_y, ccf_z, mri_x, mri_y, mri_z), ont_id in
                    zip(probe_electrodes, pos_xyz, ont_ids)
                    if ont_id > 0 and electrode['electrode'] in inserted_electrodes)

            # ideally ElectrodePosition.insert(...) but some are outside of CCF...
            log.info('inserting channel ccf position')
            histology.ElectrodeCCFPosition.insert1(self.egroup, ignore_extra_fields=True,
                                                   skip_duplicates=True)

            for r in recs:
                log.debug('... adding probe/position: {}'.format(r))
                try:
                    histology.ElectrodeCCFPosition.ElectrodePosition.insert1(
                        r, ignore_extra_fields=True, allow_direct_insert=True)
                except Exception as e:  # XXX: no way to be more precise in dj
                    # log.warning('... ERROR!: {}'.format(repr(e)))
                    histology.ElectrodeCCFPosition.ElectrodePositionError.insert1(
                        r, ignore_extra_fields=True, allow_direct_insert=True)

            log.info('... ok.')

        return True

    def _load_histology_track(self):

        log.info('... probe {} probe-track ingest.'.format(self.probe))

        trackpaths, shanks = self._search_histology_files('histology_file')

        if trackpaths:
            log.info('... found probe {} histology file(s): {}'.format(
                self.probe, trackpaths))
        else:
            raise FileNotFoundError

        conv = (('landmark_name', str), ('warp', lambda x: x.lower() == 'true'),
                ('subj_x', float), ('subj_y', float), ('subj_z', float),
                ('ccf_x', float), ('ccf_y', float), ('ccf_z', float))

        for trackpath, shank_no in zip(trackpaths, shanks):
            recs = []
            with open(trackpath.as_posix(), newline='') as f:
                rdr = csv.reader(f)
                for row in rdr:
                    assert len(row) == 8
                    rec = {c[0]: c[1](d) for c, d in zip(conv, row)}
                    recs.append(rec)

            # Subject -> CCF Transformation

            top = {'subject_id': self.session['subject_id']}

            if not (histology.SubjectToCCFTransformation & top).fetch(limit=1):

                log.info('... adding new raw -> ccf coordinates')

                histology.SubjectToCCFTransformation.insert1(
                    top, allow_direct_insert=True)

                histology.SubjectToCCFTransformation.Landmark.insert(
                    ({**top, **rec} for rec in
                     (r for r in recs if r['warp'] is True)),
                    allow_direct_insert=True, ignore_extra_fields=True)

            else:
                log.debug('... skipping raw -> ccf coordinates')

            # LabeledProbeTrack

            top = {**self.egroup, 'labeling_date': None, 'dye_color': None}

            histology.LabeledProbeTrack.insert1(
                top, ignore_extra_fields=True, allow_direct_insert=True, skip_duplicates=True)

            histology.LabeledProbeTrack.Point.insert(
                ({**top, 'order': rec_order, 'shank': shank_no, **rec} for rec_order, rec in
                 enumerate((r for r in recs if r['warp'] is False))),
                ignore_extra_fields=True, allow_direct_insert=True)

        return True

    def _search_histology_files(self, file_type):
        """
        :param file_type, either:
        + histology_file - format: landmarks_{water_res_number}_{session_date}_{session_time}_{probe_no}_{shank_no}.csv
        + landmark_file - format landmarks_{water_res_number}_{session_date}_{session_time}_{probe_no}_{shank_no}_siteInfo.mat
        Returns a list of files (1 file for SS, 4 for MS)
        """

        file_format_map = {'landmark_file': '_siteInfo.mat',
                           'histology_file': '.csv'}

        # ---- probefile - landmarks_{water_res_number}_{session_date}_{session_time}_{probe_no}_{shank_no}.csv
        histology_files = [f for f in list(self.directory.glob('landmarks*')) if re.match(
            'landmarks_{}_{}(_\d\d\d\d\d\d)?_{}(_\d)?{}'.format(self.water, self.session_date_str,
                                                                self.probe, file_format_map[file_type]), f.name)]

        if len(histology_files) < 1:
            raise FileNotFoundError('Probe {} histology file {} not found!'.format(self.probe,
                                                                                   'landmarks_{}_{}*{}'.format(
                                                                                       self.water,
                                                                                       self.session_date_str,
                                                                                       file_format_map[file_type])))
        elif len(histology_files) == 1:
            corresponding_shanks = [1]
            if len(self.shanks) != 1:
                raise HistologyFileError('Only 1 file found ({}) for a {}-shank probe'.format(histology_files[0].name, len(self.shanks)))

            match = re.search('landmarks_{}_{}_?(.*)_{}_?(.*){}'.format(
                self.water, self.session_date_str, self.probe, file_format_map[file_type]), histology_files[0].name)
            session_time_str, _ = match.groups()
            if session_time_str == '':
                same_day_sess_count = len(experiment.Session & {'subject_id': self.session['subject_id'], 'session_date': self.session['session_date']})
                if same_day_sess_count != 1:
                    raise HistologyFileError('{} same-day sessions found - but only 1 histology file found ({}) with no "session_time" specified'.format(
                        same_day_sess_count, histology_files[0].name))
            else:
                behavior_time_str = re.search('_(\d{6}).mat', self.q_behavior_file.fetch1('behavior_file')).groups()[0]
                if session_time_str != behavior_time_str:
                    raise HistologyFileError('Only 1 histology file found ({}) with "session_time" ({}) different from "behavior_time" ({})'.format(
                        histology_files[0].name, session_time_str, behavior_time_str))

        else:
            behavior_time_str = re.search('_(\d{6}).mat', self.q_behavior_file.fetch1('behavior_file')).groups()[0]
            file_format = 'landmarks_{}_{}_{}_{}*{}'.format(self.water, self.session_date_str,
                                                            behavior_time_str,
                                                            self.probe, file_format_map[file_type])
            histology_files = [f for f in list(self.directory.glob('landmarks*')) if re.match(
                'landmarks_{}_{}_{}_{}(_\d)?{}'.format(self.water, self.session_date_str, behavior_time_str,
                                                       self.probe, file_format_map[file_type]), f.name)]

            if len(histology_files) < 1:
                raise FileNotFoundError('Probe {} histology file {} not found!'.format(self.probe, file_format))

            if len(histology_files) != len(self.shanks):  # ensure 1 file per shank
                raise HistologyFileError('{} files found for a {}-shank probe'.format(len(histology_files), len(self.shanks)))

            corresponding_shanks = [int(re.search('landmarks_.*_(\d)_(\d){}'.format(file_format_map[file_type]),
                                                  f.as_posix()).groups()[1]) for f in histology_files]

        return histology_files, corresponding_shanks


class HistologyFileError(Exception):
    """Raise when error encountered when ingesting probe insertion"""
    def __init__(self, msg=None):
        super().__init__('Histology File Error: \n{}'.format(msg))
    pass
