import logging
import pathlib
import csv
import re
import json
import math

import numpy as np
import scipy.io as scio
import datajoint as dj
from datetime import datetime

from pipeline import lab, ephys, experiment, ccf, histology, report
from pipeline import get_schema_name, dict_to_hash

from pipeline.ingest import behavior as behavior_ingest

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

    Histology paths should be layed out as follows:

      {histology_data_path}/{h2o}/histology/{files}

    Expected files are documented in HistologyIngest._search_histology_files.
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
        self.rig = self.session['rig']
        self.session_date_str = self.session['session_date'].strftime('%Y%m%d')

        # electrode configuration
        self.egroup = (ephys.ProbeInsertion * lab.ElectrodeConfig.ElectrodeGroup & key).fetch1('KEY')
        self.shanks = np.unique((lab.ElectrodeConfig.Electrode * lab.ProbeType.Electrode & self.egroup).fetch('shank'))

        # behavior_file
        if not (behavior_ingest.BehaviorIngest.BehaviorFile & key):
            log.warning('Missing BehaviorFile for session: {}. Skipping...'.format(self.session))
            return

        self.q_behavior_file = (behavior_ingest.BehaviorIngest.BehaviorFile & key)

        for rigpath in rigpaths:
            if self.rig == 'RRig-MTL':
                directory = pathlib.Path(rigpath, self.water) / '{}_{}'.format(
                    self.session_date_str, self.probe)
                try:
                    self.directory = next(directory.rglob('channel_locations*.json')).parent
                    break
                except StopIteration:
                    pass
            else:
                directory = pathlib.Path(rigpath, self.water, 'histology')
                if directory.exists():
                    self.directory = directory
                    break
        else:
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

        ccf_res = ccf.CCFLabel.CCF_R3_20UM_RESOLUTION   # 20um voxel size

        log.info('... probe {} position ingest.'.format(self.probe))
        found = self._search_histology_files('landmark_file')

        if found['format'] == 1:

            probefiles = found['histology_files']
            shanks = found['corresponding_shanks']

            log.info('... found probe {} histology file(s) {}'.format(
                self.probe, probefiles))

            for probepath, shank_no in zip(probefiles, shanks):
                hist = scio.loadmat(probepath, struct_as_record=False, squeeze_me=True)['site']
                px_res = int(hist.pos.mmPerPixel * 1000)

                # probe CCF 3D positions
                ccf_xyz = np.vstack([hist.pos.x, hist.pos.y, hist.pos.z]).T * px_res
                # and quantizing to CCF voxel size;
                ccf_xyz = ccf_res * np.around(ccf_xyz / ccf_res)

                # bundle with MRI position
                mri_xyz = np.vstack([hist.warp.x, hist.warp.y, hist.warp.z]).T * px_res
                pos_xyz = np.hstack([ccf_xyz, mri_xyz])

                # probe CCF regions
                ont_ids = np.where(np.isnan(hist.ont.id), 0, hist.ont.id)

                probe_electrodes = (ephys.ProbeInsertion.proj() * lab.ProbeType.Electrode & self.egroup
                                    & {'shank': shank_no}).fetch(as_dict=True, order_by='electrode asc')

                if len(ont_ids) < len(probe_electrodes):
                    raise HistologyFileError('Expecting at minimum {} electrodes - found {}'.format(
                        len(probe_electrodes), len(ont_ids)))

                ont_ids = ont_ids[:len(probe_electrodes)]
                pos_xyz = pos_xyz[:len(probe_electrodes), :]

                recording_electrodes = (ephys.ProbeInsertion.proj() * lab.ElectrodeConfig.Electrode.proj()
                                        * lab.ProbeType.Electrode.proj('shank')
                                        & self.egroup & {'shank': shank_no}).fetch(
                    'electrode', order_by='electrode asc')

                recs = ({**electrode, **self.egroup, 'ccf_label_id': ccf.CCFLabel.CCF_R3_20UM_ID,
                         'ccf_x': int(ccf_x), 'ccf_y': int(ccf_y), 'ccf_z': int(ccf_z),
                         'mri_x': mri_x, 'mri_y': mri_y, 'mri_z': mri_z}
                        for electrode, (ccf_x, ccf_y, ccf_z, mri_x, mri_y, mri_z), ont_id in
                        zip(probe_electrodes, pos_xyz, ont_ids)
                        if ont_id > 0 and electrode['electrode'] in recording_electrodes)

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

        if found['format'] == 2:
            # load data,
            for channel_location_file, shank_no in zip(found['channel_location_files'],
                                                       found['corresponding_shanks']):

                log.debug('loading format 2 channels from {}'.format(channel_location_file))
                with open(channel_location_file, 'r') as fh:
                    cloc_raw = json.loads(fh.read())

                cloc_data = {'origin': cloc_raw['origin']}

                if len(cloc_data['origin'].keys()) > 1:
                    log.error('More than one origin region found ({}). skipping.'.format(
                        cloc_data['origin']))
                    return

                # ensuring channel data is sorted;
                cloc_keymap = {int(k.split('_')[1]): k for k
                               in cloc_raw.keys() if 'channel_' in k}

                cloc_data['channels'] = np.array(
                    [tuple(cloc_raw[cloc_keymap[k]].values()) for k in sorted(
                        cloc_keymap.keys())],
                    dtype=[
                        ('x', np.float), ('y', np.float), ('z', np.float),
                        ('axial', np.float), ('lateral', np.float),
                        ('brain_region_id', np.int), ('brain_region', np.object)])

                # get/scale xyz positions
                pos_xyz_raw = np.array([cloc_data['channels'][i]
                                        for i in ('x', 'y', 'z')]).T

                pos_origin = cloc_data['origin'][
                    list(cloc_data['origin'].keys())[0]]

                pos_xyz = np.copy(pos_xyz_raw)

                # by adjusting xyz axes & offsetting from origin position
                pos_xyz[:, 0] = pos_origin[0] + pos_xyz_raw[:, 0]
                pos_xyz[:, 1] = pos_origin[2] - pos_xyz_raw[:, 2]
                pos_xyz[:, 2] = pos_origin[1] - pos_xyz_raw[:, 1]

                # and quantizing to CCF voxel size;
                pos_xyz = ccf_res * np.around(pos_xyz / ccf_res)

                # get recording geometry,
                probe_electrodes = (lab.ProbeType.Electrode
                                    & (ephys.ProbeInsertion & self.egroup)
                                    & {'shank': shank_no}).fetch(order_by='electrode asc')

                rec_electrodes = np.array(
                    [cloc_data['channels']['lateral'],
                     cloc_data['channels']['axial']]).T

                # adjusting to boundaries, # FIXME: WHY, ISOK?
                # also: example session was -= 11; this seems more robust.
                rec_electrodes[:, 0] = (
                    16 * (np.around(rec_electrodes[:, 0] / 16) - 1))

                # to find corresponding electrodes,
                elec_coord = np.array(
                    [probe_electrodes['x_coord'], probe_electrodes['y_coord']]).T

                elec_coord_map = {tuple(i[1]): i[0]
                                  for i in enumerate(elec_coord)}

                rec_to_elec_idx = np.array([elec_coord_map[tuple(i)]
                                            for i in rec_electrodes])

                # and then insert the ElectrodeCCFPosition records.
                log.debug('... adding ElectrodeCCFPosition: {}'.format(
                    self.egroup))

                histology.ElectrodeCCFPosition.insert1(
                    self.egroup, ignore_extra_fields=True, skip_duplicates=True)

                for z in zip(probe_electrodes[rec_to_elec_idx]['electrode'],
                             pos_xyz[:, 0], pos_xyz[:, 1], pos_xyz[:, 2]):

                    z = [int(i) for i in z]  # integer CCF voxels, electrodes.
                    rec = {**self.egroup, 'electrode': z[0],
                           'ccf_label_id': ccf.CCFLabel.CCF_R3_20UM_ID,
                           'ccf_x': z[1], 'ccf_y': z[2], 'ccf_z': z[3]}
                            # via nullable: 'mri_x': 0, 'mri_y': 0, 'mri_z': 0

                    log.debug('...... adding ElectrodePosition: {}'.format(rec))

                    try:
                        histology.ElectrodeCCFPosition.ElectrodePosition.insert1(rec)
                    except Exception as e:  # XXX: no way to be more precise in dj
                        log.warning('...... ElectrodePositionError: {}'.format(
                            repr(e)))
                        histology.ElectrodeCCFPosition.ElectrodePositionError.insert1(rec)

            return True

    def _load_histology_track(self):

        log.info('... probe {} probe-track ingest.'.format(str(self.probe)))

        found = self._search_histology_files('histology_file')

        probe_track_key = {**self.egroup, 'labeling_date': None, 'dye_color': None}

        histology.LabeledProbeTrack.insert1(
            probe_track_key, ignore_extra_fields=True, allow_direct_insert=True, skip_duplicates=True)

        if found['format'] == 1:
            trackpaths = found['histology_files']
            shanks = found['corresponding_shanks']

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

                subj_key = {'subject_id': self.session['subject_id']}

                if not (histology.SubjectToCCFTransformation & subj_key).fetch(limit=1):

                    log.info('... adding new raw -> ccf coordinates')

                    histology.SubjectToCCFTransformation.insert1(
                        subj_key, allow_direct_insert=True)

                    histology.SubjectToCCFTransformation.Landmark.insert(
                        ({**subj_key, **rec} for rec in
                         (r for r in recs if r['warp'] is True)),
                        allow_direct_insert=True, ignore_extra_fields=True)

                else:
                    log.debug('... skipping raw -> ccf coordinates')

                # LabeledProbeTrack
                histology.LabeledProbeTrack.Point.insert(
                    ({**probe_track_key, 'order': rec_order, 'shank': shank_no, **rec} for rec_order, rec in
                     enumerate((r for r in recs if r['warp'] is False))),
                    ignore_extra_fields=True, allow_direct_insert=True)

        if found['format'] == 2:
            for channel_location_file, xyz_picks_file, shank_no in zip(
                    found['channel_location_files'],
                    found['xyz_picks_files'],
                    found['corresponding_shanks']):

                log.debug('loading format 2 channels from {}'.format(channel_location_file))
                with open(channel_location_file, 'r') as fh:
                    channel_location = json.loads(fh.read())
                with open(xyz_picks_file, 'r') as fh:
                    xyz_pick = json.loads(fh.read())

                if len(channel_location['origin']) > 1:
                    log.error('More than one origin region found ({}). skipping.'.format(
                        channel_location['origin']))
                    return

                pos_origin = next(iter(channel_location['origin'].values()))
                xyz_pick = np.array(xyz_pick['xyz_picks'])

                shank_xyz = np.copy(xyz_pick)
                # by adjusting xyz axes & offsetting from origin position
                shank_xyz[:, 0] = pos_origin[0] + xyz_pick[:, 0]
                shank_xyz[:, 1] = pos_origin[2] - xyz_pick[:, 2]
                shank_xyz[:, 2] = pos_origin[1] - xyz_pick[:, 1]

                # LabeledProbeTrack
                histology.LabeledProbeTrack.Point.insert(
                    ({**probe_track_key, 'order': idx, 'shank': shank_no,
                      'ccf_x': x, 'ccf_y': y, 'ccf_z': z} for idx, (x, y, z) in
                     enumerate(shank_xyz)),
                    ignore_extra_fields=True, allow_direct_insert=True)

        return True

    def _search_histology_files(self, file_type):
        """
        :param file_type: either 'landmark_file' or 'histology_file'.

        FIXME name (original) format:

          + histology_file - format: landmarks_{water_res_number}_{session_date}_{session_time}_{probe_no}_{shank_no}.csv
          + landmark_file - format landmarks_{water_res_number}_{session_date}_{session_time}_{probe_no}_{shank_no}_siteInfo.mat

          Returns a list of files (1 file for SS, 4 for MS)

        FIXME name (new) format will also include:

          + SC030_20191002_1_channel_locations.json
          + SC030_20191002_1_channels.localCoordinates.npy
          + channels.rawInd.npy

        if available.

        returns values of the form:

            {
                'format': 1,
                'histology_files': histology_files,
                'corresponding_shanks': corresponding_shanks
            }

        for FIXME name (original) format, and:

            {
                'format': 2,
                'histology_files': histology_files,
                'corresponding_shanks': corresponding_shanks
                'raw_ind': rawInd file
                'channel_locations': channel_locations.json file
                'channel_local_coordinates': localCoordinates.npy file
            }

        for FIXME name (new) format.
        """

        # Detecting format #2 (.json)
        if self.rig == 'RRig-MTL':
            filename_prefix = ''
        else:
            filename_prefix = '{}_{}_{}_'.format(
                self.water, self.session_date_str, self.probe)

        format_number = 2 if (self.directory / '{}channel_locations*.json'.format(filename_prefix)).exists() else 1

        if format_number == 1:
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
                histology_sesstime_files = [f for f in list(self.directory.glob('landmarks*')) if re.match(
                    'landmarks_{}_{}_{}_{}(_\d)?{}'.format(self.water, self.session_date_str, behavior_time_str,
                                                           self.probe, file_format_map[file_type]), f.name)]

                if len(histology_sesstime_files) < 1:  # case of no session-time in filename
                    if len(histology_files) != len(self.shanks):  # ensure 1 file per shank
                        raise FileNotFoundError('Probe {} histology file {} not found!'.format(self.probe, file_format))
                    else:
                        histology_sesstime_files = histology_files
                elif len(histology_sesstime_files) != len(self.shanks):  # ensure 1 file per shank
                    raise HistologyFileError('{} files found for a {}-shank probe'.format(len(histology_sesstime_files),
                                                                                          len(self.shanks)))

                corresponding_shanks = [int(re.search('landmarks_.*_(\d)_(\d){}'.format(file_format_map[file_type]),
                                                      f.as_posix()).groups()[1]) for f in histology_sesstime_files]

            res = {
                'format': 1,
                'histology_files': histology_files,
                'corresponding_shanks': corresponding_shanks
            }
        elif format_number == 2:
            log.debug('format 2 detected..')

            channel_locations_files = list(self.directory.glob(
                '{}channel_locations*.json'.format(filename_prefix)))
            xyz_picks_files = list(self.directory.glob('xyz_picks*.json'))

            if len(channel_locations_files) < 1:
                raise FileNotFoundError(
                    'Probe {} histology file {} not found!'.format(
                        self.probe,
                        '{}channel_locations*.json'.format(filename_prefix)))
            elif len(channel_locations_files) == 1:
                corresponding_shanks = [1]
                if len(self.shanks) != 1:
                    raise HistologyFileError(
                        'Only 1 file found ({}) for a {}-shank probe'.format(
                            channel_locations_files[0].name, len(self.shanks)))
                if 'shank' in channel_locations_files[0].stem and channel_locations_files[0].stem[-1] != 1:
                    raise HistologyFileError('The histology file found ({}) is unexpected for this 1-shank probe')
            else:
                if len(channel_locations_files) != len(self.shanks):  # ensure 1 file per shank
                    raise HistologyFileError(
                        '{} files found for a {}-shank probe'.format(len(channel_locations_files),
                                                                     len(self.shanks)))
                corresponding_shanks = [int(f.stem[-1]) for f in channel_locations_files]

            res = {
                'format': 2,
                'channel_locations_files': channel_locations_files,
                'xyz_picks_files': xyz_picks_files,
                'corresponding_shanks': corresponding_shanks
            }

        return res


# ================== HELPER FUNCTIONS ====================


def archive_electrode_histology(insertion_key, note='', delete=False):
    """
    For the specified "insertion_key" copy from histology.ElectrodeCCFPosition and histology.LabeledProbeTrack
     (and their respective part tables) to histology.ArchivedElectrodeHistology
    If "delete" == True - delete the records associated with the "insertion_key" from:
        + histology.ElectrodeCCFPosition
        + histology.LabeledProbeTrack
        + report.ProbeLevelDriftMap
        + report.ProbeLevelCoronalSlice
    """

    e_ccfs = {d['electrode']: d
              for d in (histology.ElectrodeCCFPosition.ElectrodePosition & insertion_key).fetch(as_dict=True)}
    e_error_ccfs = {d['electrode']: d
                    for d in (histology.ElectrodeCCFPosition.ElectrodePositionError & insertion_key).fetch(as_dict=True)}
    e_ccfs_hash = dict_to_hash({**e_ccfs, **e_error_ccfs})

    if histology.ArchivedElectrodeHistology & {'archival_hash': e_ccfs_hash}:
        if delete:
            if dj.utils.user_choice('The specified ElectrodeCCF has already been archived!\nProceed with delete?') != 'yes':
                return
        else:
            print('An identical set of the specified ElectrodeCCF has already been archived')
            return

    archival_time = datetime.now()
    with histology.ArchivedElectrodeHistology.connection.transaction:
        histology.ArchivedElectrodeHistology.insert1({**insertion_key, 'archival_time': archival_time,
                                                      'archival_note': note, 'archival_hash': e_ccfs_hash})
        histology.ArchivedElectrodeHistology.ElectrodePosition.insert(
            (histology.ElectrodeCCFPosition.ElectrodePosition & insertion_key).proj(
                ..., archival_time='"{}"'.format(archival_time)))
        histology.ArchivedElectrodeHistology.ElectrodePositionError.insert(
            (histology.ElectrodeCCFPosition.ElectrodePositionError & insertion_key).proj(
                ..., archival_time='"{}"'.format(archival_time)))
        histology.ArchivedElectrodeHistology.LabeledProbeTrack.insert(
            (histology.LabeledProbeTrack & insertion_key).proj(
                ..., archival_time='"{}"'.format(archival_time)))
        histology.ArchivedElectrodeHistology.ProbeTrackPoint.insert(
            (histology.LabeledProbeTrack.Point & insertion_key).proj(
                ..., archival_time='"{}"'.format(archival_time)))

        if delete:
            with dj.config(safemode=False):
                (histology.ElectrodeCCFPosition & insertion_key).delete()
                (histology.LabeledProbeTrack & insertion_key).delete()
                (report.ProbeLevelDriftMap & insertion_key).delete()
                (report.ProbeLevelCoronalSlice & insertion_key).delete()
                (HistologyIngest & insertion_key).delete()


class HistologyFileError(Exception):
    """Raise when error encountered when ingesting probe insertion"""
    def __init__(self, msg=None):
        super().__init__('Histology File Error: \n{}'.format(msg))
    pass
