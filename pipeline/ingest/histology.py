
import logging
import pathlib
import csv

import numpy as np
import scipy.io as scio
import datajoint as dj

from pipeline import lab
from pipeline import ephys
from pipeline import experiment
from pipeline import ccf
from pipeline import histology
from pipeline.ingest import ephys as ephys_ingest

from .. import get_schema_name

schema = dj.schema(get_schema_name('ingest_histology'))

log = logging.getLogger(__name__)


@schema
class HistologyIngest(dj.Imported):
    definition = """
    -> ephys_ingest.EphysIngest
    """

    class HistologyFile(dj.Part):
        definition = """
        -> master
        probe_insertion_number:         tinyint         # electrode group
        histology_file:                 varchar(255)    # rig file subpath
        """

    class LandmarkFile(dj.Part):
        definition = """
        -> master
        probe_insertion_number:         tinyint         # electrode group
        landmark_file:                  varchar(255)    # rig file subpath
        """

    def make(self, key):
        '''
        HistologyIngest .make() function
        '''
        # TODO: check the length of the `site.ont.name` variable,
        #   and only ingest the sites with an ontology associated to it.
        log.info('HistologyIngest().make(): key: {}'.format(key))

        session = (experiment.Session & key).fetch1()

        rigpath = ephys_ingest.EphysDataPath().fetch1('data_path')
        subject_id = session['subject_id']
        session_date = session['session_date']
        water = (
            lab.WaterRestriction() & {'subject_id': subject_id}
        ).fetch1('water_restriction_number')

        egmap = {e['insertion_number']: e

                 for e in (ephys.ProbeInsertion
                           * lab.ElectrodeConfig.ElectrodeGroup
                           & session).fetch('KEY')}

        for probe in range(1, 3):

            self.load_histology_probe(key, session, probe, egmap, rigpath,
                                      subject_id, session_date, water)

            self.load_histology_track(key, session, probe, egmap, rigpath,
                                      subject_id, session_date, water)

        log.info('HistologyIngest().make(): {} complete.'.format(key))
        self.insert1(key)

    def load_histology_probe(self, key, session, probe, egmap, rigpath,
                             subject_id, session_date, water):

        sz = 20   # 20um voxel size

        log.info('... probe {} position ingest.'.format(probe))

        if probe not in egmap:
            msg = '... probe {} ephys N/A, skipping.'.format(probe)
            log.info(msg)
            raise StopIteration(msg)

        # subpaths like:
        # directory: {h2o}/{yyyy-mm-dd}/histology
        # file: landmarks_{h2o}_{YYYYMMDD}_{probe}_siteInfo.mat

        directory = pathlib.Path(
            water, session_date.strftime('%Y-%m-%d'), 'histology')

        file = 'landmarks_{}_{}_{}_siteInfo.mat'.format(
            water, session['session_date'].strftime('%Y%m%d'), probe)
        subpath = directory / file
        fullpath = rigpath / subpath

        if not fullpath.exists():
            msg = '... probe {} histology file {} N/A, skipping.'.format(
                probe, fullpath)
            log.info(msg)
            raise StopIteration(msg)

        log.info('... found probe {} histology file {}'.format(
            probe, fullpath))

        hist = scio.loadmat(
            fullpath, struct_as_record=False, squeeze_me=True)['site']

        # probe CCF 3D positions
        pos_xyz = np.vstack([hist.pos.x, hist.pos.y, hist.pos.z]).T * sz

        # probe CCF regions
        names = hist.ont.name
        valid = [isinstance(n, (str,)) for n in names]
        goodn = np.where(np.array(valid))[0]

        electrodes = (ephys.ProbeInsertion * lab.ElectrodeConfig.Electrode
                      & egmap[probe]).fetch(order_by='electrode asc')

        # XXX: we index pos_xyz by 'goodn' directly,
        #   .. rather than via electrodes[goodn]['electrode']

        recs = ((*l[0], ccf.CCFLabel.CCF_R3_20UM_ID, *l[1]) for l in
                zip(electrodes[goodn], pos_xyz[goodn]))

        # ideally:
        # ephys.ElectrodeGroup.ElectrodePosition.insert(
        #     recs, allow_direct_insert=True)
        # but hitting ccf coordinate issues..

        log.info('inserting channel ccf position')
        ephys.ElectrodeCCFPosition.insert1(
            egmap[probe], ignore_extra_fields=True)

        for r in recs:
            log.debug('... adding probe/position: {}'.format(r))
            try:
                histology.ElectrodeCCFPosition.ElectrodePosition.insert1(
                    r, ignore_extra_fields=True, allow_direct_insert=True)
            except Exception as e:  # XXX: no way to be more precise in dj
                log.warning('... ERROR!'.format(repr(e)))
                ephys.ElectrodeCCFPosition.ElectrodePositionError.insert1(
                    r, ignore_extra_fields=True, allow_direct_insert=True)

        log.info('... ok.')

    def load_histology_track(key, session, probe, egmap, rigpath, subject_id,
                             session_date, water):

        conv = (('landmark_name', str), ('warp', bool),
                ('raw_x', float), ('raw_y', float), ('raw_z', float),
                ('x', float), ('y', float), ('z', float))

        directory = pathlib.Path(
            water, session_date.strftime('%Y-%m-%d'), 'histology')

        file = 'landmarks_{}_{}_{}_siteInfo.csv'.format(
            water, session['session_date'].strftime('%Y%m%d'), probe)
        subpath = directory / file
        fullpath = rigpath / subpath

        recs = []
        with open(fullpath, newline='') as f:
            rdr = csv.reader(f)
            for row in rdr:
                assert len(row) == 8
                rec = {c[0]: c[1](d) for c, d in zip(conv, row)}
                recs.append(rec)

        # Raw -> CCF Transformation

        top = {'subject_id': subject_id}

        if not (histology.RawToCCFTransformation & top).fetch(limit=1):

            log.info('... adding new raw -> ccf coordinates')
            histology.RawToCCFTransformation.insert1(
                top, allow_direct_insert=True)

            histology.RawToCCFTransformation.Landmark.insert1(
                ({**top, **rec} for rec in
                 (r for r in recs if r['warp'] is True)),
                allow_direct_insert=True, ignore_extra_fields=True)

        else:
            log.debug('... skipping raw -> ccf coordinates')

        # LabeledProbeTrack

        top = {**egmap[probe], 'labeling_date': None, 'dye_color': None}

        histology.LabeledProbeTrack.insert1(
            top, ignore_extra_fields=True, allow_direct_insert=True)

        histology.LabeledProbeTrack.Point.insert1(
            ({**top, **rec} for rec in
             (r for r in recs if r['warp'] is False)),
            ignore_extra_fields=True, allow_direct_insert=True)
