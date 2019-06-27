
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

        egmap = {e['insertion_number']: e
                 for e in (ephys.ProbeInsertion
                           * lab.ElectrodeConfig.ElectrodeGroup
                           & session).fetch('KEY')}

        if not len(egmap):
            log.info('... no probe information. skipping.'.format(key))
            return

        rigpath = ephys_ingest.EphysDataPath().fetch1('data_path')
        subject_id = session['subject_id']
        session_date = session['session_date']
        water = (
            lab.WaterRestriction() & {'subject_id': subject_id}
        ).fetch1('water_restriction_number')

        directory = pathlib.Path(
            rigpath, water, session_date.strftime('%Y-%m-%d'), 'histology')

        for probe in range(1, 3):

            probefile = 'landmarks_{}_{}_{}_siteInfo.mat'.format(
                water, session['session_date'].strftime('%Y%m%d'), probe)
            trackfile = 'landmarks_{}_{}_{}.csv'.format(
                water, session['session_date'].strftime('%Y%m%d'), probe)

            probepath = directory / probefile
            trackpath = directory / trackfile

            try:

                self._load_histology_probe(
                    key, session, egmap, probe, probepath)

                self._load_histology_track(
                    key, session, egmap, probe, trackpath)

            except StopIteration:
                pass

        log.info('HistologyIngest().make(): {} complete.'.format(key))
        self.insert1(key)

    def _load_histology_probe(self, key, session, egmap, probe, probepath):

        sz = 20   # 20um voxel size

        log.info('... probe {} position ingest.'.format(probe))

        if probe not in egmap:
            msg = '... probe {} ephys N/A, skipping.'.format(probe)
            log.info(msg)
            raise StopIteration(msg)

        if not probepath.exists():
            msg = '... probe {} histology file {} N/A, skipping.'.format(
                probe, probepath)
            log.info(msg)
            raise StopIteration(msg)  # skip to next probe

        log.info('... found probe {} histology file {}'.format(
            probe, probepath))

        hist = scio.loadmat(
            probepath, struct_as_record=False, squeeze_me=True)['site']

        # probe CCF 3D positions
        pos_xyz = np.vstack([hist.pos.x, hist.pos.y, hist.pos.z,
                             hist.warp.x, hist.warp.y, hist.warp.z]).T * sz

        # probe CCF regions
        names = hist.ont.name
        valid = [isinstance(n, (str,)) for n in names]
        goodn = np.where(np.array(valid))[0]

        electrodes = (ephys.ProbeInsertion.proj() * lab.Probe.Electrode.proj()
                      & egmap[probe]).fetch(order_by='electrode asc')

        recs = ((*l[0], ccf.CCFLabel.CCF_R3_20UM_ID, *l[1]) for l in
                zip(electrodes[goodn], pos_xyz[goodn]))

        # ideally ElectrodePosition.insert(...) but some are outside of CCF...
        log.info('inserting channel ccf position')
        histology.ElectrodeCCFPosition.insert1(
            egmap[probe], ignore_extra_fields=True)

        for r in recs:
            log.debug('... adding probe/position: {}'.format(r))
            try:
                histology.ElectrodeCCFPosition.ElectrodePosition.insert1(
                    r, ignore_extra_fields=True, allow_direct_insert=True)
            except Exception as e:  # XXX: no way to be more precise in dj
                log.warning('... ERROR!: {}'.format(repr(e)))
                histology.ElectrodeCCFPosition.ElectrodePositionError.insert1(
                    r, ignore_extra_fields=True, allow_direct_insert=True)

        log.info('... ok.')

    def _load_histology_track(self, key, session, egmap, probe, trackpath):

        conv = (('landmark_name', str), ('warp', lambda x: x == 'true'),
                ('subj_x', float), ('subj_y', float), ('subj_z', float),
                ('ccf_x', float), ('ccf_y', float), ('ccf_z', float))

        if not trackpath.exists():
            msg = '... probe {} track file {} N/A, skipping.'.format(
                probe, trackpath)
            log.info(msg)
            raise Exception(msg)  # error: no histology without track info

        recs = []
        with open(trackpath, newline='') as f:
            rdr = csv.reader(f)
            for row in rdr:
                assert len(row) == 8
                rec = {c[0]: c[1](d) for c, d in zip(conv, row)}
                recs.append(rec)

        # Subject -> CCF Transformation

        top = {'subject_id': session['subject_id']}

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

        top = {**egmap[probe], 'labeling_date': None, 'dye_color': None}

        histology.LabeledProbeTrack.insert1(
            top, ignore_extra_fields=True, allow_direct_insert=True)

        histology.LabeledProbeTrack.Point.insert(
            ({**top, 'order': rec[0], **rec[1]} for rec in
             enumerate((r for r in recs if r['warp'] is False))),
            ignore_extra_fields=True, allow_direct_insert=True)
