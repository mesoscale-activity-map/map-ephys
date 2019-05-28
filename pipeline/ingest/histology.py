
import os
import logging

import numpy as np
import scipy.io as scio
import datajoint as dj

from pipeline import lab
from pipeline import ephys
from pipeline import experiment
from pipeline import ccf
from pipeline.ingest import ephys as ephys_ingest

from code import interact
from collections import ChainMap

log = logging.getLogger(__name__)

schema = dj.schema(dj.config['custom'].get(
    'ingest.histology.database',
    '{}_ingestHistology'.format(dj.config['database.user'])))


@schema
class HistologyIngest(dj.Imported):
    definition = """
    -> ephys_ingest.EphysIngest
    """

    class HistologyFile(dj.Part):
        definition = """
        -> master
        electrode_group:        tinyint         # electrode group
        histology_file:         varchar(255)    # rig file subpath
        """

    def make(self, key):
        '''
        HistologyIngest .make() function
        '''
        # todo: check the length of the `site.ont.name` variable,
        #   and only ingest the sites with an ontology associated to it.
        log.info('HistologyIngest().make(): key: {}'.format(key))

        session = (experiment.Session & key).fetch1()

        rigpath = ephys_ingest.EphysDataPath().fetch1('data_path')
        subject_id = session['subject_id']
        session_date = session['session_date']
        water = (
            lab.WaterRestriction() & {'subject_id': subject_id}
        ).fetch1('water_restriction_number')

        egmap = {e['electrode_group']: e
                 for e in (ephys.ElectrodeGroup & session).fetch('KEY')}

        errlabel = ccf.CCFLabel.CCF_R3_20UM_ERROR

        for probe in range(1, 3):

            log.info('... probe {} position ingest.'.format(probe))

            if probe not in egmap:
                log.info('... probe {} ephys N/A, skipping.'.format(probe))
                continue

            # subpaths like:
            # directory: {h2o}/{yyyy-mm-dd}/histology
            # file: landmarks_{h2o}_{YYYYMMDD}_{probe}_siteInfo.mat

            directory = os.path.join(
                water, session_date.strftime('%Y-%m-%d'), 'histology')
            file = 'landmarks_{}_{}_{}_siteInfo.mat'.format(
                water, session['session_date'].strftime('%Y%m%d'), probe)
            subpath = os.path.join(directory, file)
            fullpath = os.path.join(rigpath, subpath)

            if not os.path.exists(fullpath):
                log.info('... probe {} histology file {} N/A, skipping.'
                         .format(probe, fullpath))
                continue

            log.info('... found probe {} histology file {}'.format(
                probe, fullpath))

            hist = scio.loadmat(fullpath)['site']

            # probe CCF 3D positions
            pos = hist['pos'][0][0][0]
            pos_x = pos['x'][0].T[0]
            pos_y = pos['y'][0].T[0]
            pos_z = pos['z'][0].T[0]
            pos_xyz = np.array((pos_x, pos_y, pos_z)).T * 20

            # probe CCF regions
            names = hist['ont'][0][0]['name'][0][0].T[0]
            named = {np.str_: True, np.ndarray: False}  # XXX: np.where?
            valid = [named[type(n[0])] for n in names]
            goodn = np.where(np.array(valid))[0]

            # XXX: to verify - electrode off-by-one in ingest (e.g. mat->py)??
            electrodes = (ephys.ElectrodeGroup.Electrode
                          & egmap[probe]).fetch(order_by='electrode asc')

            # interact('histoloading', local=dict(ChainMap(locals(), globals())))

            # XXX: off by one in ephys.ElectrodeGroup 'builder' routine?
            #   .. relatedly, we index pos_xyz by 'goodn' directly,
            #   .. rather than via electrodes[goodn]['electrode']
            recs = ((*l[0], ccf.CCFLabel.CCF_R3_20UM_ID, *l[1]) for l in
                    zip(electrodes[goodn], pos_xyz[goodn]))

            # ideally:
            # ephys.ElectrodeGroup.ElectrodePosition.insert(
            #     recs, allow_direct_insert=True)
            # but hitting ccf coordinate issues..:

            for r in recs:
                log.debug('... adding probe/position: {}'.format(r))
                try:
                    ephys.ElectrodeGroup.ElectrodePosition.insert1(
                        r, ignore_extra_fields=True, allow_direct_insert=True)
                except Exception as e:
                    log.warning('... ERROR!')
                    ephys.ElectrodeGroup.ElectrodePositionError.insert1(
                        r, ignore_extra_fields=True, allow_direct_insert=True)

            log.info('... ok.')

        log.info('HistologyIngest().make(): {} complete.'.format(key))
        self.insert1(key)
