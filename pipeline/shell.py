# map-ephys interative shell module

import os
import sys
import logging
from code import interact
import time
import numpy as np
import pandas as pd
import datetime

import datajoint as dj

from pipeline import lab
from pipeline import experiment
from pipeline import ccf
from pipeline import ephys
from pipeline import histology
from pipeline import tracking
from pipeline import psth
from pipeline import report
from pipeline import export
from pipeline import publication


pipeline_modules = [lab, ccf, experiment, ephys, histology, tracking, psth]

log = logging.getLogger(__name__)


def usage_exit():
    print("usage: {p} [{c}] <args>"
          .format(p=os.path.basename(sys.argv[0]),
                  c='|'.join(list(actions.keys()))))
    sys.exit(0)


def logsetup(*args):
    level_map = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    level = level_map[args[0]] if args else logging.INFO

    logfile = dj.config.get('custom', {'logfile': None}).get('logfile', None)

    if logfile:
        handlers = [logging.StreamHandler(), logging.FileHandler(logfile)]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(level=logging.ERROR, handlers=handlers)

    log.setLevel(level)

    logging.getLogger('pipeline').setLevel(level)
    logging.getLogger('pipeline.psth').setLevel(level)
    logging.getLogger('pipeline.ccf').setLevel(level)
    logging.getLogger('pipeline.report').setLevel(level)
    logging.getLogger('pipeline.publication').setLevel(level)
    logging.getLogger('pipeline.ingest.behavior').setLevel(level)
    logging.getLogger('pipeline.ingest.ephys').setLevel(level)
    logging.getLogger('pipeline.ingest.tracking').setLevel(level)
    logging.getLogger('pipeline.ingest.histology').setLevel(level)


def ingest_behavior(*args):
    from pipeline.ingest import behavior as behavior_ingest
    behavior_ingest.BehaviorIngest().populate(display_progress=True)


def ingest_ephys(*args):
    from pipeline.ingest import ephys as ephys_ingest
    ephys_ingest.EphysIngest().populate(display_progress=True)


def ingest_tracking(*args):
    from pipeline.ingest import tracking as tracking_ingest
    tracking_ingest.TrackingIngest().populate(display_progress=True)


def ingest_histology(*args):
    from pipeline.ingest import histology as histology_ingest
    histology_ingest.HistologyIngest().populate(display_progress=True)


def load_animal(excel_fp, sheet_name='Sheet1'):
    df = pd.read_excel(excel_fp, sheet_name)
    df.columns = [cname.lower().replace(' ', '_') for cname in df.columns]

    subjects, water_restrictions, subject_ids = [], [], []
    for i, row in df.iterrows():
        if row.subject_id not in subject_ids and {'subject_id': row.subject_id} not in lab.Subject.proj():
            subject = {'subject_id': row.subject_id, 'username': row.username,
                       'cage_number': row.cage_number, 'date_of_birth': row.date_of_birth.date(),
                       'sex': row.sex, 'animal_source': row.animal_source}
            wr = {'subject_id': row.subject_id, 'water_restriction_number': row.water_restriction_number,
                  'cage_number': row.cage_number, 'wr_start_date': row.wr_start_date.date(),
                  'wr_start_weight': row.wr_start_weight}
            subject_ids.append(row.subject_id)
            subjects.append(subject)
            water_restrictions.append(wr)

    lab.Subject.insert(subjects)
    lab.WaterRestriction.insert(water_restrictions)

    log.info('Inserted {} subjects'.format(len(subjects)))
    log.info('Water restriction number: {}'.format([s['water_restriction_number'] for s in water_restrictions]))


def load_insertion_location(excel_fp, sheet_name='Sheet1'):
    from pipeline.ingest import behavior as behav_ingest

    df = pd.read_excel(excel_fp, sheet_name)
    df.columns = [cname.lower().replace(' ', '_') for cname in df.columns]

    insertion_locations = []
    recordable_brain_regions = []
    for i, row in df.iterrows():
        sess_key = experiment.Session & (behav_ingest.BehaviorIngest.BehaviorFile
                                         & {'subject_id': row.subject_id, 'session_date': row.session_date.date()}
                                         & 'behavior_file LIKE "%{}%{}_{:06}%"'.format(
                    row.water_restriction_number, row.session_date.date().strftime('%Y%m%d'), int(row.behaviour_time)))
        if sess_key:
            pinsert_key = dict(sess_key.fetch1('KEY'), insertion_number=row.insertion_number)
            if pinsert_key in ephys.ProbeInsertion.proj():
                if not (ephys.ProbeInsertion.InsertionLocation & pinsert_key):
                    insertion_locations.append(dict(pinsert_key, skull_reference=row.skull_reference,
                                                    ap_location=row.ap_location, ml_location=row.ml_location,
                                                    depth=row.depth, theta=row.theta, phi=row.phi, beta=row.beta))
                if not (ephys.ProbeInsertion.RecordableBrainRegion & pinsert_key):
                    recordable_brain_regions.append(dict(pinsert_key, brain_area=row.brain_area,
                                                         hemisphere=row.hemisphere))

    log.debug('InsertionLocation: {}'.format(insertion_locations))
    log.debug('RecordableBrainRegion: {}'.format(recordable_brain_regions))

    ephys.ProbeInsertion.InsertionLocation.insert(insertion_locations)
    ephys.ProbeInsertion.RecordableBrainRegion.insert(recordable_brain_regions)

    log.info('load_insertion_location - Number of insertions: {}'.format(len(insertion_locations)))


def populate_ephys(populate_settings={'reserve_jobs': True, 'display_progress': True}):

    log.info('experiment.PhotostimBrainRegion.populate()')
    experiment.PhotostimBrainRegion.populate(**populate_settings)

    log.info('ephys.UnitCoarseBrainLocation.populate()')
    ephys.UnitCoarseBrainLocation.populate(**populate_settings)

    log.info('ephys.UnitStat.populate()')
    ephys.UnitStat.populate(**populate_settings)

    log.info('ephys.UnitCellType.populate()')
    ephys.UnitCellType.populate(**populate_settings)


def populate_psth(populate_settings={'reserve_jobs': True, 'display_progress': True}):

    log.info('psth.UnitPsth.populate()')
    psth.UnitPsth.populate(**populate_settings)

    log.info('psth.PeriodSelectivity.populate()')
    psth.PeriodSelectivity.populate(**populate_settings)

    log.info('psth.UnitSelectivity.populate()')
    psth.UnitSelectivity.populate(**populate_settings)


def generate_report(populate_settings={'reserve_jobs': True, 'display_progress': True}):
    from pipeline import report
    for report_tbl in report.report_tables:
        log.info(f'Populate: {report_tbl.full_table_name}')
        report_tbl.populate(**populate_settings)


def sync_report():
    from pipeline import report
    for report_tbl in report.report_tables:
        log.info(f'Sync: {report_tbl.full_table_name} - From {report.store_location} - To {report.store_stage}')
        report_tbl.fetch()


def nuke_all():
    if 'nuclear_option' not in dj.config:
        raise RuntimeError('nuke_all() function not enabled')

    from pipeline.ingest import behavior as behavior_ingest
    from pipeline.ingest import ephys as ephys_ingest
    from pipeline.ingest import tracking as tracking_ingest
    from pipeline.ingest import histology as histology_ingest

    ingest_modules = [behavior_ingest, ephys_ingest, tracking_ingest,
                      histology_ingest]

    for m in reversed(ingest_modules):
        m.schema.drop()

    # production lab schema is not map project specific, so keep it.
    for m in reversed([m for m in pipeline_modules if m is not lab]):
        m.schema.drop()


def publish(*args):
    from pipeline import publication  # triggers ingest, so skipped
    publication.ArchivedRawEphys.populate()
    publication.ArchivedTrackingVideo.populate()


def export_recording(*args):
    if not args:
        print("usage: {} export-recording \"probe key\"\n"
              "  where \"probe key\" specifies a ProbeInsertion")
        return

    ik = eval(args[0])  # "{k: v}" -> {k: v}
    fn = args[1] if len(args) > 1 else None
    export.export_recording(ik, fn)


def shell(*args):
    interact('map shell.\n\nschema modules:\n\n  - {m}\n'
             .format(m='\n  - '.join(
                 '.'.join(m.__name__.split('.')[1:])
                 for m in pipeline_modules)),
             local=globals())


def ccfload(*args):
    ccf.CCFAnnotation.load_ccf_r3_20um()


def erd(*args):
    mods = (ephys, lab, experiment, tracking, psth, ccf, histology,
            report, publication)
    for mod in mods:
        modname = str().join(mod.__name__.split('.')[1:])
        fname = os.path.join('images', '{}.png'.format(modname))
        print('saving', fname)
        dj.ERD(mod, context={modname: mod}).save(fname)


def automate_computation():
    from pipeline import report
    populate_settings = {'reserve_jobs': True, 'suppress_errors': True, 'display_progress': True}
    while True:
        log.info('Populate for: Ephys - PSTH - Report')
        populate_ephys(populate_settings)
        populate_psth(populate_settings)
        generate_report(populate_settings)

        log.info('report.delete_outdated_probe_tracks()')
        report.delete_outdated_probe_tracks()

        # random sleep time between 5 to 10 minutes
        sleep_time = np.random.randint(300, 600)
        log.info('Sleep: {} minutes'.format(sleep_time / 60))
        time.sleep(sleep_time)


def sync_and_external_cleanup():
    if dj.config['custom'].get('allow_external_cleanup', False):
        from pipeline import report

        while True:
            sync_report()
            log.info('Report "report_store" external cleanup')
            report.schema.external['report_store'].delete(delete_external_files=True)
            time.sleep(1800)  # once every 30 minutes
    else:
        print("allow_external_cleanup disabled, set dj.config['custom']['allow_external_cleanup'] = True to enable")


actions = {
    'ingest-behavior': ingest_behavior,
    'ingest-ephys': ingest_ephys,
    'ingest-tracking': ingest_tracking,
    'ingest-histology': ingest_histology,
    'populate-psth': populate_psth,
    'publish': publish,
    'export-recording': export_recording,
    'generate-report': generate_report,
    'sync-report': sync_report,
    'shell': shell,
    'erd': erd,
    'ccfload': ccfload,
    'automate-computation': automate_computation,
    'automate-sync-and-cleanup': sync_and_external_cleanup,
    'load-insertion-location': load_insertion_location,
    'load-animal': load_animal
}
