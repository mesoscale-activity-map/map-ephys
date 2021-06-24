# map-ephys interative shell module

import os
import sys
import logging
from code import interact
from datetime import datetime
from textwrap import dedent
import time
import numpy as np
import pandas as pd
import re
import datajoint as dj
from pymysql.err import OperationalError


from pipeline import (lab, experiment, tracking, ephys, report, psth, psth_foraging, ccf,
                      histology, export, publication, globus, foraging_analysis,
                      get_schema_name)

pipeline_modules = [lab, ccf, experiment, ephys, publication, report,
                    foraging_analysis, histology, tracking, psth]

log = logging.getLogger(__name__)


def usage_exit():
    print(dedent(
        '''
        usage: {} cmd args

        where 'cmd' is one of:

        {}
        ''').lstrip().rstrip().format(
            os.path.basename(sys.argv[0]),
            str().join("  - {}: {}\n".format(k, v[1])
                       for k, v in actions.items())))

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

    datefmt = '%Y-%m-%d %H:%M:%S'
    msgfmt = '%(asctime)s:%(levelname)s:%(module)s:%(funcName)s:%(message)s'

    logging.basicConfig(format=msgfmt, datefmt=datefmt, level=logging.ERROR,
                        handlers=handlers)

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


def ingest_foraging_behavior(*args):
    from pipeline.ingest import behavior as behavior_ingest
    behavior_ingest.BehaviorBpodIngest().populate(
        display_progress=True, reserve_jobs=True)


def ingest_ephys(*args):
    from pipeline.ingest import ephys as ephys_ingest
    ephys_ingest.EphysIngest().populate(display_progress=True)


def ingest_tracking(*args):
    from pipeline.ingest import tracking as tracking_ingest
    tracking_ingest.TrackingIngest().populate(display_progress=True)


def ingest_histology(*args):
    from pipeline.ingest import histology as histology_ingest
    histology_ingest.HistologyIngest().populate(display_progress=True)


def ingest_all(*args):

    log.info('running auto ingest')

    ingest_behavior(args)
    ingest_ephys(args)
    ingest_tracking(args)
    ingest_histology(args)

    def_sheet = {'recording_notes_spreadsheet': None,
                 'recording_notes_sheet_name': None}

    sfile = dj.config.get('custom', def_sheet).get('recording_notes_spreadsheet')
    sname = dj.config.get('custom', def_sheet).get('recording_notes_sheet_name')

    if sfile:
        load_insertion_location(sfile, sheet_name=sname)


def load_animal(excel_fp, sheet_name='Sheet1'):
    df = pd.read_excel(excel_fp, sheet_name, engine='openpyxl')
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
    log.info('loading probe insertions from spreadsheet {}'.format(excel_fp))

    try:
        df = pd.read_excel(excel_fp, sheet_name, engine='openpyxl')
        from_excel = True
    except:
        df = pd.read_csv(excel_fp)
        from_excel = False
        
            
    df.columns = [cname.lower().replace(' ', '_') for cname in df.columns]

    insertion_locations = []
    recordable_brain_regions = []
    for i, row in df.iterrows():
        try:
            int(row.behaviour_time)
            valid_btime = True
        except ValueError:
            log.debug('Invalid behaviour time: {} - try single-sess per day'.format(row.behaviour_time))
            valid_btime = False
        
        session_date = row.session_date.date() if from_excel else datetime.strptime(row.session_date,'%Y-%m-%d')
        if 'foraging' in row.project:
            behavior_ingest_table = behav_ingest.BehaviorBpodIngest
            bf_reg_exp = '(\d{8}-\d{6})'
            bf_datetime_format = '%Y%m%d-%H%M%S'
        else:
            behavior_ingest_table = behav_ingest.BehaviorIngest
            bf_reg_exp = '(\d{8}_\d{6}).mat'
            bf_datetime_format = '%Y%m%d_%H%M%S'

        if valid_btime:
            sess_key = experiment.Session & (
                behavior_ingest_table.BehaviorFile
                & {'subject_id': row.subject_id, 'session_date': session_date}
                & 'behavior_file LIKE "%{}%{}_{:06}%"'.format(
                    row.water_restriction_number, session_date.strftime('%Y%m%d'), int(row.behaviour_time)))
        else:
            sess_key = False

        if not sess_key:
            sess_key = experiment.Session & {'subject_id': row.subject_id, 'session_date': session_date}
            if len(sess_key) == 1:
                # case of single-session per day - ensuring session's datetime matches the filename
                # session_datetime and datetime from filename should not be more than 3 hours apart
                bf = (behavior_ingest_table.BehaviorFile & sess_key).fetch('behavior_file')[0]
                bf_datetime = re.search(bf_reg_exp, bf).groups()[0]
                bf_datetime = datetime.strptime(bf_datetime, bf_datetime_format)
                s_datetime = sess_key.proj(s_dt='cast(concat(session_date, " ", session_time) as datetime)').fetch1('s_dt')
                if abs((s_datetime - bf_datetime).total_seconds()) > 10800:  # no more than 3 hours
                    log.debug('Unmatched sess_dt ({}) and behavior_dt ({}). Skipping...'.format(s_datetime, bf_datetime))
                    continue
            else:
                continue

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


def load_ccf(*args):
    ccf.CCFBrainRegion.load_regions()
    ccf.CCFAnnotation.load_ccf_annotation()


def load_meta_foraging():  
    '''
    Load metadata for the foraging task
    Adapted from Marton's code: https://github.com/rozmar/DataPipeline/blob/master/ingest/datapipeline_metadata.py
    '''
    import pathlib
    meta_dir = dj.config.get('custom', {}).get('behavior_bpod', []).get('meta_dir')
    meta_lab_dir = dj.config.get('custom', {}).get('behavior_bpod', []).get('meta_lab_dir')
    
    # --- Add experimenters ---
    print('Adding experimenters...')
    df_experimenters = pd.read_csv(pathlib.Path(meta_lab_dir) / 'Experimenter.csv')
    
    duplicate_num = 0
    for experimenter in df_experimenters.iterrows():
        experimenter = experimenter[1]
        experimenternow = {'username':experimenter['username'],'fullname':experimenter['fullname']}
        try:
            lab.Person().insert1(experimenternow)
            print('  added experimenter: ',experimenternow['username'])
        except dj.errors.DuplicateError:
            duplicate_num += 1
            #  print('  duplicate. experimenter: ',experimenternow['username'], ' already exists')
    print(f'  {duplicate_num} experimenters already exist')
    
    # --- Add rigs ---
    print('Adding rigs... ')
    df_rigs = pd.read_csv(pathlib.Path(meta_lab_dir) / 'Rig.csv')
    
    duplicate_num = 0
    for rig in df_rigs.iterrows():
        rig = rig[1]
        rignow = {'rig':rig['rig'],'room':rig['room'],'rig_description':rig['rig_description']}
        try:
            lab.Rig().insert1(rignow)
            print('  added rig: ', rignow['rig'])
        except dj.errors.DuplicateError:
            duplicate_num += 1
            # print('  duplicate. rig: ',rignow['rig'], ' already exists')
    print(f'  {duplicate_num} rigs already exist')
            
    # --- Add viruses ---
    # Not implemented for now.  Han
  
    # --- Add subjects and water restrictions ---
    print('Adding subjects and water restrictions...')
    df_surgery = pd.read_csv(pathlib.Path(meta_dir) / 'Surgery.csv')
    
    # For each entry
    duplicate_subject_num = 0
    duplicate_WR_num = 0

    for item in df_surgery.iterrows():
        item = item[1]
        
        if item['project'] == 'foraging' and (item['status'] == 'training' or item['status'] == 'sacrificed'):
            
            # -- Add lab.Subject() --
            subjectdata = {
                    'subject_id': item['animal#'],
                    'cage_number': item['cage#'],
                    'date_of_birth': item['DOB'],
                    'sex': item['sex'],
                    'username': item['experimenter'],
                    }
            try:
                lab.Subject().insert1(subjectdata)
                print('  added subject: ', item['animal#'])
            except dj.errors.DuplicateError:
                duplicate_subject_num += 1
                # print('  duplicate. animal :',item['animal#'], ' already exists')
                
            # -- Add lab.Surgery() --
            # Not implemented. Han

            # -- Virus injection --
            # Not implemented. Han
                
            # -- Add lab.WaterRestriction() --
            if item['ID']:
                # Get water restriction start date and weight
                subject_csv = pathlib.Path(meta_dir) / '{}.csv'.format(item['ID'])
                if subject_csv.exists():
                    df_wr = pd.read_csv(subject_csv)
                else:
                    print('  No metadata csv found for {}'.format(item['ID']))
                    continue
            
                wrdata = {
                        'subject_id':item['animal#'],
                        'water_restriction_number': item['ID'],
                        'cage_number': item['cage#'],
                        'wr_start_date': df_wr['Date'][0],
                        'wr_start_weight': df_wr['Weight'][0],
                        }
                try:
                    lab.WaterRestriction().insert1(wrdata)
                    print('  added WR: ', item['ID'])
                except dj.errors.DuplicateError:
                    duplicate_WR_num += 1
                    # print('  duplicate. water restriction:', item['ID'], ' already exists')
                    
    print(f'  {duplicate_subject_num} subjects and {duplicate_WR_num} WRs already exist')
    

def populate_ephys(populate_settings={'reserve_jobs': True, 'display_progress': True}):

    log.info('experiment.PhotostimBrainRegion.populate()')
    experiment.PhotostimBrainRegion.populate(**populate_settings)

    log.info('ephys.UnitCoarseBrainLocation.populate()')
    ephys.UnitCoarseBrainLocation.populate(**populate_settings)

    log.info('ephys.UnitStat.populate()')
    ephys.UnitStat.populate(**populate_settings)

    log.info('ephys.UnitCellType.populate()')
    ephys.UnitCellType.populate(**populate_settings)

    log.info('ephys.MAPClusterMetric.populate()')
    ephys.MAPClusterMetric.populate(**populate_settings)

    log.info('ephys.MAPClusterMetric.populate()')
    histology.InterpolatedShankTrack.populate(**populate_settings)


def populate_psth(populate_settings={'reserve_jobs': True, 'display_progress': True}):

    log.info('psth.UnitPsth.populate()')
    psth.UnitPsth.populate(**populate_settings)

    log.info('psth.PeriodSelectivity.populate()')
    psth.PeriodSelectivity.populate(**populate_settings)

    log.info('psth.UnitSelectivity.populate()')
    psth.UnitSelectivity.populate(**populate_settings)


def populate_foraging_psth(populate_settings={'reserve_jobs': True, 'display_progress': True}):

    log.info('psth.UnitPsth.populate()')
    psth_foraging.UnitPsth.populate(**populate_settings)

    log.info('psth.PeriodSelectivity.populate()')
    psth_foraging.PeriodSelectivity.populate(**populate_settings)

    log.info('psth.UnitSelectivity.populate()')
    psth_foraging.UnitSelectivity.populate(**populate_settings)


def populate_foraging_analysis(populate_settings={'reserve_jobs': True, 'display_progress': True}):
    log.info('foraging_analysis.TrialStats.populate()')
    foraging_analysis.TrialStats.populate(**populate_settings)

    log.info('foraging_analysis.BlockStats.populate()')
    foraging_analysis.BlockStats.populate(**populate_settings)

    log.info('foraging_analysis.SessionTaskProtocol.populate()')
    foraging_analysis.SessionTaskProtocol.populate(**populate_settings)

    log.info('foraging_analysis.SessionStats.populate()')
    foraging_analysis.SessionStats.populate(**populate_settings)

    log.info('foraging_analysis.BlockFraction.populate()')
    foraging_analysis.BlockFraction.populate(**populate_settings)

    log.info('foraging_analysis.SessionMatching.populate()')
    foraging_analysis.SessionMatching.populate(**populate_settings)

    log.info('foraging_analysis.BlockEfficiency.populate()')
    foraging_analysis.BlockEfficiency.populate(**populate_settings)


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


def publication_login(*args):
    cfgname = args[0] if len(args) else 'local'

    if 'custom' in dj.config and 'globus.token' in dj.config['custom']:
        del dj.config['custom']['globus.token']

    from pipeline.globus import GlobusStorageManager

    GlobusStorageManager()

    if cfgname == 'local':
        dj.config.save_local()
    elif cfgname == 'global':
        dj.config.save_global()
    else:
        log.warning('unknown configuration {}. not saving'.format(cfgname))


def publication_publish_ephys(*args):
    publication.ArchivedRawEphys.populate()


def publication_publish_video(*args):
    publication.ArchivedTrackingVideo.populate()


def publication_discover_ephys(*args):
    publication.ArchivedRawEphys.discover()


def publication_discover_video(*args):
    publication.ArchivedTrackingVideo.discover()


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


def erd(*args):
    report = dj.create_virtual_module('report', get_schema_name('report'))
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
        populate_foraging_analysis(populate_settings)
        generate_report(populate_settings)

        log.info('report.delete_outdated_session_plots()')
        try:
            report.delete_outdated_session_plots()
        except OperationalError as e:  # in case of mysql deadlock - code: 1213
            if e.args[0] == 1213:
                pass

        log.info('report.delete_outdated_project_plots()')
        try:
            report.delete_outdated_project_plots()
        except OperationalError as e:  # in case of mysql deadlock - code: 1213
            if e.args[0] == 1213:
                pass

        log.info('Delete empty ingestion tables')
        delete_empty_ingestion_tables()

        # random sleep time between 5 to 10 minutes
        sleep_time = np.random.randint(300, 600)
        log.info('Sleep: {} minutes'.format(sleep_time / 60))
        time.sleep(sleep_time)


def delete_empty_ingestion_tables():
    from pipeline.ingest import ephys as ephys_ingest
    from pipeline.ingest import tracking as tracking_ingest
    from pipeline.ingest import histology as histology_ingest

    with dj.config(safemode=False):
        try:
            (ephys_ingest.EphysIngest & (ephys_ingest.EphysIngest
                                         - ephys.ProbeInsertion).fetch('KEY')).delete()
            (tracking_ingest.TrackingIngest & (tracking_ingest.TrackingIngest
                                               - tracking.Tracking).fetch('KEY')).delete()
            (histology_ingest.HistologyIngest & (histology_ingest.HistologyIngest
                                                 - histology.ElectrodeCCFPosition).fetch('KEY')).delete()
        except OperationalError as e:  # in case of mysql deadlock - code: 1213
            if e.args[0] == 1213:
                pass


def sync_and_external_cleanup():
    if dj.config['custom'].get('allow_external_cleanup', False):
        from pipeline import report

        while True:
            log.info('Sync report')
            sync_report()
            log.info('Report "report_store" external cleanup')
            report.schema.external['report_store'].delete(delete_external_files=True)
            log.info('Delete filepath-exists error jobs')
            # This happens when workers attempt to regenerate the plots when the corresponding external files has not yet been deleted
            (report.schema.jobs & 'error_message LIKE "DataJointError: A different version of%"').delete()
            time.sleep(1800)  # once every 30 minutes
    else:
        print("allow_external_cleanup disabled, set dj.config['custom']['allow_external_cleanup'] = True to enable")


def loop(*args):
    ''' run subsequent 'cmd args ..'. command in a loop '''
    cmd, *newargs = args

    cmd_fn = actions[cmd][0]

    log.info('running {} in a loop'.format(cmd))

    nruns = nerrs = 0

    while True:
        log.info('{} loop round {} (nerrs: {})'.format(cmd, nruns, nerrs))
        try:
            cmd_fn(*args)
        except Exception as e:
            log.info('{} loop round {} recieved error: {}'.format(
                cmd, nruns, repr(e)))
            log.debug('exception:', exc_info=1)
            nerrs += 1

        nruns += 1

        log.info('{} loop - waiting 5 seconds before next round'.format(cmd))
        time.sleep(5)


actions = {
    'ingest-behavior': (ingest_behavior, 'ingest behavior data'),
    'ingest-foraging': (ingest_behavior, 'ingest foraging behavior data'),
    'ingest-ephys': (ingest_ephys, 'ingest ephys data'),
    'ingest-tracking': (ingest_tracking, 'ingest tracking data'),
    'ingest-histology': (ingest_histology, 'ingest histology data'),
    'ingest-all': (ingest_all, 'run auto ingest job (load all types)'),
    'populate-psth': (populate_psth, 'populate psth schema'),
    'publication-login': (publication_login, 'login to globus'),
    'publication-publish-ephys': (publication_publish_ephys,
                                  'publish raw ephys data to globus'),
    'publication-publish-video': (publication_publish_video,
                                  'publish raw video data to globus'),
    'publication-discover-ephys': (publication_discover_ephys,
                                   'discover raw ephys data on globus'),
    'publication-discover-video': (publication_discover_video,
                                   'discover raw video data on globus'),
    'export-recording': (export_recording, 'export data to .mat'),
    'generate-report': (generate_report, 'run report generation logic'),
    'sync-report': (sync_report, 'sync report data locally'),
    'shell': (shell, 'interactive shell'),
    'erd': (erd, 'write DataJoint ERDs to files'),
    'load-ccf': (load_ccf, 'load CCF reference atlas'),
    'automate-computation': (automate_computation, 'run report worker job'),
    'automate-sync-and-cleanup': (sync_and_external_cleanup,
                                  'run report cleanup job'),
    'load-insertion-location': (load_insertion_location,
                                'load ProbeInsertions from .xlsx'),
    'load-animal': (load_animal, 'load subject data from .xlsx'),
    'load-meta-foraging': (load_meta_foraging, 'load foraging meta information from .csv'),
    'loop': (loop, 'run subsequent command and args in a loop')
}
