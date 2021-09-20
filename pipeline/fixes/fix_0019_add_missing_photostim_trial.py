import os
import logging
import pathlib
from datetime import datetime
from decimal import Decimal
import datajoint as dj
import re
import numpy as np

from pipeline import lab, experiment, report
from pipeline.ingest import behavior as behavior_ingest
from pipeline.fixes import schema, FixHistory


os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

log = logging.getLogger(__name__)
log.setLevel('INFO')

"""
Many photostim trials are falsely not labeled as photostim trials,
 hence not added in experiment.PhotostimTrial and experiment.PhotostimEvent
This is due to a bug in comparing "this_trial_delay_duration == 1.2", 
 where `this_trial_delay_duration` could be 1.999999999 (due to floating error)
 
This fix reload the behavior files (for delay-response task only),
 and add the missing photostim trials
"""


@schema
class FixMissingPhotostimTrial(dj.Manual):
    definition = """ # This table accompanies fix_0019
    -> FixHistory
    -> experiment.SessionTrial
    ---
    is_added: bool  # was this trial incorrectly labeled as non-photostim and needed to be added
    """


def fix_photostim_trial(session_keys={}):
    """
    This fix applies to sessions ingested with the BehaviorIngest's make() only,
    as opposed to BehaviorBpodIngest (for Foraging Task)
    """
    sessions_2_update = (experiment.Session & behavior_ingest.BehaviorIngest
                         & (experiment.BehaviorTrial & 'task = "audio delay"')
                         & session_keys)
    sessions_2_update = sessions_2_update - FixMissingPhotostimTrial

    if not sessions_2_update:
        return

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}

    log.info('Fixing {} session(s)'.format(len(sessions_2_update)))
    for key in sessions_2_update.fetch('KEY'):
        success, missing_photostim_trials = _fix_one_session(key)
        if success:
            added_photostim_trials = [t['trial'] for t in missing_photostim_trials]
            FixHistory.insert1(fix_hist_key, skip_duplicates=True)
            FixMissingPhotostimTrial.insert([{
                **fix_hist_key, **tkey,
                'is_added': 1 if tkey['trial'] in added_photostim_trials else 0}
                for tkey in (experiment.SessionTrial & key).fetch('KEY')])


def _fix_one_session(key):
    log.info('Running fix for session: {}'.format(key))
    key = (experiment.Session & key).fetch1()
    behavior_fp = (behavior_ingest.BehaviorIngest.BehaviorFile & key).fetch1('behavior_file')

    for rig, rig_path, _ in behavior_ingest.get_behavior_paths():
        try:
            path = next(pathlib.Path(rig_path).rglob(f'{behavior_fp}*'))
            break
        except StopIteration:
            pass
    else:
        log.warning(f'Behavior file not found on this machine: {behavior_fp}')
        return None, None

    # distinguishing "delay-response" task or "multi-target-licking" task
    task_type = behavior_ingest.detect_task_type(path)
    assert task_type == 'delay-response'

    # skip too small behavior file (only for 'delay-response' task)
    if task_type == 'delay-response' and os.stat(path).st_size / 1024 < 1000:
        log.info('skipping file {} - too small'.format(path))
        return None, None

    log.debug('loading file {}'.format(path))

    # Read from behavior file and parse all trial info (the heavy lifting here)
    _, rows = behavior_ingest.BehaviorIngest._load(key, path, task_type)

    photostim_trials = [{**r, **key} for r in rows['photostim_trial']]
    photostim_trial_events = [{**r, **key} for r in rows['photostim_trial_event']]

    current_photostim_trials = (experiment.PhotostimTrial & key).fetch('trial')
    missing_photostim_trials = [r for r in photostim_trials
                                if r['trial'] not in current_photostim_trials]

    if missing_photostim_trials:
        # Photostim Insertion
        photostims = behavior_ingest.photostims

        photostim_ids = np.unique(
            [r['photo_stim'] for r in rows['photostim_trial_event']])

        unknown_photostims = np.setdiff1d(
            photostim_ids, list(photostims.keys()))

        if unknown_photostims:
            raise ValueError(
                'Unknown photostim protocol: {}'.format(unknown_photostims))

        if photostim_ids.size > 0:
            log.info('BehaviorIngest.make(): ... experiment.Photostim')
            for stim in photostim_ids:
                experiment.Photostim.insert1(
                    dict(key, **photostims[stim]), ignore_extra_fields=True)

                experiment.Photostim.PhotostimLocation.insert(
                    (dict(key, **loc,
                          photo_stim=photostims[stim]['photo_stim'])
                     for loc in photostims[stim]['locations']),
                    ignore_extra_fields=True)

        log.info('BehaviorIngest.make(): ... experiment.PhotostimTrial')
        experiment.PhotostimTrial.insert(photostim_trials,
                                         ignore_extra_fields=True,
                                         allow_direct_insert=True)

        log.info('BehaviorIngest.make(): ... experiment.PhotostimTrialEvent')
        experiment.PhotostimEvent.insert(photostim_trial_events,
                                         ignore_extra_fields=True,
                                         allow_direct_insert=True)

    return True, missing_photostim_trials


if __name__ == '__main__':
    fix_photostim_trial()