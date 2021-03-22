import os
import logging
import pathlib
from datetime import datetime
import datajoint as dj

from pipeline import experiment
from pipeline.ingest import behavior as behavior_ingest
from pipeline.ingest.behavior import get_behavior_paths

from pipeline.fixes import schema, FixHistory

os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

log = logging.getLogger(__name__)

"""
Up to this moment, a the auto-water status of a trial is determined based on a single criterion:
    SessionData.TrialSettings(i).GUI.AutoWater == 1

However, auto-water is also controlled by the behavior system based on historical performance of the animal for a given trial.
In particular, the auto-water status is determined also based on:
    t.settings.GaveFreeReward[0] == 1  # free water given on the right port
    t.settings.GaveFreeReward[1] == 1  # free water given on the left port

This fix goes through each trial, check and update the auto-water status
"""


@schema
class FixAutoWater(dj.Manual):
    definition = """ # This table accompanies fix_0018
    -> FixHistory
    -> experiment.BehaviorTrial
    ---
    auto_water_needed_fix: bool  # was auto_water for this trial incorrect and needed fix
    """


def fix_autowater_trial(session_keys={}):
    """
    This fix applies to sessions ingested with the BehaviorIngest's make() only,
    as opposed to BehaviorBpodIngest (for Foraging Task)
    """
    sessions_2_update = (experiment.Session & behavior_ingest.BehaviorIngest & session_keys)
    sessions_2_update = sessions_2_update - FixAutoWater

    if not sessions_2_update:
        return

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}

    log.info('--- Fixing {} session(s) ---'.format(len(sessions_2_update)))
    for key in sessions_2_update.fetch('KEY'):
        success, incorrect_autowater_trials = _fix_one_session(key)
        if success:
            needed_fix_trials = [t['trial'] for t, _ in incorrect_autowater_trials]
            FixHistory.insert1(fix_hist_key, skip_duplicates=True)
            FixAutoWater.insert([{**fix_hist_key, **tkey,
                                  'auto_water_needed_fix': 1 if tkey['trial'] in needed_fix_trials else 0}
                                 for tkey in (experiment.BehaviorTrial & key).fetch('KEY')])
            log.info('\tAuto-water fixing for session {} finished'.format(key))
        else:
            log.info('\t!!! Fixing session {} failed! Skipping...'.format(key))


def _fix_one_session(key):
    log.info('Running fix for session: {}'.format(key))
    # get filepath of the behavior file for this session - and read the .mat file
    rel_paths = (behavior_ingest.BehaviorIngest.BehaviorFile & key).fetch('behavior_file')
    if len(rel_paths) > 1:
        log.warning('Found multiple behavior files for this session {} (unable to handle this case yet)'.format(key))
        return False, None
    else:
        rel_path = rel_paths[0]

    session = (experiment.Session & key).fetch1()
    rig_path = [p for r, p, _ in get_behavior_paths() if r == session['rig']]

    if len(rig_path) != 1:
        log.warning('No behavior data path found for rig: {} (please check "behavior_data_paths" in dj_local_conf)'.format(session['rig']))
        return False, None
    else:
        rig_path = rig_path[0]

    paths = list(pathlib.Path(rig_path).rglob(f'*/{rel_path}'))

    if len(paths) != 1:
        raise FileNotFoundError(f'Unable to identify/resolve behavior file - Found {len(paths)}: {paths}')
    else:
        path = paths[0]

    # extract trial data from behavior file to compare "auto_water" value stored in db
    skey, rows = behavior_ingest.BehaviorIngest._load(session, path)

    trial_keys, auto_waters = (experiment.BehaviorTrial & key).fetch('KEY', 'auto_water', order_by='trial')

    assert len(trial_keys) == len(rows['behavior_trial'])

    # extract correct "auto_water" status from behavior file, compare to the value stored in db
    incorrect_autowater_trials = []
    for tkey, auto_water, behavior_trial_data in zip(trial_keys, auto_waters, rows['behavior_trial']):
        assert tkey['trial'] == behavior_trial_data['trial']
        if auto_water != behavior_trial_data['auto_water']:
            incorrect_autowater_trials.append((tkey, behavior_trial_data['auto_water']))

    # transactional update for this session
    log.info('Correcting auto_water for {} trials'.format(len(incorrect_autowater_trials)))
    if len(incorrect_autowater_trials):
        with experiment.BehaviorTrial.connection.transaction:
            for tkey, auto_water in incorrect_autowater_trials:
                (experiment.BehaviorTrial & tkey)._update('auto_water', int(auto_water))

    return True, incorrect_autowater_trials


if __name__ == '__main__':
    fix_autowater_trial()