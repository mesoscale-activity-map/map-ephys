import os
import logging
import pathlib
from datetime import datetime
from decimal import Decimal
import datajoint as dj
import re

from pipeline import lab, experiment, report
from pipeline.ingest import behavior as behavior_ingest
from pipeline.fixes import schema, FixHistory


os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

log = logging.getLogger(__name__)


"""
Up to this moment, a trial is tagged as a photostim trial based on a single criterion:
    SessionData.StimTrial > 0 
    
However, in fact, whether a trial undergo photostim or not also depends on 3 other criteria,
in all, the following logic would accurately capture the photostim status of a trial:

```
TrialshadRealPhotoStim = (SessionData.StimTrials(i) > 0 && SessionData.TrialSettings(i).GUI.DelayPeriod == 1.2 && ...
    SessionData.TrialSettings(i).GUI.ProtocolType ==5 && SessionData.TrialSettings(i).GUI.Autolearn == 4) 
```

In other word:
+ SessionData.StimTrial > 0 
+ The duration of the delay period is 1.2 sec
+ Trial's protocol == 5 (for Rig2) or > 4 (for Rig3)
+ Autolearn == 4

This fix goes through each trial currently labeled as photostim trial,
 re-assess and remove from the PhotostimTrial table if those criteria are not met
"""


@schema
class FixPhotostimTrial(dj.Manual):
    definition = """ # This table accompanies fix_0017
    -> FixHistory
    -> experiment.SessionTrial
    ---
    is_removed: bool  # was this trial incorrectly labeled as photostim and needed to be removed from PhotostimTrial
    """


def fix_photostim_trial(session_keys={}):
    """
    This fix applies to sessions ingested with the BehaviorIngest's make() only,
    as opposed to BehaviorBpodIngest (for Foraging Task)
    """
    sessions_2_update = (experiment.Session & behavior_ingest.BehaviorIngest
                         & experiment.PhotostimTrial & session_keys)
    sessions_2_update = sessions_2_update - FixPhotostimTrial

    if not sessions_2_update:
        return

    fix_hist_key = {'fix_name': pathlib.Path(__file__).name,
                    'fix_timestamp': datetime.now()}

    log.info('Fixing {} session(s)'.format(len(sessions_2_update)))
    for key in sessions_2_update.fetch('KEY'):
        success, invalid_photostim_trials = _fix_one_session(key)
        if success:
            removed_photostim_trials = [t['trial'] for t in invalid_photostim_trials]
            FixHistory.insert1(fix_hist_key, skip_duplicates=True)
            FixPhotostimTrial.insert([{**fix_hist_key, **tkey,
                                       'is_removed': 1 if tkey['trial'] in removed_photostim_trials else 0}
                                      for tkey in (experiment.SessionTrial & key).fetch('KEY')])


def _fix_one_session(key):
    log.info('Running fix for session: {}'.format(key))
    # determine if this session's photostim is: `early-delay` or `late-delay`
    path = (behavior_ingest.BehaviorIngest.BehaviorFile & key).fetch('behavior_file')[0]
    h2o = (lab.WaterRestriction & key).fetch1('water_restriction_number')

    photostim_period = 'early-delay'
    rig_name = re.search('Recording(Rig\d)_', path)
    if re.match('SC', h2o) and rig_name:
        rig_name = rig_name.groups()[0]
        if rig_name == "Rig3":
            photostim_period = 'late-delay'

    invalid_photostim_trials = []
    for trial_key in (experiment.PhotostimTrial & key).fetch('KEY'):
        protocol_type = int((experiment.TrialNote & trial_key
                             & 'trial_note_type = "protocol #"').fetch1('trial_note'))
        autolearn = int((experiment.TrialNote & trial_key
                         & 'trial_note_type = "autolearn"').fetch1('trial_note'))

        if photostim_period == 'early-delay':
            valid_protocol = protocol_type == 5
        elif photostim_period == 'late-delay':
            valid_protocol = protocol_type > 4

        delay_duration = (experiment.TrialEvent & trial_key & 'trial_event_type = "delay"').fetch(
            'duration', order_by='trial_event_time DESC', limit=1)[0]

        if not (valid_protocol and autolearn == 4 and delay_duration == Decimal('1.2')):
            # all criteria not met, this trial should not have been a photostim trial
            invalid_photostim_trials.append(trial_key)

    log.info('Deleting {} incorrectly labeled PhotostimTrial'.format(len(invalid_photostim_trials)))
    if len(invalid_photostim_trials):
        with dj.config(safemode=False):
            # delete invalid photostim trials
            (experiment.PhotostimTrial & invalid_photostim_trials).delete()
            # delete ProbeLevelPhotostimEffectReport figures associated with this session
            (report.ProbeLevelPhotostimEffectReport & key).delete()

    return True, invalid_photostim_trials


if __name__ == '__main__':
    fix_photostim_trial()