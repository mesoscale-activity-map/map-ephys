#! /usr/bin/env python

import logging

import datajoint as dj
import re

from pipeline import lab, experiment
from pipeline.ingest import behavior as behav_ingest


log = logging.getLogger(__name__)


def update_photostim_event_time():
    """
    Updating photostim event time for Susu's sessions where behavior is from Rig3 (Rig3 as part of the behavior file name)
    For these sessions, photostimulation is late-delay, i.e. photostim onset is 0.5 second prior to the response-period (go-cue)
    """
    for session_key in (experiment.Session & 'username = "susu"').fetch('KEY'):
        behav_fname = (behav_ingest.BehaviorIngest.BehaviorFile & session_key).fetch1('behavior_file')
        rig_name = re.search('Recording(Rig\d)_', behav_fname)
        if rig_name is None:
            log.warning('No rig-info in behavior file ({}) for session: {}. Skipping...'.format(behav_fname, session_key))
            continue

        rig_name = rig_name.groups()[0]
        log.info('Found rig-name: {} from behavior file ({})'.format(rig_name, behav_fname))

        if rig_name == "Rig3":
            log.info('Matching "RecordingRig3", proceed with updating photostim onset')
            with dj.conn().transaction:
                for trial_key in (experiment.PhotostimTrial & session_key).fetch('KEY', order_by='trial'):
                    # get go-cue, compute photostim onset
                    go_cue_time = (experiment.TrialEvent & trial_key & 'trial_event_type = "go"').fetch1('trial_event_time')
                    photostim_onset = float(go_cue_time) - 0.5
                    # update
                    (experiment.PhotostimEvent & trial_key)._update('photostim_event_time', photostim_onset)


if __name__ == '__main__':
    update_photostim_event_time()
