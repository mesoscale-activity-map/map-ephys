#! /usr/bin/env python

import logging

import datajoint as dj
import re
import numpy as np

from pipeline import lab, experiment
from pipeline.ingest import behavior as behav_ingest


log = logging.getLogger(__name__)


def update_delay_event_duration():
    delay_events = experiment.TrialEvent & 'trial_event_type = "delay"'
    next_events = experiment.TrialEvent & delay_events.proj(trial_event_id='trial_event_id + 1')

    ekeys, delay_onsets, delay_durs = delay_events.fetch('KEY', 'trial_event_time', 'duration')
    next_ekeys, next_event_onsets = next_events.fetch('KEY', 'trial_event_time')

    new_durs = next_event_onsets - delay_onsets
    incorrect_dur_events = new_durs != delay_durs

    log.info('Updating duration for {} events'.format(len(incorrect_dur_events)))
    for ekey, dur in zip(np.array(ekeys)[incorrect_dur_events], new_durs[incorrect_dur_events]):
        (experiment.TrialEvent & ekey)._update('duration', dur)
