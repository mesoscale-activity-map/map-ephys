import numpy as np

from . import (experiment, psth, ephys)


def _get_trial_event_times(events, units, trial_cond_name):
    """
    Get median event start times from all unit-trials from the specified "trial_cond_name" and "units" - aligned to GO CUE
    :param events: list of events
    """
    events = list(events) + ['go']

    event_types, event_times = (psth.TrialCondition().get_trials(trial_cond_name)
                                * (experiment.TrialEvent & [{'trial_event_type': eve} for eve in events])
                                & units).fetch('trial_event_type', 'trial_event_time')
    period_starts = [(event_type, np.nanmedian((event_times[event_types == event_type]
                                                - event_times[event_types == 'go']).astype(float)))
                     for event_type in events[:-1] if len(event_times[event_types == event_type])]
    present_events, event_starts = list(zip(*period_starts))
    return np.array(present_events), np.array(event_starts)


def _get_units_hemisphere(units):
    ml_locations = np.unique((ephys.ProbeInsertion.InsertionLocation & units).fetch('ml_location'))
    if len(ml_locations) == 0:
        raise Exception('No ProbeInsertion.InsertionLocation available')
    if (ml_locations > 0).any() and (ml_locations < 0).any():
        raise ValueError('The specified units belongs to both hemispheres...')
    if (ml_locations > 0).all():
        return 'right'
    elif (ml_locations < 0).all():
        return 'left'
    else:
        assert (ml_locations == 0).all()  # sanity check
        raise ValueError('Ambiguous hemisphere: ML locations are all 0...')


def _get_clustering_method(probe_insertion):
    clustering_methods = (ephys.ClusteringMethod & (ephys.Unit & probe_insertion)).fetch('clustering_method')
    if len(clustering_methods) == 1:
        return clustering_methods[0]
    else:
        raise ValueError(f'Found multiple clustering methods: {clustering_methods}')
