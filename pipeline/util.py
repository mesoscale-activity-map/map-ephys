import numpy as np

from . import (experiment, psth, ephys)


def _get_trial_event_times(events, units, trial_cond_name):
    """
    Get median event start times from all unit-trials from the specified "trial_cond_name" and "units" - aligned to GO CUE
    For trials with multiple events of the same type, use the one occurred last
    :param events: list of events
    """
    events = list(events) + ['go']

    tr_OI = (psth.TrialCondition().get_trials(trial_cond_name) & units).proj()
    tr_events = {}
    for eve in events:
        if eve not in tr_events:
            tr_events[eve] = (tr_OI.aggr(
                experiment.TrialEvent & {'trial_event_type': eve}, trial_event_id='max(trial_event_id)')
                              * experiment.TrialEvent).fetch('trial_event_time', order_by='trial')

    present_events, event_starts = [], []
    for etype, etime in tr_events.items():
        if etype in events[:-1]:
            present_events.append(etype)
            event_starts.append(np.nanmedian(etime.astype(float) - tr_events["go"].astype(float)))

    return np.array(present_events), np.array(event_starts)


def _get_stim_onset_time(units, trial_cond_name):
    stim_onsets = (experiment.PhotostimEvent.proj('photostim_event_time')
                   * (experiment.TrialEvent & 'trial_event_type="go"').proj(go_time='trial_event_time')
                   & psth.TrialCondition().get_trials(trial_cond_name) & units).proj(
        stim_onset_from_go='photostim_event_time - go_time').fetch('stim_onset_from_go')
    return np.nanmean(stim_onsets.astype(float))


def _get_units_hemisphere(units):
    """
    Return the hemisphere ("left" or "right") that the specified units belong to,
     based on the targeted insertion location - "ephys.ProbeInsertion.InsertionLocation"
    :param units: either a list of unit_keys or a query of the ephys.Unit table
    :return: "left" or "right"
    """
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
    """
    Return the "clustering_method" used to estimate the all the units for the provided "probe_insertion"
    :param probe_insertion: an "ephys.ProbeInsertion" key
    :return: clustering_method
    """
    clustering_methods = (ephys.ClusteringMethod & (ephys.Unit & probe_insertion)).fetch('clustering_method')
    if len(clustering_methods) == 1:
        return clustering_methods[0]
    else:
        raise ValueError(f'Found multiple clustering methods: {clustering_methods}')
