import numpy as np
from . import (experiment, psth, ephys, psth_foraging)


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


def _get_ephys_trial_event_times(all_align_types, align_to, trial_keys):
    """
    Similar to _get_trial_event_times, except:
        1. for the foraging task only (using psth_foraging.TrialCondition)
        2. use ephys.TrialEvent (NI time) instead of experiment.TrialEvent (bpod time)
        3. directly use trial_keys instead of units + trial_cond_name to get trials
        4. align to arbitrary event, not just GO CUE
        5. allow trial_offset (like in psth_foraging), e.g., trial start of the *NEXT* trial
        
    :param all_align_types: list of psth_foraging.AlignType()
    :param align_to: psth_foraging.AlignType(), event to align
    :param trial_keys: 
    """
    
    
    tr_events = {}
    for eve in all_align_types:
        q_align = psth_foraging.AlignType & {'align_type_name': eve}
        trial_offset, time_offset = q_align.fetch1('trial_offset', 'time_offset')
        
        offset_trial_keys = (experiment.BehaviorTrial & trial_keys.proj(_='trial', trial=f'trial + {trial_offset}'))
        
        times = (offset_trial_keys.aggr(
            ephys.TrialEvent & q_align, trial_event_id='max(trial_event_id)')
                 * ephys.TrialEvent).fetch('trial_event_time', order_by='trial').astype(float)
        
        tr_events[eve] = times + float(time_offset)
  
    event_starts = []
    for etime in tr_events.values():
        event_starts.append(np.nanmedian(etime - tr_events[align_to]))
            
    return np.array(event_starts)


def _get_stim_onset_time(units, trial_cond_name):
    psth_schema = psth_foraging if 'foraging' in trial_cond_name else psth

    stim_onsets = (experiment.PhotostimEvent.proj('photostim_event_time')
                   * (experiment.TrialEvent & 'trial_event_type="go"').proj(go_time='trial_event_time')
                   & psth_schema.TrialCondition().get_trials(trial_cond_name) & units).proj(
        stim_onset_from_go='photostim_event_time - go_time').fetch('stim_onset_from_go')
    return np.nanmean(stim_onsets.astype(float))


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
