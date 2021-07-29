import numpy as np
import datajoint as dj
from . import (lab, experiment, psth, ephys, psth_foraging, foraging_model, foraging_analysis)


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
    min_len = np.inf
    for eve in all_align_types:
        q_align = psth_foraging.AlignType & {'align_type_name': eve}
        trial_offset, time_offset = q_align.fetch1('trial_offset', 'time_offset')
        
        offset_trial_keys = (experiment.BehaviorTrial & trial_keys.proj(_='trial', trial=f'trial + {trial_offset}'))
        
        times = (offset_trial_keys.aggr(
            ephys.TrialEvent & q_align, trial_event_id='max(trial_event_id)')
                 * ephys.TrialEvent).fetch('trial_event_time', order_by='trial').astype(float)
        
        tr_events[eve] = times + float(time_offset)
        min_len = min(min_len, len(times))  # To account for possible different lengths due to trial_offset ~= 0
  
    event_starts = []
    for etime in tr_events.values():
        event_starts.append(np.nanmedian(etime[:min_len] - tr_events[align_to][:min_len]))
            
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


def _get_unit_independent_variable(unit_key, model_id, var_name=None):
    """
    Get independent variable over trial for a specified unit (ignored trials are skipped)
    @param unit_key:
    @param model_id:
    @param var_name
    @return: DataFrame (trial, variables)
    """

    hemi = _get_units_hemisphere(unit_key)
    contra, ipsi = ['right', 'left'] if hemi == 'left' else ['left', 'right']

    # Get latent variables from model fitting
    q_latent_variable = (foraging_model.FittedSessionModel.TrialLatentVariable
                         & unit_key
                         & {'model_id': model_id})

    # Flatten latent variables to generate columns like 'left_action_value', 'right_choice_prob'
    latent_variables = q_latent_variable.heading.secondary_attributes
    q_latent_variable_all = dj.U('trial') & q_latent_variable
    for lv in latent_variables:
        for prefix, side in zip(['left_', 'right_', 'contra_', 'ipsi_'],
                                ['left', 'right', contra, ipsi]):
            # Better way here?
            q_latent_variable_all *= eval(f"(q_latent_variable & {{'water_port': '{side}'}}).proj({prefix}{lv}='{lv}', {prefix}='water_port')")

    # Add relative and total value
    q_latent_variable_all = q_latent_variable_all.proj(...,
                                                       relative_action_value_lr='right_action_value - left_action_value',
                                                       relative_action_value_ic='contra_action_value - ipsi_action_value',
                                                       total_action_value='contra_action_value + ipsi_action_value')

    # Add choice
    q_independent_variable = (q_latent_variable_all * experiment.WaterPortChoice).proj(...,
                                                                                       choice='water_port',
                                                                                       choice_lr='water_port="right"',
                                                                                       choice_ic=f'water_port="{contra}"')

    # Add reward
    q_independent_variable = (q_independent_variable * experiment.BehaviorTrial.proj('outcome')).proj(...,
                                                                                                       reward='outcome="hit"'
                                                                                                       )

    df = q_independent_variable.fetch(format='frame', order_by='trial').reset_index()
    
    # Compute RPE
    df['rpe'] = np.nan
    df['rpe'][0] = df.reward[0]
    for side in ['left', 'right']:
        _idx = df[(df.choice == side) & (df.trial > 1)].index
        df.rpe.iloc[_idx] = df.reward.iloc[_idx] - df[f'{side}_action_value'].iloc[_idx - 1].values

    return df if var_name is None else df[['trial', var_name]]


def _get_sess_info(sess_key):
    s = (experiment.Session * foraging_analysis.SessionStats * lab.WaterRestriction & sess_key).fetch1()

    return f"{s['water_restriction_number']}, Session {s['session']}, {s['session_date']}\n" \
           f"{s['session_total_trial_num']} trials, ignored {s['session_ignore_num']/s['session_total_trial_num']*100:.2g}%\n" \
           f"foraging eff. {s['session_foraging_eff_optimal']*100:.2g}% (adj. {s['session_foraging_eff_optimal_random_seed']*100:.2g}%)"
