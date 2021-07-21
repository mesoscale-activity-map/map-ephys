"""
Foraging behavior loader utilities developed by Marton Rozsa
https://github.com/rozmar/DataPipeline/blob/master/Behavior/behavior_rozmar.py
"""

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

import datajoint as dj
plt.style.use('seaborn-talk')
from pipeline import lab, experiment, ephys
from pipeline.ingest import behavior as behavior_ingest


def loaddirstucture(projectdir, projectnames_needed=None, experimentnames_needed=None,
                    setupnames_needed=None):

    projectdir = Path(projectdir)
    if not projectdir.exists():
        raise RuntimeError('Project directory not found: {}'.format(projectdir))

    dirstructure = dict()
    projectnames = list()
    experimentnames = list()
    setupnames = list()
    sessionnames = list()
    subjectnames = list()

    for projectname in projectdir.iterdir():
        if projectname.is_dir() and (not projectnames_needed or projectname.name in projectnames_needed):
            dirstructure[projectname.name] = dict()
            projectnames.append(projectname.name)

            for subjectname in (projectname / 'subjects').iterdir():
                if subjectname.is_dir():
                    subjectnames.append(subjectname.name)

            for experimentname in (projectname / 'experiments').iterdir():
                if experimentname.is_dir() and (
                        not experimentnames_needed or experimentname.name in experimentnames_needed):
                    dirstructure[projectname.name][experimentname.name] = dict()
                    experimentnames.append(experimentname.name)

                    for setupname in (experimentname / 'setups').iterdir():
                        if setupname.is_dir() and (not setupnames_needed or setupname.name in setupnames_needed):
                            setupnames.append(setupname.name)
                            dirstructure[projectname.name][experimentname.name][setupname.name] = list()

                            for sessionname in (setupname / 'sessions').iterdir():
                                if sessionname.is_dir():
                                    sessionnames.append(sessionname.name)
                                    dirstructure[projectname.name][experimentname.name][setupname.name].append(
                                        sessionname.name)
    return dirstructure, projectnames, experimentnames, setupnames, sessionnames, subjectnames


def load_and_parse_a_csv_file(csvfilename):
    df = pd.read_csv(csvfilename, delimiter = ';', skiprows = 6)
    df = df[df['TYPE'] != '|']  # delete empty rows
    df = df[df['TYPE'] != 'During handling of the above exception, another exception occurred:']  # delete empty rows
    df = df[df['MSG'] != ' ']  # delete empty rows
    df = df[df['MSG'] != '|']  # delete empty rows
    df = df.reset_index(drop = True)  # resetting indexes after deletion
    try:
        df['PC-TIME'] = df['PC-TIME'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))  # converting string time to datetime
    except ValueError:  # sometimes pybpod don't write out the whole number...
        badidx = df['PC-TIME'].str.find('.') == -1
        if len(df['PC-TIME'][badidx]) == 1:
            df['PC-TIME'][badidx] = df['PC-TIME'][badidx] + '.000000'
        else:
            df['PC-TIME'][badidx] = [df['PC-TIME'][badidx] + '.000000']
        df['PC-TIME'] = df['PC-TIME'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))  # converting string time to datetime
        
    tempstr = df['+INFO'][df['MSG'] == 'CREATOR-NAME'].values[0]
    experimenter = tempstr[2:tempstr[2:].find('"') + 2]  # +2
    tempstr = df['+INFO'][df['MSG'] == 'SUBJECT-NAME'].values[0]
    subject = tempstr[2:tempstr[2:].find("'") + 2]  # +2
    df['experimenter'] = experimenter
    df['subject'] = subject
    
    # adding trial numbers in session
    idx = (df[df['TYPE'] == 'TRIAL']).index.to_numpy()
    idx = np.concatenate(([0], idx, [len(df)]), 0)
    idxdiff = np.diff(idx)
    Trialnum = np.array([])
    for i, idxnumnow in enumerate(idxdiff):  # zip(np.arange(0:len(idxdiff)),idxdiff):#
        Trialnum = np.concatenate((Trialnum, np.zeros(idxnumnow) + i), 0)
    df['Trial_number_in_session'] = Trialnum
    # =============================================================================
    #     # adding trial types
    #     tic = time.time()
    #     indexes = df[df['MSG'] == 'Trialtype:'].index + 1 #+2
    #     if len(indexes)>0:
    #         if 'Trialtype' not in df.columns:
    #             df['Trialtype']=np.NaN
    #         trialtypes = df['MSG'][indexes]
    #         trialnumbers = df['Trial_number_in_session'][indexes].values
    #         for trialtype,trialnum in zip(trialtypes,trialnumbers):
    #             #df['Trialtype'][df['Trial_number_in_session'] == trialnum] = trialtype
    #             df.loc[df['Trial_number_in_session'] == trialnum, 'Trialtype'] = trialtype
    #     toc = time.time()
    #     print(['trial types:',toc-tic])
    # =============================================================================
    # adding block numbers
    indexes = df[df['MSG'] == 'Blocknumber:'].index + 1  # +2
    if len(indexes) > 0:
        if 'Block_number' not in df.columns:
            df['Block_number'] = np.NaN
        blocknumbers = df['MSG'][indexes]
        trialnumbers = df['Trial_number_in_session'][indexes].values
        for blocknumber, trialnum in zip(blocknumbers, trialnumbers):
            # df['Block_number'][df['Trial_number_in_session'] == trialnum] = int(blocknumber)
            try:
                df.loc[df['Trial_number_in_session'] == trialnum, 'Block_number'] = int(blocknumber)
            except:
                df.loc[df['Trial_number_in_session'] == trialnum, 'Block_number'] = np.nan

    # adding accumulated rewards -L,R,M
    for direction in ['L', 'R', 'M']:
        indexes = df[df['MSG'] == 'reward_{}_accumulated:'.format(direction)].index + 1  # +2
        if len(indexes) > 0:
            if 'reward_{}_accumulated'.format(direction) not in df.columns:
                df['reward_{}_accumulated'.format(direction)] = np.NaN
            accumulated_rewards = df['MSG'][indexes]
            trialnumbers = df['Trial_number_in_session'][indexes].values
            for accumulated_reward, trialnum in zip(accumulated_rewards, trialnumbers):
                # df['Block_number'][df['Trial_number_in_session'] == trialnum] = int(blocknumber)
                try:
                    df.loc[df['Trial_number_in_session'] == trialnum, 'reward_{}_accumulated'.format(
                        direction)] = accumulated_reward == 'True'
                except:
                    df.loc[
                        df['Trial_number_in_session'] == trialnum, 'reward_{}_accumulated'.format(direction)] = np.nan
                   
    # adding trial numbers -  the variable names are crappy.. sorry
    indexes = df[df['MSG'] == 'Trialnumber:'].index + 1  # +2
    if len(indexes) > 0:
        if 'Trial_number' not in df.columns:
            df['Trial_number'] = np.NaN
        blocknumbers = df['MSG'][indexes]
        trialnumbers = df['Trial_number_in_session'][indexes].values
        for blocknumber, trialnum in zip(blocknumbers, trialnumbers):
            # df['Trial_number'][df['Trial_number_in_session'] == trialnum] = int(blocknumber)
            try:
                df.loc[df['Trial_number_in_session'] == trialnum, 'Trial_number'] = int(blocknumber)
            except:
                df.loc[df['Trial_number_in_session'] == trialnum, 'Block_number'] = np.nan

    # saving variables (if any)
    variableidx = (df[df['MSG'] == 'Variables:']).index.to_numpy()
    if len(variableidx) > 0:
        d = {}
        exec('variables = ' + df['MSG'][variableidx + 1].values[0], d)
        for varname in d['variables'].keys():
            if ('reward_probabilities' in varname or '_ch_in' in varname or '_ch_out' in varname
                or varname in ['retract_motor_signal', 'protract_motor_signal']):  
                # For the variables that never change within one **bpod** session, 
                # only save to the first row of the dataframe to save time and space
                df['var:' + varname] = None   # Initialize with None
                df.at[0, 'var:' + varname] = d['variables'][varname]   # Only save to the first row
            else:        
                if isinstance(d['variables'][varname], (list, tuple)):
                    templist = list()
                    for idx in range(0, len(df)):
                        templist.append(d['variables'][varname])
                    df['var:' + varname] = templist
                else:
                    df['var:' + varname] = d['variables'][varname]
                    
    # updating variables
    variableidxs = (df[df['MSG'] == 'Variables updated:']).index.to_numpy()
    for variableidx in variableidxs:
        d = {}
        exec('variables = ' + df['MSG'][variableidx + 1], d)
        for varname in d['variables'].keys():
            # Skip the variables that never change within one **bpod** session
            if ('reward_probabilities' in varname 
                or '_ch_in' in varname or '_ch_out' in varname
                or varname in ['retract_motor_signal', 'protract_motor_signal']):     
                continue
            
            if isinstance(d['variables'][varname], (list, tuple)):
                templist = list()
                idxs = list()
                for idx in range(variableidx, len(df)):
                    idxs.append(idx)
                    templist.append(d['variables'][varname])
                df['var:' + varname][variableidx:] = templist.copy()
            # =============================================================================
            #                 print(len(templist))
            #                 print(len(idxs))
            #                 print(templist)
            #                 df.loc[idxs, 'var:'+varname] = templist
            # =============================================================================
            else:
                # df['var:'+varname][variableidx:] = d['variables'][varname]
                df.loc[range(variableidx, len(df)), 'var:' + varname] = d['variables'][varname]

    # saving motor variables (if any)
    variableidx = (df[df['MSG'] == 'LickportMotors:']).index.to_numpy()
    if len(variableidx) > 0:
        d = {}
        exec('variables = ' + df['MSG'][variableidx + 1].values[0], d)
        for varname in d['variables'].keys():
            df['var_motor:' + varname] = d['variables'][varname]
            
    # extracting reward probabilities from variables (already in behavior.py; no need to broadcast to every time stamp in df)
    # if ('var:reward_probabilities_L' in df.columns) and ('Block_number' in df.columns):
    #     probs_l = df['var:reward_probabilities_L'][0]
    #     probs_r = df['var:reward_probabilities_R'][0]
    #     df['reward_p_L'] = np.nan
    #     df['reward_p_R'] = np.nan
    #     if ('var:reward_probabilities_M' in df.columns) and ('Block_number' in df.columns):
    #         probs_m = df['var:reward_probabilities_M'][0]
    #         df['reward_p_M'] = np.nan
    #     for blocknum in df['Block_number'].unique():
    #         if not np.isnan(blocknum):
    #             df.loc[df['Block_number'] == blocknum, 'reward_p_L'] = probs_l[int(blocknum - 1)]
    #             df.loc[df['Block_number'] == blocknum, 'reward_p_R'] = probs_r[int(blocknum - 1)]
    #             if ('var:reward_probabilities_M' in df.columns) and ('Block_number' in df.columns):
    #                 df.loc[df['Block_number'] == blocknum, 'reward_p_M'] = probs_m[int(blocknum - 1)]
                    
    return df


def compare_pc_and_bpod_times(q_sess=dj.AndList(['water_restriction_number = "HH09"', 'session < 10'])):
    '''
    Compare PC-TIME and BPOD-TIME of pybpod csv file
    This is a critical validation for ephys timing alignment
    This function requires raw .csv data to be present under `behavior_bpod`.`project_paths`    
    
    [Conclusions]:
    1. PC-TIME has an average delay of 4 ms and sometimes much much longer!
    2. We should always use BPOD-TIME (at least for all offline analysis)

    Parameters
    ----------
    q_sess : TYPE, optional
        DESCRIPTION. Query of session. The default is dj.AndList(['water_restriction_number = "HH09"', 'session < 10']).

    Returns
    -------
    None.

    '''
    csv_folders = (behavior_ingest.BehaviorBpodIngest.BehaviorFile * lab.WaterRestriction & q_sess
                 ).fetch('behavior_file')
    
    current_project_path = dj.config['custom']['behavior_bpod']['project_paths'][0]
    current_ingestion_folder = re.findall('(.*)Behavior_rigs.*', current_project_path)[0]
    
    event_to_compare = ['GoCue', 'Choice_L', 'Choice_R', 'ITI', 'End']
    event_pc_times = {event: list() for event in event_to_compare}
    event_bpod_times = {event: list() for event in event_to_compare}
    
    # --- Loop over all files
    for csv_folder in tqdm(csv_folders):
        csv_file = (current_ingestion_folder + re.findall('.*(Behavior_rigs.*)', csv_folder)[0]
                    + '/'+ csv_folder.split('/')[-1] + '.csv')
        df_behavior_session = load_and_parse_a_csv_file(csv_file)
        
        # ---- Integrity check of the current bpodsess file ---
        # It must have at least one 'trial start' and 'trial end'
        trial_start_idxs = df_behavior_session[(df_behavior_session['TYPE'] == 'TRIAL') & (
                    df_behavior_session['MSG'] == 'New trial')].index
        if not len(trial_start_idxs):
            continue   # Make sure 'start' exists, otherwise move on to try the next bpodsess file if exists     
            
        # For each trial
        trial_end_idxs = trial_start_idxs[1:].append(pd.Index([(max(df_behavior_session.index))]))        
        
        for trial_start_idx, trial_end_idx in zip(trial_start_idxs, trial_end_idxs):
            df_behavior_trial = df_behavior_session[trial_start_idx:trial_end_idx + 1]
            
            pc_time_trial_start = df_behavior_trial.iloc[0]['PC-TIME']
            
            for event in event_to_compare:
                idx = df_behavior_trial[(df_behavior_trial['TYPE'] == 'TRANSITION') & (
                                df_behavior_trial['MSG'] == event)].index
                if not len(idx):
                    continue
                
                # PC-TIME
                pc_time = (df_behavior_trial.loc[idx]['PC-TIME']- pc_time_trial_start).values / np.timedelta64(1, 's')
                event_pc_times[event].append(pc_time[0])
                
                # BPOD-TIME
                bpod_time = df_behavior_trial.loc[idx]['BPOD-INITIAL-TIME'].values
                event_bpod_times[event].append(bpod_time[0])
        
    # --- Plotting ---
    fig = plt.figure()
    ax = fig.subplots(1,2)
    for event in event_pc_times:
        ax[0].plot(event_bpod_times[event], event_pc_times[event], '*', label=event)
        ax[1].hist((np.array(event_pc_times[event]) - np.array(event_bpod_times[event])) * 1000, range=(0,20), bins=500, label=event)
    
    ax_max = max(max(ax[0].get_xlim()), max(ax[0].get_ylim()))
    ax[0].plot([0, ax_max], [0, ax_max], 'k:')    
    ax[0].set_xlabel('Bpod time from trial start (s)')
    ax[0].set_ylabel('PC time (s)')
    ax[1].set_xlabel('PC time lag (ms)')
    ax[1].set_title(f'{q_sess}\ntotal {len(csv_folders)} bpod csv files')
    ax[0].legend()
    

def compare_ni_and_bpod_times(q_sess=dj.AndList(['subject_id = "473361"', 'session >= 57']), event_to_align='bitcodestart'):
    '''
    Compare NI-TIME and BPOD-TIME
    This is a critical validation for ephys timing alignment
    
    [Conclusions]:
    1. No bias between NI time and bpod time, but there's jitter up to 10 ms!
    2. We should trust NI time (?)

    Parameters
    ----------
    q_sess : TYPE, optional
        DESCRIPTION. Query of session. The default is dj.AndList(['water_restriction_number = "HH09"', 'session >= 50']).

    Returns
    -------
    None.

    '''
    
    event_all = ['bitcodestart', 'go', 'choice', 'trialend']
    event_to_compare = [e for e in event_all if e != event_to_align]
    
    # 1. -- all events, Bpod vs NIXD --
    # Ephys time
    ephys_times = (ephys.TrialEvent & q_sess & f'trial_event_type IN {tuple(event_all)}').fetch(format='frame')
    ephys_times = ephys_times.reset_index().pivot(index = ['subject_id', 'session', 'trial'], columns='trial_event_type').trial_event_time
    session_times = ephys_times.bitcodestart
    ephys_times = ephys_times.sub(ephys_times[event_to_align], axis=0).drop(columns=event_to_align)  # To each bitcode start (bpodstart is sometimes problematic if a new bpod session is started)
    
    # Bpod time
    bpod_times = (experiment.TrialEvent & q_sess & f'trial_event_type IN {tuple(event_all)}' ).fetch(format='frame')
    bpod_times = bpod_times.reset_index().pivot(index = ['subject_id', 'session', 'trial'], columns='trial_event_type').trial_event_time.astype(float)  # Already related to bpod start
    bpod_times = bpod_times.sub(bpod_times[event_to_align], axis=0).drop(columns=event_to_align)  # To each trial's bpod start
    
    # Plot: Bpod vs NIXD, distribution of differences
    fig = plt.figure(figsize=(8,13))
    ax = fig.subplots(len(event_to_compare), 2)
    max_error = 0
    for n, event in enumerate(event_to_compare):
        ax[n, 0].plot(ephys_times[event], bpod_times[event], '*', label=event)
        ax[n, 1].hist((np.array(bpod_times[event].astype(float)) - np.array(ephys_times[event])) * 1000, bins=100, label=event)
        ax_max = max(max(ax[n, 0].get_xlim()), max(ax[n, 0].get_ylim()))
        ax_min = min(min(ax[n, 0].get_xlim()), min(ax[n, 0].get_ylim()))
        ax[n, 0].plot([ax_min, ax_max], [ax_min, ax_max], 'k:')
        ax[n, 0].legend()
        max_error = max(max_error, max(ax[n, 1].get_xlim()))

    for _a in ax[:, 1]:
        _a.set_xlim(-max_error, max_error)
        
    ax[-1, 0].set_xlabel(f'NI time to {event_to_align} (s)')
    ax[-1, 0].set_ylabel('Bpod time (s)')
    ax[-1, 1].set_xlabel('Bpod time - NI time (ms)')
    ax[0, 1].set_title(f'{q_sess}\ntotal {len(bpod_times)} trials', fontsize=8)
    
    
    # 2. -- first lick: NIXA (raw lickboard, close to ground truth?), NIXD (bpod digital to NI), and Bpod (bpod csv file) --
    ephys_go_cue = (ephys.TrialEvent & 'trial_event_type = "go"' & q_sess).proj(ephys_go='trial_event_time')
    nixa_first_lick = ephys_go_cue.proj().aggr(
        (ephys.ActionEvent * ephys_go_cue) & 'action_event_time >= ephys_go', 
        nixa='min(action_event_time)')    # Session-time of first lick of each trial

    q_all = (nixa_first_lick.proj(..., tmp='trial_event_id')   # NIXA
            * (ephys.TrialEvent & 'trial_event_type = "choice"').proj(nixd='trial_event_time', tmp1='trial_event_id')   # NIXD
            * (ephys.TrialEvent & f'trial_event_type = "{event_to_align}"').proj(ni_align='trial_event_time')
            * (experiment.TrialEvent & 'trial_event_type = "choice"').proj(bpod='trial_event_time', tmp2='trial_event_id')  # Bpod csv
            * (experiment.TrialEvent & f'trial_event_type = "{event_to_align}"').proj(bpod_align='trial_event_time', tmp3='trial_event_id')
            & q_sess).proj(...,
                           nixa_aligned='nixa - ni_align',
                           nixd_aligned='nixd - ni_align',
                           bpod_aligned='bpod - bpod_align')

    diff_cut_off = 30   # (ms) Cutoff where discrepancy is due to lick faster than 10 ms go cue bitcode (has been decrease to 1ms)
    nixa, nixd, bpod = (q_all & f'abs(nixa_aligned - nixd_aligned) < {diff_cut_off/1000}').fetch('nixa_aligned', 'nixd_aligned', 'bpod_aligned')
    n_cut_off = len(q_all) - len(nixa)

    nixa *= 1000
    nixd *= 1000
    bpod = bpod.astype(float) * 1000

    # (Bpod - NIXA) vs (NIXD - NIXA)
    fig = plt.figure(figsize=(8,8))
    ax = fig.subplots(2, 2)
    ax[0,0].plot(bpod - nixa, nixd - nixa, '.')
    ax[0,0].set(xlabel='bpod - nixa (ms)', ylabel='nixd - nixa')
    ax[0,1].hist(bpod - nixa, bins = 100)
    ax[0,1].set(xlabel='bpod - nixa (ms)', title=f'first lick after go, total_trial = {len(bpod)}, cut_off = {n_cut_off}')
    ax[1,0].hist(nixd - nixa, bins = 100)
    ax[1,0].set(xlabel='nixd - nixa (ms)')
    ax[1,1].hist(bpod - nixd, bins = 100)
    ax[1,1].set(xlabel='bpod - nixd (ms)')
    
    # 3. -- Error as a function of session time --
    # For the up-to-10-ms unacceptable jitter between times,
    # I have narrowed it down to the precision of computing float numbers!
    # Here I just compare choice times from ephys_times and q_all, which are 
    # using exactly the same data (ephys.TrialEvent), but by different approaches:
    #    ephys_times: fetch and substract
    #    q_all: substract and fetch
    
    session_times = session_times[~np.isnan(ephys_times.choice)]
    fetch_and_substract = ephys_times.choice[~np.isnan(ephys_times.choice)]
    substract_and_fetch = q_all.fetch('nixd_aligned')
    plt.figure()
    plt.plot(session_times, fetch_and_substract - substract_and_fetch, '.')
    

    
    import pdb; pdb.set_trace()
