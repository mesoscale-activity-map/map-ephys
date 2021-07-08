# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 23:36:36 2021

@author: houha
"""

#%%  
import datajoint as dj
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')

import numpy as np
import pandas as pd
import re
from tqdm import tqdm

from pipeline import experiment, lab
from pipeline.ingest import behavior as behavior_ingest
from pipeline.ingest.util import load_and_parse_a_csv_file


def compare_pc_and_bpod_times(q_sess=dj.AndList(['water_restriction_number = "HH09"', 'session < 10'])):
    '''
    Compare PC-TIME and BPOD-TIME of pybpod csv file
    This is critical for ephys timing alignment
    [Conclusions]:
    1. PC-TIME has an average delay of 4 ms and sometimes much much longer!
    2. We should always use BPOD-TIME (at least for all offline analysis)
    '''
    #%%
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
    
