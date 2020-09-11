"""
Foraging behavior loader utilities developed by Marton Rozsa
https://github.com/rozmar/DataPipeline/blob/master/Behavior/behavior_rozmar.py
"""

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import pickle
import shutil


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
                df['var:' + varname][0] = d['variables'][varname]   # Only save to the first row
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
