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
    # extracting reward probabilities from variables
    if ('var:reward_probabilities_L' in df.columns) and ('Block_number' in df.columns):
        probs_l = df['var:reward_probabilities_L'][0]
        probs_r = df['var:reward_probabilities_R'][0]
        df['reward_p_L'] = np.nan
        df['reward_p_R'] = np.nan
        if ('var:reward_probabilities_M' in df.columns) and ('Block_number' in df.columns):
            probs_m = df['var:reward_probabilities_M'][0]
            df['reward_p_M'] = np.nan
        for blocknum in df['Block_number'].unique():
            if not np.isnan(blocknum):
                df.loc[df['Block_number'] == blocknum, 'reward_p_L'] = probs_l[int(blocknum - 1)]
                df.loc[df['Block_number'] == blocknum, 'reward_p_R'] = probs_r[int(blocknum - 1)]
                if ('var:reward_probabilities_M' in df.columns) and ('Block_number' in df.columns):
                    df.loc[df['Block_number'] == blocknum, 'reward_p_M'] = probs_m[int(blocknum - 1)]
    return df


def loadcsvdata(projectdir,
                bigtable=pd.DataFrame(),
                projectnames_needed=None,
                experimentnames_needed=None,
                setupnames_needed=None,
                sessionnames_needed=None,
                load_only_last_day=False):
    bigtable_orig = bigtable
    # bigtable=pd.DataFrame()
    # projectdir = Path('/home/rozmar/Network/BehaviorRig/Behavroom-Stacked-2/labadmin/Documents/Pybpod/Projects')
    # projectdir = Path('/home/rozmar/Data/Behavior/Projects')

    projectdir = Path(projectdir)
    if not projectdir.exists():
        raise RuntimeError('Project directory not found: {}'.format(projectdir))

    if type(bigtable_orig) == pd.DataFrame and len(bigtable) > 0:
        sessionnamessofar = bigtable['session'].unique()
        sessionnamessofar = np.sort(sessionnamessofar)
        sessionnametodel = sessionnamessofar[-1]
        bigtable = bigtable[bigtable['session'] != sessionnametodel]
        sessionnamessofar = sessionnamessofar[:-1]  # we keep reloading the last one
    else:
        sessionnamessofar = []
    projectnames = list()
    for projectname in projectdir.iterdir():
        if projectname.is_dir() and (not projectnames_needed or projectname.name in projectnames_needed):
            projectnames.append(projectname)
            experimentnames = list()
            for experimentname in (projectname / 'experiments').iterdir():
                if experimentname.is_dir() and (
                        not experimentnames_needed or experimentname.name in experimentnames_needed):
                    experimentnames.append(experimentname)
                    setupnames = list()
                    for setupname in (experimentname / 'setups').iterdir():
                        if setupname.is_dir() and (not setupnames_needed or setupname.name in setupnames_needed):
                            # if setupname.name == 'Foraging-0':
                            setupnames.append(setupname)
                            # a json file can be opened here
                            sessionnames = list()

                            if load_only_last_day:
                                for sessionname in (setupname / 'sessions').iterdir():
                                    if sessionname.is_dir() and (
                                            not sessionnames_needed or sessionname.name in sessionnames_needed):
                                        sessionnames.append(sessionname.name[:8])  # only the date
                                sessionnames = np.sort(sessionnames)
                                sessiondatetoload = sessionnames[-1]
                                sessionnames = list()

                            for sessionname in (setupname / 'sessions').iterdir():
                                if sessionname.is_dir() and (
                                        not sessionnames_needed or sessionname.name in sessionnames_needed) and (
                                        not load_only_last_day or sessiondatetoload in sessionname.name):
                                    sessionnames.append(sessionname)
                                    csvfilename = (sessionname / (sessionname.name + '.csv'))
                                    if csvfilename.is_file() and sessionname.name not in sessionnamessofar:  # there is a .csv file
                                        df = load_and_parse_a_csv_file(csvfilename)
                                        df['project'] = projectname.name
                                        df['experiment'] = experimentname.name
                                        df['setup'] = setupname.name
                                        df['session'] = sessionname.name
                                        if type(bigtable) != pd.DataFrame or len(bigtable) == 0:
                                            bigtable = df
                                        else:
                                            for colname in df.columns:
                                                if colname not in bigtable.columns:
                                                    bigtable[colname] = np.NaN
                                            for colname in bigtable.columns:
                                                if colname not in df.columns:
                                                    df[colname] = np.NaN
                                            bigtable = bigtable.append(df)

    bigtable = bigtable.drop_duplicates(subset = ['TYPE', 'PC-TIME', 'MSG', '+INFO'])
    if type(bigtable_orig) == pd.DataFrame and len(bigtable) != len(bigtable_orig):
        bigtable = bigtable.reset_index(drop = True)
    return bigtable


# %%
def minethedata(data):
    idxes = dict()
    times = dict()
    values = dict()
    idxes['lick_L'] = data['var:WaterPort_L_ch_in'] == data['+INFO'].values
    times['lick_L'] = data['PC-TIME'][idxes['lick_L']]
    idxes['choice_L'] = (data['MSG'] == 'Choice_L') & (data['TYPE'] == 'TRANSITION')
    times['choice_L'] = data['PC-TIME'][idxes['choice_L']]
    idxes['reward_L'] = (data['MSG'] == 'Reward_L') & (data['TYPE'] == 'TRANSITION')
    times['reward_L'] = data['PC-TIME'][idxes['reward_L']]
    idxes['autowater_L'] = (data['MSG'] == 'Auto_Water_L') & (data['TYPE'] == 'TRANSITION')
    times['autowater_L'] = data['PC-TIME'][idxes['autowater_L']]
    idxes['autowater_R'] = (data['MSG'] == 'Auto_Water_R') & (data['TYPE'] == 'TRANSITION')
    times['autowater_R'] = data['PC-TIME'][idxes['autowater_R']]
    idxes['lick_R'] = data['var:WaterPort_R_ch_in'] == data['+INFO']
    times['lick_R'] = data['PC-TIME'][idxes['lick_R']]
    idxes['choice_R'] = (data['MSG'] == 'Choice_R') & (data['TYPE'] == 'TRANSITION')
    times['choice_R'] = data['PC-TIME'][idxes['choice_R']]
    idxes['reward_R'] = (data['MSG'] == 'Reward_R') & (data['TYPE'] == 'TRANSITION')
    times['reward_R'] = data['PC-TIME'][idxes['reward_R']]
    if 'var:WaterPort_M_ch_in' in data.keys():
        idxes['lick_M'] = data['var:WaterPort_M_ch_in'] == data['+INFO']
        times['lick_M'] = data['PC-TIME'][idxes['lick_M']]
        idxes['choice_M'] = (data['MSG'] == 'Choice_M') & (data['TYPE'] == 'TRANSITION')
        times['choice_M'] = data['PC-TIME'][idxes['choice_M']]
        idxes['reward_M'] = (data['MSG'] == 'Reward_M') & (data['TYPE'] == 'TRANSITION')
        times['reward_M'] = data['PC-TIME'][idxes['reward_M']]
        idxes['autowater_M'] = (data['MSG'] == 'Auto_Water_M') & (data['TYPE'] == 'TRANSITION')
        times['autowater_M'] = data['PC-TIME'][idxes['autowater_M']]
    idxes['trialstart'] = data['TYPE'] == 'TRIAL'
    times['trialstart'] = data['PC-TIME'][idxes['trialstart']]
    idxes['trialend'] = data['TYPE'] == 'END-TRIAL'
    times['trialend'] = data['PC-TIME'][idxes['trialend']]
    idxes['GoCue'] = (data['MSG'] == 'GoCue') & (data['TYPE'] == 'TRANSITION')
    times['GoCue'] = data['PC-TIME'][idxes['GoCue']]

    idxes['reward_p_L'] = idxes['GoCue']
    times['reward_p_L'] = data['PC-TIME'][idxes['reward_p_L']]
    values['reward_p_L'] = data['reward_p_L'][idxes['reward_p_L']]

    idxes['reward_p_R'] = idxes['GoCue']
    times['reward_p_R'] = data['PC-TIME'][idxes['reward_p_R']]
    values['reward_p_R'] = data['reward_p_R'][idxes['reward_p_R']]

    idxes['motor_position_lateral'] = idxes['GoCue']
    times['motor_position_lateral'] = data['PC-TIME'][idxes['motor_position_lateral']]
    values['motor_position_lateral'] = data['var_motor:LickPort_Lateral_pos'][idxes['motor_position_lateral']]

    idxes['motor_position_rostrocaudal'] = idxes['GoCue']
    times['motor_position_rostrocaudal'] = data['PC-TIME'][idxes['motor_position_rostrocaudal']]
    values['motor_position_rostrocaudal'] = data['var_motor:LickPort_RostroCaudal_pos'][
        idxes['motor_position_rostrocaudal']]

    if 'var:WaterPort_M_ch_in' in data.keys():
        idxes['reward_p_M'] = idxes['GoCue']
        times['reward_p_M'] = data['PC-TIME'][idxes['reward_p_M']]
        values['reward_p_M'] = data['reward_p_M'][idxes['reward_p_M']]
    idxes['p_reward_ratio'] = idxes['GoCue']
    times['p_reward_ratio'] = times['reward_p_R']
    values['p_reward_ratio'] = values['reward_p_R'] / (values['reward_p_R'] + values['reward_p_L'])
    if 'reward_p_M' in values.keys() and len(values['reward_p_M']) > 0:
        values['p_reward_ratio_R'] = values['reward_p_R'] / (
                    values['reward_p_R'] + values['reward_p_L'] + values['reward_p_M'])
        values['p_reward_ratio_M'] = values['reward_p_M'] / (
                    values['reward_p_R'] + values['reward_p_L'] + values['reward_p_M'])
        values['p_reward_ratio_L'] = values['reward_p_L'] / (
                    values['reward_p_R'] + values['reward_p_L'] + values['reward_p_M'])
    return times, idxes, values


# %%
# bigtable = loadcsvdata(projectdir = '/home/rozmar/Data/Behavior/Projects')

# %%
# df = bigtable

def save_pickles_for_online_analysis(projectdir, projectnames_needed=None, experimentnames_needed=None,
                                     setupnames_needed=None, load_only_last_day=False):
    dirstructure = dict()
    projectnames = list()
    experimentnames = list()
    setupnames = list()
    sessionnames = list()

    # projectdir= defpath
    projectdir = Path(projectdir)
    if not projectdir.exists():
        raise RuntimeError('Project directory not found: {}'.format(projectdir))

    for projectname in projectdir.iterdir():
        if projectname.is_dir() and (not projectnames_needed or projectname.name in projectnames_needed):
            dirstructure[projectname.name] = dict()
            projectnames.append(projectname.name)

            projectdir_export = projectname / 'experiments_exported'
            if 'experiments_exported' not in os.listdir(projectname):
                (projectdir_export).mkdir()

            for experimentname in (projectname / 'experiments').iterdir():
                if experimentname.is_dir() and (
                        not experimentnames_needed or experimentname.name in experimentnames_needed):
                    dirstructure[projectname.name][experimentname.name] = dict()
                    experimentnames.append(experimentname.name)

                    experimentname_export = projectdir_export / experimentname.name
                    if experimentname.name not in os.listdir(projectdir_export):
                        (experimentname_export).mkdir()
                        experimentname_export = experimentname_export / 'setups'
                        (experimentname_export).mkdir()
                    else:
                        experimentname_export = experimentname_export / 'setups'
                    for setupname in (experimentname / 'setups').iterdir():
                        if setupname.is_dir() and (not setupnames_needed or setupname.name in setupnames_needed):
                            setupnames.append(setupname.name)
                            dirstructure[projectname.name][experimentname.name][setupname.name] = list()
                            # %
                            setupname_export = experimentname_export / setupname.name
                            if setupname.name not in os.listdir(experimentname_export):
                                (setupname_export).mkdir()
                                setupname_export = setupname_export / 'sessions'
                                (setupname_export).mkdir()
                            else:
                                setupname_export = setupname_export / 'sessions'

                            if load_only_last_day:
                                sessionnames_forsort = list()
                                for sessionname in (setupname / 'sessions').iterdir():
                                    if sessionname.is_dir():
                                        sessionnames_forsort.append(sessionname.name[:8])  # only the date
                                sessionnames_forsort = np.sort(np.unique(sessionnames_forsort))
                                sessiondatestoload = sessionnames_forsort[-5:]

                            for sessionname in (setupname / 'sessions').iterdir():
                                if sessionname.is_dir() and (
                                        not load_only_last_day or sessionname.name[:8] in sessiondatestoload):
                                    sessionnames.append(sessionname.name)
                                    dirstructure[projectname.name][experimentname.name][setupname.name].append(
                                        sessionname.name)
                                    if not os.path.exists(setupname_export / (sessionname.name + '.pkl')):
                                        doit = True
                                    elif os.stat(setupname_export / (sessionname.name + '.pkl')).st_mtime < os.stat(
                                            sessionname / (sessionname.name + '.csv')).st_mtime:
                                        doit = True
                                    else:
                                        doit = False
                                    if doit and os.path.exists(sessionname / (sessionname.name + '.csv')):
                                        df = load_and_parse_a_csv_file(sessionname / (sessionname.name + '.csv'))
                                        variables = dict()
                                        try:
                                            variables['times'], variables['idxes'], variables['values'] = minethedata(
                                                df)
                                            variables['experimenter'] = df['experimenter'][0]
                                            variables['subject'] = df['subject'][0]
                                        except:
                                            variables = dict()
                                        with open(setupname_export / (sessionname.name + '.tmp'), 'wb') as outfile:
                                            pickle.dump(variables, outfile)
                                        shutil.move(setupname_export / (sessionname.name + '.tmp'),
                                                    setupname_export / (sessionname.name + '.pkl'))


# =============================================================================
#                                     else:
#                                         print(sessionname.name+' skipped' )
# =============================================================================

def load_pickles_for_online_analysis(projectdir, projectnames_needed=None, experimentnames_needed=None,
                                     setupnames_needed=None, subjectnames_needed=None, load_only_last_day=False):
    # =============================================================================
    #     projectdir = Path(defpath)
    #     projectnames_needed = None
    #     experimentnames_needed = ['Foraging_homecage']
    #     setupnames_needed=None
    #     subjectnames_needed = None
    #     load_only_last_day = True
    # =============================================================================

    variables_out = dict()

    projectdir = Path(projectdir)
    if not projectdir.exists():
        raise RuntimeError('Project directory not found: {}'.format(projectdir))

    for projectname in projectdir.iterdir():
        if projectname.is_dir() and (not projectnames_needed or projectname.name in projectnames_needed):
            for experimentname in (projectname / 'experiments_exported').iterdir():
                if experimentname.is_dir() and (
                        not experimentnames_needed or experimentname.name in experimentnames_needed):

                    for setupname in (experimentname / 'setups').iterdir():
                        if setupname.is_dir() and (not setupnames_needed or setupname.name in setupnames_needed):
                            if load_only_last_day:
                                sessionnames = list()
                                for sessionname in os.listdir(setupname / 'sessions'):
                                    sessionnames.append(sessionname[:8])  # only the date
                                sessionnames = np.sort(np.unique(sessionnames))
                                sessiondatestoload = sessionnames[-5:]
                            for sessionname in os.listdir(setupname / 'sessions'):
                                if sessionname[-3:] == 'pkl' and (
                                        not load_only_last_day or sessionname[:8] in sessiondatestoload):
                                    # print('opening '+ sessionname)
                                    with open(setupname / 'sessions' / sessionname, 'rb') as readfile:
                                        variables_new = pickle.load(readfile)
                                    if len(variables_new.keys()) > 0:
                                        if not subjectnames_needed or variables_new['subject'] in subjectnames_needed:
                                            if len(variables_out.keys()) == 0:
                                                variables_out['times'] = dict()
                                                variables_out['times']['alltimes'] = np.asarray([])
                                                for key in variables_new['times'].keys():
                                                    variables_out['times'][key] = variables_new['times'][key].values
                                                    if len(variables_out['times']['alltimes']) > 0:
                                                        variables_out['times']['alltimes'] = np.concatenate((
                                                                                                            variables_out[
                                                                                                                'times'][
                                                                                                                'alltimes'],
                                                                                                            variables_new[
                                                                                                                'times'][
                                                                                                                key].values))
                                                    else:
                                                        variables_out['times']['alltimes'] = variables_new['times'][
                                                            key].values
                                                variables_out['values'] = dict()
                                                for key in variables_new['values'].keys():
                                                    variables_out['values'][key] = variables_new['values'][key].values
                                            else:
                                                for key in variables_new['times'].keys():
                                                    variables_out['times'][key] = np.concatenate((
                                                                                                 variables_out['times'][
                                                                                                     key],
                                                                                                 variables_new['times'][
                                                                                                     key].values))
                                                    variables_out['times']['alltimes'] = np.concatenate((variables_out[
                                                                                                             'times'][
                                                                                                             'alltimes'],
                                                                                                         variables_new[
                                                                                                             'times'][
                                                                                                             key].values))
                                                for key in variables_new['values'].keys():
                                                    variables_out['values'][key] = np.concatenate((variables_out[
                                                                                                       'values'][key],
                                                                                                   variables_new[
                                                                                                       'values'][
                                                                                                       key].values))

    return variables_out
