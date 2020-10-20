import pandas as pd
from pipeline import lab, experiment, foraging_analysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import timedelta
from scipy import stats
#dj.conn()


#%%
def merge_dataframes_with_nans(df_1,df_2,basiscol):
    basiscol = 'trial'
    colstoadd = list()
# =============================================================================
#     df_1 = df_behaviortrial
#     df_2 = df_reactiontimes
# =============================================================================
    for colnow in df_2.keys():
        if colnow not in df_1.keys():
            df_1[colnow] = np.nan
            colstoadd.append(colnow)
    for line in df_2.iterrows():
        for colname in colstoadd:
            df_1.loc[df_1[basiscol]==line[1][basiscol],colname]=line[1][colname]
    return df_1

def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = VPacker(children=boxes,align="center",pad=0, sep=1)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = HPacker(children=boxes,align="center", pad=0, sep=1)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.10, 0.2), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)
        

def extract_trials(plottype = '2lickport',
                   wr_name = 'FOR01',
                   sessions = (5,11),
                   show_bias_check_trials = True,
                   kernel = np.ones(10)/10,
                   filters=None,
                   local_matching = {'calculate_local_matching': True,
                                     'sliding_window':100,
                                     'matching_window':300,
                                     'matching_step':30,
                                     'efficiency_type': 'ideal'}):
    
    movingwindow = local_matching['sliding_window']
    fit_window = local_matching['matching_window']
    fit_step = local_matching['matching_step']
    
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(wr_name)).fetch1('subject_id')

    # Query of session-block-trial info
    q_session_block_trial = (experiment.SessionTrial * experiment.SessionBlock.BlockTrial
                             & 'subject_id = {}'.format(subject_id) 
                             & 'session >= {}'.format(sessions[0]) 
                             & 'session <= {}'.format(sessions[1])) 
    
    # Query of behavior info
    df_behaviortrial = pd.DataFrame((q_session_block_trial                         # Session-block-trial
                        * experiment.WaterPortChoice.proj(trial_choice='water_port')   # Choice
                        * experiment.BehaviorTrial                                 # Outcome
                        * (experiment.TrialEvent & 'trial_event_type = "go"')      # Go cue
                        * foraging_analysis.TrialStats                      # Reaction time
                        ).fetch('session', 'trial', 'block',
                                'trial_choice', 'outcome', 'early_lick',
                                'start_time', 'reaction_time',
                                # 'p_reward_left',
                                # 'p_reward_right',
                                # 'p_reward_middle',
                                'trial_event_time',
                                as_dict=True,
                                order_by=('session','trial')))
                                                                             
    # Handle p_{waterport}_reward
    for water_port in experiment.WaterPort.fetch('water_port'):
        p_reward_this = (q_session_block_trial * experiment.SessionBlock.WaterPortRewardProbability 
                                                      & 'water_port = "{}"'.format(water_port)).fetch(
                                                          'reward_probability', order_by=('session','trial')).astype(float)
        if len(p_reward_this) == len(df_behaviortrial):
            df_behaviortrial[f'p_reward_{water_port}'] = p_reward_this


    unique_sessions = df_behaviortrial['session'].unique()
    df_behaviortrial['iti']=np.nan
    df_behaviortrial['delay']=np.nan
    df_behaviortrial['early_count']=0
    df_behaviortrial.loc[df_behaviortrial['early_lick']=='early', 'early_count'] = 1
    df_behaviortrial['ignore_rate']= np.nan
    df_behaviortrial['reaction_time_smoothed'] = np.nan
    if type(filters) == dict:
        df_behaviortrial['keep_trial']=1
    
    for session in unique_sessions:
        # Relative to the beginning of selected sessions, not day 1 of training.
        total_trials_so_far = (foraging_analysis.SessionStats()&'subject_id = {}'.format(subject_id) 
                               &'session < {}'.format(session) &'session >= {}'.format(sessions[0])).fetch('session_total_trial_num')
        # bias_check_trials_now = (foraging_analysis.SessionStats()&'subject_id = {}'.format(subject_id) &'session = {}'.format(session)).fetch1('session_bias_check_trial_num')
        total_trials_so_far =sum(total_trials_so_far)
        gotime  = df_behaviortrial.loc[df_behaviortrial['session']==session, 'trial_event_time']
        trialtime  = df_behaviortrial.loc[df_behaviortrial['session']==session, 'start_time']
        itis = np.concatenate([[np.nan],np.diff(np.asarray(trialtime+gotime,float))])
        df_behaviortrial.loc[df_behaviortrial['session']==session, 'iti'] = itis
        df_behaviortrial.loc[df_behaviortrial['session']==session, 'delay'] = np.asarray(gotime,float)
        
        df_behaviortrial.loc[df_behaviortrial['session']==session, 'ignore_rate'] = np.convolve(df_behaviortrial.loc[df_behaviortrial['session']==session, 'outcome'] =='ignore',kernel,'same')
        reaction_time_interpolated = np.asarray(pd.DataFrame(np.asarray(df_behaviortrial.loc[df_behaviortrial['session']==session, 'reaction_time'].values,float)).interpolate().values.ravel().tolist())*1000
        df_behaviortrial.loc[df_behaviortrial['session']==session, 'reaction_time_smoothed'] = np.convolve( reaction_time_interpolated,kernel,'same')
        df_behaviortrial.loc[df_behaviortrial['session']==session, 'trial'] += total_trials_so_far
        
        if type(filters) == dict:
            max_idx = (df_behaviortrial.loc[df_behaviortrial['session']==session, 'ignore_rate'] > filters['ignore_rate_max']/100).idxmax()
            
            session_first_trial_idx = (df_behaviortrial['session']==session).idxmax()
            #print(max_idx)
            if max_idx > session_first_trial_idx or df_behaviortrial['ignore_rate'][session_first_trial_idx] > filters['ignore_rate_max']/100:
                df_behaviortrial.loc[df_behaviortrial.index.isin(np.arange(max_idx,len(df_behaviortrial))) & (df_behaviortrial['session']==session),'keep_trial']=0

#%
    if type(filters) == dict:
        trialstokeep = df_behaviortrial['keep_trial'] == 1
        df_behaviortrial = df_behaviortrial[trialstokeep]
        df_behaviortrial = df_behaviortrial.reset_index(drop=True)
        
    if not show_bias_check_trials:
        realtraining = (df_behaviortrial['p_reward_left']<1) & (df_behaviortrial['p_reward_right']<1) & ((df_behaviortrial['p_reward_middle']<1) | df_behaviortrial['p_reward_middle'].isnull())
        df_behaviortrial = df_behaviortrial[realtraining]
        df_behaviortrial = df_behaviortrial.reset_index(drop=True)
        
        
    #% --- calculating local matching, bias, reward rate ---
    if local_matching['calculate_local_matching']:
        movingwindow = min(movingwindow, len(df_behaviortrial))
        kernel = np.ones(movingwindow)/movingwindow
        p1 = np.asarray(np.max([df_behaviortrial['p_reward_right'],df_behaviortrial['p_reward_left']],0),float)
        p0 = np.asarray(np.min([df_behaviortrial['p_reward_right'],df_behaviortrial['p_reward_left']],0),float)
        
        m_star_greedy = np.floor(np.log(1-p1+1e-10)/np.log(1-p0+1e-10))
        p_star_greedy = p1 + (1-(1-p0)**(m_star_greedy+1)-p1**2)/(m_star_greedy+1)
        
        local_reward_rate = np.convolve(df_behaviortrial['outcome']=='hit',kernel,'same')
        max_reward_rate = np.convolve(p_star_greedy,kernel,'same')
        local_efficiency = local_reward_rate/max_reward_rate
        choice_right = np.asarray(df_behaviortrial['trial_choice']=='right')
        choice_left = np.asarray(df_behaviortrial['trial_choice']=='left')
        choice_middle = np.asarray(df_behaviortrial['trial_choice']=='middle')
        
        reward_rate_right =np.asarray((df_behaviortrial['trial_choice']=='right') &(df_behaviortrial['outcome']=='hit'))
        reward_rate_left =np.asarray((df_behaviortrial['trial_choice']=='left') &(df_behaviortrial['outcome']=='hit'))
        reward_rate_middle =np.asarray((df_behaviortrial['trial_choice']=='middle') &(df_behaviortrial['outcome']=='hit'))
        
    # =============================================================================
    #     choice_fraction_right = np.convolve(choice_right,kernel,'same')/np.convolve(choice_right+choice_left+choice_middle,kernel,'same')
    #     reward_fraction_right = np.convolve(reward_rate_right,kernel,'same')/local_reward_rate
    # =============================================================================
        with np.errstate(divide='ignore'):
            choice_rate_right= np.convolve(choice_right,kernel,'same')/(np.convolve(choice_left+choice_middle,kernel,'same') )
            reward_rate_right = np.convolve(reward_rate_right,kernel,'same')/(np.convolve(reward_rate_left+reward_rate_middle,kernel,'same') )
        slopes = list()
        intercepts = list()
        trial_number = list()
        for center_trial in np.arange(np.round(fit_window/2),len(df_behaviortrial),fit_step):
            #%
            reward_rates_now = reward_rate_right[int(np.round(center_trial-fit_window/2)):int(np.round(center_trial+fit_window/2))]
            choice_rates_now = choice_rate_right[int(np.round(center_trial-fit_window/2)):int(np.round(center_trial+fit_window/2))]
            todel = (reward_rates_now==0) | (choice_rates_now==0)
            reward_rates_now = reward_rates_now[~todel]
            choice_rates_now = choice_rates_now[~todel]
            try:
                x, y = np.log2(reward_rates_now), np.log2(choice_rates_now)
                idx = np.isfinite(x) & np.isfinite(y)
                slope_now, intercept_now = np.polyfit(x[idx], y[idx], 1)
                slopes.append(slope_now)
                intercepts.append(intercept_now)
                trial_number.append(center_trial)
            except:
                pass
            
        df_behaviortrial['local_efficiency']=local_efficiency
        df_behaviortrial['local_matching_slope'] = np.nan
        df_behaviortrial.loc[trial_number,'local_matching_slope']=slopes
        df_behaviortrial['local_matching_bias'] = np.nan
        df_behaviortrial.loc[trial_number,'local_matching_bias']=intercepts
    #%%
    
    return df_behaviortrial

#%%

def plot_rt_iti(df_behaviortrial,
                ax1,
                ax2,
                plottype = '2lickport',
                wr_name = 'FOR01',
                sessions = (5,11),
                show_bias_check_trials = True,
                kernel = np.ones(10)/10):
    
        
    sessionswitches = np.where(np.diff(df_behaviortrial['session'].values)>0)[0]
    if len(sessionswitches)>0:
        for trialnum_now in sessionswitches:
            ax1.plot([df_behaviortrial['trial'][trialnum_now],df_behaviortrial['trial'][trialnum_now]],[0,1000],'b--')    
            ax2.plot([df_behaviortrial['trial'][trialnum_now],df_behaviortrial['trial'][trialnum_now]],[0,1000],'b--')    
    
    
    delay_smoothed = np.asarray(pd.DataFrame(np.asarray(df_behaviortrial['delay'].values,float)).interpolate().values.ravel().tolist())
    delay_smoothed  = np.convolve(delay_smoothed ,kernel,'same')
    ax1.plot(df_behaviortrial['trial'],df_behaviortrial['reaction_time_smoothed'],'k-')#np.convolve(RT,kernel,'same')
    ax1.plot(df_behaviortrial['trial'],delay_smoothed,'m-') #np.convolve(df_behaviortrial['delay'],kernel,'same')
    ax1.plot(df_behaviortrial['trial'],np.convolve(df_behaviortrial['iti'],kernel,'same'),'r-') #
    ax1.set_yscale('log')
    ax1.set_ylim([1,1000])
    ax1.set_xlim([np.min(df_behaviortrial['trial'])-10,np.max(df_behaviortrial['trial'])+10]) 
    ax11 = ax1.twinx()
    ax11.plot(df_behaviortrial['trial'],df_behaviortrial['ignore_rate'],'g-') #np.convolve(df_behaviortrial['outcome']=='ignore',kernel,'same')
    ax11.set_ylim([0,1])
    ax11.set_ylabel('Ignore rate',color='g')
    ax11.spines["right"].set_color("green")
    
    multicolor_ylabel(ax1,('Delay (s)', ' ITI (s)','Reaction time (ms)'),('k','r','m'),axis='y')
    
    ax2.plot(df_behaviortrial['trial'],np.convolve(df_behaviortrial['early_count'],kernel,'same'),'y-')
    ax2.set_ylim([0,2])
    ax2.set_xlim([np.min(df_behaviortrial['trial'])-10,np.max(df_behaviortrial['trial'])+10]) 
    ax2.set_ylabel('Early lick rate',color='y')
    ax2.spines["right"].set_color("yellow")
    
    ax22 = ax2.twinx()
    ax22.plot(df_behaviortrial['trial'],np.convolve(df_behaviortrial['outcome']=='hit',kernel,'same'),'r-')
    ax22.set_ylim([-1,1])
    ax22.set_ylabel('Reward rate',color='r')
    ax22.spines["right"].set_color("red")
    ax2.set_xlabel('Trial #')
#%%
def plot_trials(df_behaviortrial,
                ax1,
                ax2=None,
                plottype = '2lickport',
                wr_name = 'FOR01',
                sessions = (5,11),
                plot_every_choice = True,
                show_bias_check_trials = True,
                choice_filter = np.ones(10)/10): 
    """This function downloads foraging sessions from datajoint and plots them"""
    

    
    df_behaviortrial['trial_choice_plot'] = np.nan
    df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'left', 'trial_choice_plot'] = 0
    df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'right', 'trial_choice_plot'] = 1
    df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'middle', 'trial_choice_plot'] = .5

    trial_choice_plot_interpolated = df_behaviortrial['trial_choice_plot'].values
    nans, x= np.isnan(trial_choice_plot_interpolated), lambda z: z.nonzero()[0]
    trial_choice_plot_interpolated[nans]= np.interp(x(nans), x(~nans), trial_choice_plot_interpolated[~nans])

    if plottype == '2lickport':
        
        #df_behaviortrial['reward_ratio']=df_behaviortrial['p_reward_right']/(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left'])
        df_behaviortrial['reward_ratio']=np.asarray(df_behaviortrial['p_reward_right'],float)/np.asarray(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left'],float)
        
        bias = np.convolve(trial_choice_plot_interpolated,choice_filter,mode = 'valid')
        bias = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),bias,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
    elif plottype == '3lickport':
        df_behaviortrial['reward_ratio_1']=df_behaviortrial['p_reward_left']/(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left']+ df_behaviortrial['p_reward_middle'])
        df_behaviortrial['reward_ratio_2']=(df_behaviortrial['p_reward_left']+df_behaviortrial['p_reward_middle'])/(df_behaviortrial['p_reward_right']+df_behaviortrial['p_reward_left']+ df_behaviortrial['p_reward_middle'])
        #%
        leftchoices_filtered = np.convolve(df_behaviortrial['trial_choice'] == 'left',choice_filter,mode = 'valid')
        leftchoices_filtered = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),leftchoices_filtered ,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
        rightchoices_filtered = np.convolve(df_behaviortrial['trial_choice'] == 'right',choice_filter,mode = 'valid')
        rightchoices_filtered = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),rightchoices_filtered ,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
        middlechoices_filtered = np.convolve(df_behaviortrial['trial_choice'] == 'middle',choice_filter,mode = 'valid')
        middlechoices_filtered = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),middlechoices_filtered ,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))
        allchoices_filtered = np.convolve(df_behaviortrial['trial_choice'] != 'none',choice_filter,mode = 'valid')
        allchoices_filtered = np.concatenate((np.nan*np.ones(int(np.floor((len(choice_filter)-1)/2))),allchoices_filtered ,np.nan*np.ones(int(np.ceil((len(choice_filter)-1)/2)))))

    rewarded = (df_behaviortrial['outcome']=='hit')
    unrewarded = (df_behaviortrial['outcome']=='miss')
    
    sessionswitches = np.where(np.diff(df_behaviortrial['session'].values)>0)[0]
    if len(sessionswitches)>0:
        for trialnum_now in sessionswitches:
            ax1.plot([df_behaviortrial['trial'][trialnum_now],df_behaviortrial['trial'][trialnum_now]],[-.15,1.15],'b--')
            
    if plottype == '2lickport':
        if plot_every_choice:
            ax1.plot(df_behaviortrial['trial'][rewarded],df_behaviortrial['trial_choice_plot'][rewarded],'k|',color='black',markersize=30,markeredgewidth=2)
            ax1.plot(df_behaviortrial['trial'][unrewarded],df_behaviortrial['trial_choice_plot'][unrewarded],'|',color='gray',markersize=15,markeredgewidth=2)
        ax1.plot(df_behaviortrial['trial'],bias,'k-',label = 'choice')
        ax1.plot(df_behaviortrial['trial'],df_behaviortrial['reward_ratio'],'y-')
        ax1.set_yticks((0,1))
        ax1.set_yticklabels(('left','right'))
    elif plottype == '3lickport':
        ax1.stackplot(np.asarray(df_behaviortrial['trial'],float),  leftchoices_filtered/allchoices_filtered ,  middlechoices_filtered/allchoices_filtered ,  rightchoices_filtered/allchoices_filtered ,colors=['r','g','b'], alpha=0.3 )
        if plot_every_choice:
            ax1.plot(df_behaviortrial['trial'][rewarded],df_behaviortrial['trial_choice_plot'][rewarded],'k|',color='black',markersize=30,markeredgewidth=2)
            ax1.plot(df_behaviortrial['trial'][unrewarded],df_behaviortrial['trial_choice_plot'][unrewarded],'|',color='gray',markersize=15,markeredgewidth=2)
        ax1.plot(df_behaviortrial['trial'],df_behaviortrial['reward_ratio_1'],'y-')
        ax1.plot(df_behaviortrial['trial'],df_behaviortrial['reward_ratio_2'],'y-')
        ax1.set_yticks((0,.5,1))
        ax1.set_yticklabels(('left','middle','right'))
        ax1.set_ylim([-.1,1.1])
        
    ax1.set_xlim([np.min(df_behaviortrial['trial'])-10,np.max(df_behaviortrial['trial'])+10]) 
    
    # probabilities    
    if ax2:
        ax2.clear()
        ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_left'],'r-')
        ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_right'],'b-')
        if plottype == '3lickport':
            ax2.plot(df_behaviortrial['trial'],df_behaviortrial['p_reward_middle'],'g-')
        ax2.set_ylabel('Reward probability')
        
        if plottype == '3lickport':
            legenda = ['left','right','middle']
        else:
            legenda = ['left','right']
        ax2.legend(legenda,fontsize='small',loc = 'upper right')
        ax2.set_xlim([np.min(df_behaviortrial['trial'])-10,np.max(df_behaviortrial['trial'])+10])  
        ax2.set(ylim=(-0.1, 1.1))
        
        if len(sessionswitches)>0:
            for trialnum_now in sessionswitches:
                ax2.plot([df_behaviortrial['trial'][trialnum_now],df_behaviortrial['trial'][trialnum_now]],[-.15,1.15],'b--')
        #%%
    return ax1, ax2
    
    
def plot_efficiency_matching_bias(ax3,
                                  plottype = '2lickport',
                                  wr_name = 'FOR01',
                                  sessions = (5,11),
                                  show_bias_check_trials = True,
                                  plot_efficiency_type='ideal'):
    
    
    subject_id = (lab.WaterRestriction()&'water_restriction_number = "{}"'.format(wr_name)).fetch1('subject_id')
    if show_bias_check_trials:
        maxrealforagingvalue = -1
    else:
        maxrealforagingvalue = 0
    df_blockefficiency = pd.DataFrame(foraging_analysis.BlockEfficiency()*foraging_analysis.BlockStats()*foraging_analysis.SessionTaskProtocol() & 
                                      'subject_id = {}'.format(subject_id) &
                                      'session >= {}'.format(sessions[0]) &
                                      'session <= {}'.format(sessions[1]) &
                                      'session_real_foraging > {}'.format(maxrealforagingvalue))

    df_blockefficiency =  df_blockefficiency.sort_values(["session", "block"], ascending = (True, True))
    unique_sessions = df_blockefficiency['session'].unique()
    df_blockefficiency['session_start_trialnum']=0
    session_start_trial_nums = list()
    session_end_trial_nums = list()
    for session in unique_sessions:
        total_trials_so_far = (foraging_analysis.SessionStats()&'subject_id = {}'.format(subject_id) 
                               &'session < {}'.format(session) &'session >= {}'.format(sessions[0])).fetch('session_total_trial_num')
        #bias_check_trials_now = (foraging_analysis.SessionStats()&'subject_id = {}'.format(subject_id) &'session = {}'.format(session)).fetch1('session_bias_check_trial_num')
        total_trials_so_far =sum(total_trials_so_far)
        session_start_trial_nums.append(total_trials_so_far)

        total_trials_now = (foraging_analysis.SessionStats()&'subject_id = {}'.format(subject_id) &'session = {}'.format(session)).fetch1('session_total_trial_num')
        session_end_trial_nums.append(total_trials_so_far+total_trials_now)
        #bias_check_trials_now = (foraging_analysis.SessionStats()&'subject_id = {}'.format(subject_id) &'session = {}'.format(session)).fetch1('session_bias_check_trial_num')

        df_blockefficiency.loc[df_blockefficiency['session']==session, 'session_start_trialnum'] += total_trials_so_far
        blocks = df_blockefficiency.loc[df_blockefficiency['session']==session, 'block'].values
        trial_num_so_far = 0
        for block in blocks:
            block_idx_now = (df_blockefficiency['session']==session) & (df_blockefficiency['block']==block)
            blocktrialnum = df_blockefficiency.loc[block_idx_now, 'block_trial_num'].values[0]
            df_blockefficiency.loc[block_idx_now, 'trialnum_block_middle'] = total_trials_so_far + trial_num_so_far + blocktrialnum/2
            trial_num_so_far += blocktrialnum

    if plot_efficiency_type == 'max_prob':
        eff_text = 'block_effi_one_p_reward'
    elif plot_efficiency_type == 'sum_prob':
        eff_text = 'block_effi_sum_p_reward'
    elif plot_efficiency_type == 'max_available':
        eff_text = 'block_effi_one_a_reward'
    elif plot_efficiency_type == 'sum_available':
        eff_text = 'block_effi_sum_a_reward'
    elif plot_efficiency_type == 'ideal':
        eff_text = 'block_ideal_phat_greedy'
    elif plot_efficiency_type == 'ideal_regret':
        eff_text = 'regret_ideal_phat_greedy'
    ax3.plot(df_blockefficiency['trialnum_block_middle'],df_blockefficiency[eff_text],'ko-')        
    session_switch_trial_nums = session_start_trial_nums.copy()
    session_switch_trial_nums.append(session_end_trial_nums[-1])
    for session_switch_trial_num in session_switch_trial_nums:
        ax3.plot([session_switch_trial_num,session_switch_trial_num],[-.15,1.15],'b--')
    ax3.set_xlim([0,np.max(session_switch_trial_nums)+10]) 
    
    # Session matching
    q_session_matching = (foraging_analysis.SessionMatching.WaterPortMatching * foraging_analysis.SessionStats
                          & 'subject_id = {}'.format(subject_id) 
                          & 'session >= {}'.format(sessions[0]) 
                          & 'session <= {}'.format(sessions[1]))
    
    water_port = 'right'  # Only match_idx_r and bias_r are needed
    match_idx_r,bias_r,sessions = np.asarray((q_session_matching & 'water_port = "{}"'.format(water_port)).fetch('match_idx','bias','session'))
    bias_r = (np.asarray(bias_r,float))
    
    #bias_r = (np.asarray(bias_r,float)+np.log2(10))/(np.log2(10)*2) # converting it between 0 and 1
    
    session_middle_trial_nums = list()
    for session_now in sessions:
        sessionidx = np.where(session_now == unique_sessions)[0]
        if len(sessionidx)>0:
            session_middle_trial_nums.extend((np.asarray(session_start_trial_nums)[sessionidx] + np.asarray(session_end_trial_nums)[sessionidx])/2)
        else:
            session_middle_trial_nums.append(np.nan)
    ax3.plot(session_middle_trial_nums,match_idx_r,'ro-')
    ax3.set_ylim([-.1,1.1])
    ax33 = ax3.twinx()
    ax33.plot(session_middle_trial_nums,bias_r,'yo-')
    ax33.set_ylim(np.asarray([-1.1,1.1])*np.nanmin([np.nanmax(np.abs(bias_r)),4]))
    ax33.set_ylabel('Bias',color='y')
    ax33.spines["right"].set_color("yellow")
    #ax33.tick_params(axis='y', colors='yellow')
    multicolor_ylabel(ax3,('Efficiency', ' Matching '),('r','k'),axis='y')
    #%%
    return ax3


def plot_local_efficiency_matching_bias(df_behaviortrial,ax3):
    
    ax3.plot(df_behaviortrial['trial'],df_behaviortrial['local_efficiency'],'k-')     
    sessionswitches = np.where(np.diff(df_behaviortrial['session'].values)>0)[0]
    if len(sessionswitches)>0:
        for trialnum_now in sessionswitches:
            ax3.plot([df_behaviortrial['trial'][trialnum_now],df_behaviortrial['trial'][trialnum_now]],[-.15,1.15],'b--')
     

   #%
    trialnums = df_behaviortrial.loc[~np.isnan(df_behaviortrial['local_matching_slope']),'trial']
    local_matching_slope = df_behaviortrial.loc[~np.isnan(df_behaviortrial['local_matching_slope']),'local_matching_slope']
    local_matching_bias = df_behaviortrial.loc[~np.isnan(df_behaviortrial['local_matching_slope']),'local_matching_bias']
    ax3.plot(trialnums,local_matching_slope,'ro-')
    ax3.set_ylim([-.1,1.1])
    ax33 = ax3.twinx()
    ax33.plot(trialnums,local_matching_bias,'yo-')
    ax33.set_ylim(np.asarray([-1.1,1.1])*np.nanmin([np.nanmax(np.abs(local_matching_bias.values).tolist() + [0]),4]))
    ax33.set_ylabel('Bias',color='y')
    ax33.spines["right"].set_color("yellow")
    ax3.set_xlim([np.min(df_behaviortrial['trial'])-10,np.max(df_behaviortrial['trial'])+10]) 
    #ax33.tick_params(axis='y', colors='yellow')
    multicolor_ylabel(ax3,('Efficiency', ' Matching '),('r','k'),axis='y')
    #%%
    return ax3

def plot_training_summary(use_days_from_start=False):
    #%%
    sns.set(style="darkgrid", context="talk", font_scale=1.2)
    sns.set_palette("muted")
    all_wr = lab.WaterRestriction().fetch('water_restriction_number', order_by=('wr_start_date', 'water_restriction_number'))

    exclude = []
    # exclude = ['HH01', 'HH04', 'HH06', 'HH07']
    # highlight = {('FOR01', 'FOR02', 'FOR03', 'FOR04'): dict(color='b'),
    #              ('FOR11', 'FOR12'): dict(color='g'),
    #              }
    # highlight = {('FOR01', 'FOR02', 'FOR03', 'FOR04'): dict(color='r'),
    #               ('HH01', 'HH04', 'HH06', 'HH07'): dict(color='b'),
    #               }
    highlight = {('HH01', 'HH04'): dict(marker='.'),
                  ('HH06', 'HH07'): dict(marker='o')
                  }
    
    fig1 = plt.figure(figsize=(20,12))
    ax = fig1.subplots(2,3)
    fig1.subplots_adjust(hspace=0.5, wspace=0.3)

    fig2 = plt.figure(figsize=(20,12))
    ax2 = fig2.subplots(2,3)
    fig2.subplots_adjust(hspace=0.5, wspace=0.3)
    
    fig3 = plt.figure(figsize=(16,12))
    ax3 = fig3.subplots(2,2)
    fig3.subplots_adjust(hspace=0.5, wspace=0.3)

    # Only mice who started with 2lp task
    for wr_name in all_wr:
        if wr_name in exclude:
            continue
        
        q_two_lp_foraging_sessions = (foraging_analysis.SessionTaskProtocol * lab.WaterRestriction
                                    & 'session_task_protocol=100'
                                    & 'water_restriction_number="{}"'.format(wr_name))

        # Skip this mice if it did not started with 2lp task
        two_lp_foraging_sessions = q_two_lp_foraging_sessions.fetch('session')
        if len(two_lp_foraging_sessions) == 0 or min(two_lp_foraging_sessions) > 1:
            continue
    
        # -- Get data --
        # Basic stats
        this_mouse_session_stats_raw = (foraging_analysis.SessionStats 
                                        * (foraging_analysis.SessionMatching.WaterPortMatching.proj('match_idx', 'bias') & 'water_port="right"') 
                                        & q_two_lp_foraging_sessions
                                        ).fetch(
                                            order_by='session', format='frame').reset_index()
        
        if use_days_from_start:  # Use the actual day from the first training day
            training_day = (experiment.Session & q_two_lp_foraging_sessions).fetch('session_date')
            training_day = ((training_day - min(training_day))/timedelta(days=1)).astype(int) + 1
            
            # Insert nans if no training day
            x = np.arange(1, max(training_day)+1)
            this_mouse_session_stats = pd.DataFrame(np.nan, index=x, columns=this_mouse_session_stats_raw.columns)
            this_mouse_session_stats.loc[training_day] = this_mouse_session_stats_raw.values

        else:  # Use continuous session number
            x = this_mouse_session_stats_raw['session']
            this_mouse_session_stats = this_mouse_session_stats_raw
        
        total_trial_num = this_mouse_session_stats['session_pure_choices_num'].values
        foraging_eff = this_mouse_session_stats['session_foraging_eff_optimal'].values * 100
        foraging_eff_random_seed = this_mouse_session_stats['session_foraging_eff_optimal_random_seed'].to_numpy(dtype=np.float) * 100
        early_lick_ratio = this_mouse_session_stats['session_early_lick_ratio'].values * 100
        reward_sum_mean = this_mouse_session_stats['session_mean_reward_sum'].values
        reward_contrast_mean = this_mouse_session_stats['session_mean_reward_contrast'].values
        block_length_mean = (this_mouse_session_stats['session_total_trial_num'] / this_mouse_session_stats['session_block_num']).values
        
        double_dip = this_mouse_session_stats['session_double_dipping_ratio'].to_numpy(dtype=np.float) * 100
        double_dip_hit = this_mouse_session_stats['session_double_dipping_ratio_hit'].to_numpy(dtype=np.float) * 100
        double_dip_miss = this_mouse_session_stats['session_double_dipping_ratio_miss'].to_numpy(dtype=np.float) * 100
        
        matching_idx = this_mouse_session_stats['match_idx']    
        matching_bias = this_mouse_session_stats['bias']        
        
        # Plot settings
        plot_setting = None        
        for h_wr in highlight:
            if wr_name in h_wr:
                plot_setting = {**highlight[h_wr], 'label': wr_name}
        
        if plot_setting is None:
            plot_setting = dict(lw=0.5, color='grey')
            
        # -- 1. Total finished trials --
        ax[0,0].plot(x, total_trial_num, **plot_setting)
        
        # -- 2. Session-wise foraging efficiency (optimal) --
        ax[0,1].plot(x, foraging_eff, **plot_setting)
        ax[0,2].plot(x, foraging_eff_random_seed, **plot_setting)
        ax2[0,0].plot(total_trial_num, foraging_eff, **plot_setting)
        ax2[0,2].plot(foraging_eff, foraging_eff_random_seed, '.')
        
        # -- 3. Matching bias and slope--
        ax[1,0].plot(x, abs(matching_bias.astype(float)), **plot_setting)
        ax[1,2].plot(x, matching_idx, **plot_setting)
        
        # -- 4. Early lick ratio --
        ax[1,1].plot(x, early_lick_ratio, **plot_setting)
        
        # -- 5. Reward schedule and block structure --
        ax2[1,0].plot(x, reward_sum_mean, **plot_setting)
        ax2[1,1].plot(x, reward_contrast_mean, **plot_setting)
        ax2[1,2].plot(x, block_length_mean, **plot_setting)
        
        # -- 6. Double dipping ratio --
        ax3[0,0].plot(x, double_dip, **plot_setting)
        ax3[0,1].plot(x, double_dip_hit, **plot_setting)
        ax3[1,0].plot(x, double_dip_miss, **plot_setting)
        ax3[1,1].plot(double_dip_hit, double_dip_miss, **plot_setting)
                
    x_name = 'Days from start' if use_days_from_start else 'Session number'
        
    ax[0,0].set(xlabel=x_name, title='Total finished trials')
    ax[0,0].legend()
    
    ax[0,1].set(xlabel=x_name, title='Foraging efficiency (optimal) %')
    ax[0,2].set(xlabel=x_name, title='Foraging efficiency (optimal_random_seed) %')
    ax[1,0].set(xlabel=x_name, title='abs(matching bias)', ylim=(-0.1, 5))
    ax[1,1].set(xlabel=x_name, title='Early lick trials %')
    ax[1,2].set(xlabel=x_name, title='Matching slope', ylim=(-0.1,1.1))
    
    ax2[0,0].set(xlabel='Total finished trials', ylabel='Foraging efficiency (optimal) %')
    ax2[0,2].set(xlabel='Foraging eff optimal', ylabel='Foraging eff random seed')
    ax2[0,2].plot([0,100], [0,100], 'k--')
    ax2[1,0].set(xlabel=x_name, title='Mean reward prob sum')
    ax2[1,0].legend()
    ax2[1,1].set(xlabel=x_name, title='Mean reward prob contrast', ylim=(0,10))
    ax2[1,2].set(xlabel=x_name, title='Mean block length')
    
    ax3[0,0].set(xlabel=x_name, title='Double dipping all (%)')
    ax3[0,0].legend()
    ax3[0,1].set(xlabel=x_name, title='Double dipping if hit (%)')
    ax3[1,0].set(xlabel=x_name, title='Double dipping if miss (%)')
    ax3[1,1].set(title='Double dipping (%)', xlabel='if hit', ylabel='if miss')
    ax3[1,1].plot([0,100], [0,100], 'k--')
    
    # == Matching idx vs foraging eff (quick and dirty) ==
    matching_vs_eff_setting = {('water_restriction_number NOT LIKE "HH%"', ): dict(marker='.', color='grey', linestyle = 'None'),
                               ('water_restriction_number = "HH01"', 'water_restriction_number = "HH04"'): dict(markersize=7, marker='o', color='b', linestyle = 'None'),
                               # 'water_restriction_number LIKE "HH%"': dict(marker='o', color='b', linestyle = 'None'),
                               ('water_restriction_number = "HH06"', 'water_restriction_number = "HH07"'): dict(marker='o', color='r', linestyle = 'None'),
                               # ('water_restriction_number = "FOR01"', 'water_restriction_number = "FOR02"',
                               #  'water_restriction_number = "FOR03"', 'water_restriction_number = "FOR04"'): dict(marker='v', color='r', linestyle = 'None')
                              }
    
    for ss in matching_vs_eff_setting:
        q_this_sessions = (foraging_analysis.SessionTaskProtocol * lab.WaterRestriction 
                           & 'session_real_foraging=1' & 'session_task_protocol=100'
                           & ss
                          )
        
        this_matching_eff = (foraging_analysis.SessionStats 
                           * (foraging_analysis.SessionMatching.WaterPortMatching.proj('match_idx', 'bias') & 'water_port="right"') 
                            & q_this_sessions
                           ).fetch('match_idx', 'session_foraging_eff_optimal')
        
        x = this_matching_eff[0].astype(float)
        y = this_matching_eff[1].astype(float)*100
        mask = ~np.isnan(x) & ~np.isnan(y)
        k, b, r_value, p, std_err = stats.linregress(x[mask], y[mask])
        
        ax2[0,1].plot(this_matching_eff[0], this_matching_eff[1]*100, **matching_vs_eff_setting[ss])
        ax2[0,1].plot([np.nanmin(x),np.nanmax(x)], [k*np.nanmin(x)+b, k*np.nanmax(x)+b,], '-', color=matching_vs_eff_setting[ss]['color'], label='r = %.2g\np = %.2g'%(r_value, p))

        
    ax2[0,1].legend(fontsize=15)
    ax2[0,1].set(xlabel='Matching slope', ylabel='Foraging efficiency (optimal) %', xlim=(0,1), ylim=(50,110))


    #%%
    
def plot_session_trial_events(key_subject_id_session = dict(subject_id=472184, session=14)):  # Plot all trial events of one specific session
    #%% 
    sns.set(style="darkgrid", context="talk", font_scale=1.2)
    sns.set_palette("muted")
    fig = plt.figure(figsize=(20,12))
    gs = GridSpec(5, 1, wspace=0.4, hspace=0.4, bottom=0.07, top=0.95, left=0.1, right=0.9) 
    
    ax1 = fig.add_subplot(gs[0:3, :])
    ax2 = fig.add_subplot(gs[3, :])
    ax3 = fig.add_subplot(gs[4, :])
    ax1.get_shared_x_axes().join(ax1, ax2, ax3)
    
    # Plot settings
    plot_setting = {'left lick':'red', 'right lick':'blue'}  
    
    # -- Get event times --
    go_cue_times = (experiment.TrialEvent() & key_subject_id_session & 'trial_event_type="go"').fetch('trial_event_time', order_by='trial').astype(float)
    lick_times = pd.DataFrame((experiment.ActionEvent() & key_subject_id_session).fetch(order_by='trial'))
    
    trial_num = len(go_cue_times)
    all_trial_num = np.arange(1, trial_num+1).tolist()
    all_trial_start = [[-x] for x in go_cue_times]
    all_lick = dict()
    for event_type in plot_setting:
        all_lick[event_type] = []
        for i, trial_start in enumerate(all_trial_start):
            all_lick[event_type].append((lick_times[(lick_times['trial']==i+1) & (lick_times['action_event_type']==event_type)]['action_event_time'].values.astype(float) + trial_start).tolist())
    
    # -- All licking events (Ordered by trials) --
    ax1.plot([0, 0], [0, trial_num], 'k', lw=0.5)   # Aligned by go cue
    ax1.set(ylabel='Trial number', xlim=(-3, 3), xticks=[])

    # Batch plotting to speed up    
    ax1.eventplot(lineoffsets=all_trial_num, positions=all_trial_start, color='k')   # Aligned by go cue
    for event_type in plot_setting:
        ax1.eventplot(lineoffsets=all_trial_num, 
                     positions=all_lick[event_type],
                     color=plot_setting[event_type],
                     linewidth=2)   # Trial start
    
    # -- Histogram of all licks --
    for event_type in plot_setting:
        sns.histplot(np.hstack(all_lick[event_type]), binwidth=0.01, alpha=0.5, 
                     ax=ax2, color=plot_setting[event_type], label=event_type)  # 10-ms window
    
    ymax_tmp = max(ax2.get_ylim())  
    sns.histplot(-go_cue_times, binwidth=0.01, color='k', ax=ax2, label='trial start')  # 10-ms window
    ax2.axvline(x=0, color='k', lw=0.5)
    ax2.set(ylim=(0, ymax_tmp), xticks=[], title='All events') # Fix the ylim of left and right licks
    ax2.legend()
    
    # -- Histogram of reaction time (first lick after go cue) --   
    plot_setting = {'LEFT':'red', 'RIGHT':'blue'}  
    for water_port in plot_setting:
        this_RT = (foraging_analysis.TrialStats() & key_subject_id_session & (experiment.WaterPortChoice() & 'water_port="{}"'.format(water_port))).fetch('reaction_time').astype(float)
        sns.histplot(this_RT, binwidth=0.01, alpha=0.5, 
                     ax=ax3, color=plot_setting[water_port], label=water_port)  # 10-ms window
    ax3.axvline(x=0, color='k', lw=0.5)
    ax3.set(xlabel='Time to Go Cue (s)', title='Reaction time') # Fix the ylim of left and right licks
    ax3.legend()
    

def analyze_runlength(result_path = "..\\results\\model_comparison\\", combine_prefix = 'model_comparison_15_', 
                          group_results_name = 'group_results.npz', mice_of_interest = ['FOR05', 'FOR06'], 
                          efficiency_partitions = [30, 30],  block_partitions = [70, 70], if_first_plot = True):
    sns.set()

    # Load dataframe
    data = np.load(result_path + group_results_name, allow_pickle=True)
    group_results = data.f.group_results.item()
    results_all_mice = group_results['results_all_mice'] 
    
    palette_all = sns.color_palette("RdYlGn", max(results_all_mice['session_number'].unique()))

    for mouse in mice_of_interest:
        
        # Load raw data
        data_raw = np.load(result_path + combine_prefix + mouse + '.npz', allow_pickle=True)
        data_raw = data_raw.f.results_each_mice.item()
        
        df_this = results_all_mice[results_all_mice.mice == mouse].copy()
        df_this[['foraging_efficiency', 'prediction_accuracy_CV_test', 'prediction_accuracy_bias_only']] *= 100

        efficiency_thres = np.percentile(df_this.foraging_efficiency, [100-efficiency_partitions[0], efficiency_partitions[1]])

        #%% Plot foraging histogram 
        if if_first_plot: 
            
            x = df_this.prediction_accuracy_NONCV
            y = df_this.foraging_efficiency
            (r, p) = pearsonr(x, y)
  
            g = sns.jointplot(x="prediction_accuracy_CV_test", y="foraging_efficiency", data = df_this.sort_values(by = 'session_number'), 
                              kind="reg", color="b", marginal_kws = {'bins':20,'color':'k'}, joint_kws = {'marker':'', 'color':'k', 
                                                                                                          'label':'r$^2$ = %.3f, p = %.3f'%(r**2,p)})
            
            palette = []
            for s in np.sort(df_this['session_number'].unique()):
                palette.append(palette_all[s-1])

            g.plot_joint(plt.scatter, color = palette, sizes = df_this.n_trials**2 / 3000, alpha = 0.7)
            plt.legend()
            ax = plt.gca()
            ax.axvline(50, c='k', ls='--')
            ax.axhline(100, c='k', ls='--')
            
            ax.axhline(efficiency_thres[1], c='r', ls='-.', lw = 2)        
            ax.axhline(efficiency_thres[0], c='g', ls='-.', lw = 2)
            plt.gcf().text(0.01, 0.95, mouse)

        #%% Get grand runlength (Lau)
        good_session_idxs = df_this[df_this.foraging_efficiency >= efficiency_thres[0]].session_idx
        bad_session_idxs = df_this[df_this.foraging_efficiency <= efficiency_thres[1]].session_idx
        
        grand_session_idxs = [good_session_idxs, bad_session_idxs]
        grand_session_idxs_markers = [mouse + ' best %g%% sessions (n = %g)' % (efficiency_partitions[0], len(good_session_idxs)), 
                                      mouse + ' worst %g%% sessions (n = %g)' % (efficiency_partitions[1], len(bad_session_idxs))]
        
        for this_session_idxs, this_marker in zip(grand_session_idxs, grand_session_idxs_markers):
            
            df_run_length_Lau_all = [pd.DataFrame(), pd.DataFrame()] # First and last trials in each block
            
            for this_idx in this_session_idxs:
                #%%
                this_class = data_raw['model_comparison_session_wise'][this_idx - 1]
                this_session_num = df_this[df_this.session_idx == this_idx].session_number.values
                
                p_reward = data_raw['model_comparison_grand'].p_reward[:, data_raw['model_comparison_grand'].session_num == this_session_num]
                
                # Runlength analysis
                this_df_run_length_Lau = analyze_runlength_Lau2005(this_class.fit_choice_history, p_reward, block_partitions = block_partitions)
                
                for i in [0,1]:
                    df_run_length_Lau_all[i] = df_run_length_Lau_all[i].append(this_df_run_length_Lau[i])
                
            fig = plot_runlength_Lau2005(df_run_length_Lau_all, block_partitions)
            fig.text(0.1, 0.92, this_marker + ', mean foraging eff. = %g%%, %g blocks' %\
                           (np.mean(df_this.foraging_efficiency[this_session_idxs - 1]), len(df_run_length_Lau_all[0])), fontsize = 15)
            plt.show()
            
            
def plot_runlength_Lau2005(df_run_length_Lau, block_partitions = ['unknown', 'unknown']):
    #%%
    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8), dpi = 150)
    gs = GridSpec(2, 3, hspace = 0.6, wspace = 0.4, 
                  left = 0.1, right = 0.95, bottom = 0.14, top = 0.85)
    
    annotations = ['First %g%% trials'%block_partitions[0], 'Last %g%% trials'%block_partitions[1]]
    
    for pp, df_this_half in enumerate(df_run_length_Lau):
        
        df_this_half = df_this_half[~ df_this_half.isin([0, np.inf, -np.inf]).choice_ratio]
        df_this_half = df_this_half[~ df_this_half.isin([0, np.inf, -np.inf]).p_base_ratio]
        
        # == Fig.5 Lau 2005 ==
        ax = fig.add_subplot(gs[pp,0]) 
        
        plt.plot(df_this_half.choice_ratio, df_this_half.mean_runlength_rich, 'go', label = 'Rich', alpha = 0.7, markersize = 5)
        plt.plot(df_this_half.choice_ratio, df_this_half.mean_runlength_lean, 'rx', label = 'Lean', alpha = 0.7, markersize = 8)
        
        ax.axhline(1, c = 'r', ls = '--', lw = 1)
        plt.plot([1,16],[1,16], c ='g', ls = '--', lw = 1)
        # ax.axvline(1, c = 'k', ls = '--', lw = 1)
        
        plt.plot(mean_runlength_Bernoulli[2,:], mean_runlength_Bernoulli[0,:], 'k--', lw = 1)
        plt.plot(mean_runlength_Bernoulli[2,:], mean_runlength_Bernoulli[1,:], 'k-', lw = 1)
        
        ax.set_xscale('log')
        ax.set_xticks([1,2,4,8,16])
        ax.set_xticklabels([1,2,4,8,16])
        ax.set_xlim([0.9,16])
        ax.set_yscale('log')
        ax.set_yticks([1,2,4,8,16])
        ax.set_yticklabels([1,2,4,8,16])
        ax.set_ylim([0.9,16])
        
        # ax.axis('equal')
       
        plt.xlabel('Choice ratio (#rich / #lean)')
        plt.ylabel('Mean runlength')
        plt.legend()
        plt.title(annotations[pp])
    
        # == Mean rich runlength VS optimal rich runlength (m*) ==
        ax = fig.add_subplot(gs[pp,1]) 
        
        x = df_this_half.m_star
        y = df_this_half.mean_runlength_rich
        
        sns.regplot(x=x, y=y, ax = ax)
        
        try:
            (r, p) = pearsonr(x, y)
            plt.annotate( 'r = %.3g\np = %.3g'%(r,p), xy=(0, 0.8), xycoords=ax.transAxes, fontsize = 9)
        except:
            pass
        
        plt.plot([0, 15],[0, 15], 'b--', lw = 1)
        plt.xlabel('Optimal rich runlength')
        plt.ylabel('Mean rich runlength')
        ax.set_xlim([0,15])
        ax.set_ylim([0,15])
    
        # == Choice ratio VS optimal rich runlength (m*) ==
        ax = fig.add_subplot(gs[pp,2]) 
        x = df_this_half.m_star
        y = df_this_half.choice_ratio
        
        try:
            (r, p) = pearsonr(x, y)
            plt.annotate( 'r = %.3g\np = %.3g'%(r,p), xy=(0, 0.8), xycoords=ax.transAxes, fontsize = 9)
        except:
            pass
        
        sns.regplot(x=x, y=y, ax = ax)
        
        plt.plot([0, 15],[0, 15], 'b--', lw = 1)
        plt.xlabel('Optimal rich runlength')
        plt.ylabel('Choice ratio (#rich / #lean)')
        ax.set_xlim([0,15])
        ax.set_ylim([0,15])

    return fig
    
def plot_foragingWebGUI_subject(wr_name_selected='HH07', session_selected=17, show_others=True, use_days_from_start=False):
    #%%
    sns.set(style="darkgrid", context="talk", font_scale=1.2)
    sns.set_palette("muted")
    all_wr = lab.WaterRestriction().fetch('water_restriction_number', order_by=('wr_start_date', 'water_restriction_number'))

    fig1 = plt.figure(figsize=(10,20))
    ax = fig1.subplots(4,2)
    fig1.subplots_adjust(hspace=0.5, wspace=0.3)

    # Only mice who started with 2lp task
    for wr_name in all_wr:
        if not show_others and wr_name != wr_name_selected:
            continue
        
        q_two_lp_foraging_sessions = (foraging_analysis.SessionTaskProtocol * lab.WaterRestriction
                                    & 'session_task_protocol=100'
                                    & 'water_restriction_number="{}"'.format(wr_name))

        # Skip this mice if it did not started with 2lp task
        two_lp_foraging_sessions = q_two_lp_foraging_sessions.fetch('session')
        if len(two_lp_foraging_sessions) == 0 or min(two_lp_foraging_sessions) > 1:
            continue
    
        # -- Get data --
        # Basic stats
        this_mouse_session_stats_raw = (foraging_analysis.SessionStats 
                                        * (foraging_analysis.SessionMatching.WaterPortMatching.proj('match_idx', 'bias') & 'water_port="right"') 
                                        & q_two_lp_foraging_sessions
                                        ).fetch(
                                            order_by='session', format='frame').reset_index()
        
        if use_days_from_start:  # Use the actual day from the first training day
            training_day = (experiment.Session & q_two_lp_foraging_sessions).fetch('session_date')
            training_day = ((training_day - min(training_day))/timedelta(days=1)).astype(int) + 1
            
            # Insert nans if no training day
            x = np.arange(1, max(training_day)+1)
            this_mouse_session_stats = pd.DataFrame(np.nan, index=x, columns=this_mouse_session_stats_raw.columns)
            this_mouse_session_stats.loc[training_day] = this_mouse_session_stats_raw.values

        else:  # Use continuous session number
            x = this_mouse_session_stats_raw['session']
            this_mouse_session_stats = this_mouse_session_stats_raw
        
        total_trial_num = this_mouse_session_stats['session_pure_choices_num'].values
        foraging_eff = this_mouse_session_stats['session_foraging_eff_optimal'].values * 100
        # foraging_eff_random_seed = this_mouse_session_stats['session_foraging_eff_optimal_random_seed'].to_numpy(dtype=np.float) * 100
        early_lick_ratio = this_mouse_session_stats['session_early_lick_ratio'].values * 100
        reward_sum_mean = this_mouse_session_stats['session_mean_reward_sum'].values
        reward_contrast_mean = this_mouse_session_stats['session_mean_reward_contrast'].values
        block_length_mean = (this_mouse_session_stats['session_total_trial_num'] / this_mouse_session_stats['session_block_num']).values
        
        double_dip = this_mouse_session_stats['session_double_dipping_ratio'].to_numpy(dtype=np.float) * 100
        # double_dip_hit = this_mouse_session_stats['session_double_dipping_ratio_hit'].to_numpy(dtype=np.float) * 100
        # double_dip_miss = this_mouse_session_stats['session_double_dipping_ratio_miss'].to_numpy(dtype=np.float) * 100
        
        # matching_idx = this_mouse_session_stats['match_idx'] 
        matching_bias = this_mouse_session_stats['bias']        
        
        # Plot settings
        if wr_name == wr_name_selected:
            plot_setting = dict(lw=2, color='b')
            selected_idx = np.where(this_mouse_session_stats['session'] == session_selected)[0][0]
        else:
            plot_setting = dict(lw=0.5, color='grey')
            
        # -- 1. Total finished trials --
        ax[0,0].plot(x, total_trial_num, **plot_setting)
        if wr_name == wr_name_selected: ax[0,0].plot(x[selected_idx], total_trial_num[selected_idx], **plot_setting, marker='o', label=wr_name)
        
        # -- 2. Session-wise foraging efficiency (optimal) --
        ax[0,1].plot(x, foraging_eff, **plot_setting)
        if wr_name == wr_name_selected: ax[0,1].plot(x[selected_idx], foraging_eff[selected_idx], **plot_setting, marker='o', label=wr_name)
        
        # -- 3. Matching bias --
        ax[1,0].plot(x, abs(matching_bias.astype(float)), **plot_setting)
        if wr_name == wr_name_selected: ax[1,0].plot(x[selected_idx], abs(matching_bias.astype(float)[selected_idx]), **plot_setting, marker='o', label=wr_name)
        
        # -- 4. Early lick ratio --
        ax[1,1].plot(x, early_lick_ratio, **plot_setting)
        if wr_name == wr_name_selected: ax[1,1].plot(x[selected_idx], early_lick_ratio[selected_idx], **plot_setting, marker='o', label=wr_name)
        
        # -- 5. Reward schedule and block structure --
        ax[2,1].plot(x, reward_sum_mean, **plot_setting)
        ax[3,0].plot(x, reward_contrast_mean, **plot_setting)
        ax[3,1].plot(x, block_length_mean, **plot_setting)

        if wr_name == wr_name_selected:
            ax[2,1].plot(x[selected_idx], reward_sum_mean[selected_idx], **plot_setting, marker='o', label=wr_name)
            ax[3,0].plot(x[selected_idx], reward_contrast_mean[selected_idx], **plot_setting, marker='o', label=wr_name)
            ax[3,1].plot(x[selected_idx], block_length_mean[selected_idx], **plot_setting, marker='o', label=wr_name)
        
        # -- 6. Double dipping ratio --
        ax[2,0].plot(x, double_dip, **plot_setting)
        if wr_name == wr_name_selected: ax[2,0].plot(x[selected_idx], double_dip[selected_idx], **plot_setting, marker='o', label=wr_name)
        #  ax[0,1].plot(x, double_dip_hit, **plot_setting)
              
    x_name = 'Days from start' if use_days_from_start else 'Session number'
        
    ax[0,0].set(xlabel=x_name, title='Total finished trials')
    ax[0,0].legend()
    
    ax[0,1].set(xlabel=x_name, title='Foraging efficiency (optimal) %')
    ax[1,0].set(xlabel=x_name, title='abs(matching bias)', ylim=(-0.1, 5))
    ax[1,1].set(xlabel=x_name, title='Early lick trials %')
    
    ax[2,1].set(xlabel=x_name, title='Mean reward prob sum')
    ax[2,1].legend()
    ax[3,0].set(xlabel=x_name, title='Mean reward prob contrast', ylim=(0,10))
    ax[3,1].set(xlabel=x_name, title='Mean block length')
    
    ax[2,0].set(xlabel=x_name, title='Double dipping all (%)')
    ax[2,0].legend()
    # ax3[0,1].set(xlabel=x_name, title='Double dipping if hit (%)')
    #%%

def plot_foragingWebGUI_session(wr_name_selected='HH07', session_selected=17):  # Plot all trial events of one specific session
    # Reuse Marton's code with default parameters
    sns.set(style="darkgrid", context="talk", font_scale=1.2)
    sns.set_palette("muted")

    fig=plt.figure()
    
    ax1=fig.add_axes([0,0,2,.8])
    ax1.set_title(f'{wr_name_selected}, session {session_selected}')

    ax2=fig.add_axes([0,-.6,2,.4])
    ax3=fig.add_axes([0,-1.6,2,.8])
    ax4=fig.add_axes([0,-2.6,2,.8])
    ax5 = fig.add_axes([0,-3.6,2,.8])
    
    choice_averaging_window = 10
    choice_kernel = np.ones(choice_averaging_window)/choice_averaging_window
    
    # invoke plot functions   
    filters = {'ignore_rate_max':100,
               'averaging_window':choice_averaging_window}
    
    local_matching = {'calculate_local_matching': True,
                     'sliding_window':100,
                     'matching_window':300,
                     'matching_step':30,
                     'efficiency_type':'ideal'}
    
    df_behaviortrial = extract_trials(plottype = '2lickport',
                                      wr_name = wr_name_selected,
                                      sessions = [session_selected, session_selected],
                                      show_bias_check_trials =  True,
                                      kernel = choice_kernel,
                                      filters = filters,
                                      local_matching = local_matching)
    
    plot_trials(df_behaviortrial,
                ax1,
                ax2,
                plottype = '2lickport',
                wr_name = wr_name_selected,
                sessions = [session_selected, session_selected],
                plot_every_choice= True,
                show_bias_check_trials =  True,
                choice_filter = choice_kernel)
    
    plot_local_efficiency_matching_bias(df_behaviortrial,
                                            ax3)
    
    plot_rt_iti(df_behaviortrial,
                ax4,
                ax5,
                plottype = '2lickport',
                wr_name = wr_name_selected,
                sessions = [session_selected, session_selected],
                show_bias_check_trials =  True,
                kernel = choice_kernel
                )
    
    

def plot_foragingWebGUI_trial(wr_name_selected='HH07', session_selected=17):  # Plot all trial events of one specific session
    #%% 
    fig = plt.figure(figsize=(20,12))
    fig.suptitle(f'{wr_name_selected}, session {session_selected}')
    gs = GridSpec(5, 1, wspace=0.4, hspace=0.4, bottom=0.07, top=0.95, left=0.1, right=0.9) 
    
    ax1 = fig.add_subplot(gs[0:3, :])
    ax2 = fig.add_subplot(gs[3, :])
    ax3 = fig.add_subplot(gs[4, :])
    ax1.get_shared_x_axes().join(ax1, ax2, ax3)
    
    # Plot settings
    plot_setting = {'left lick':'red', 'right lick':'blue'}  
    
    # -- Get event times --
    key_subject_id_session = (experiment.Session() & (lab.WaterRestriction() & 'water_restriction_number="{}"'.format(wr_name_selected)) 
                              & 'session="{}"'.format(session_selected)).fetch1("KEY")
    go_cue_times = (experiment.TrialEvent() & key_subject_id_session & 'trial_event_type="go"').fetch('trial_event_time', order_by='trial').astype(float)
    lick_times = pd.DataFrame((experiment.ActionEvent() & key_subject_id_session).fetch(order_by='trial'))
    
    trial_num = len(go_cue_times)
    all_trial_num = np.arange(1, trial_num+1).tolist()
    all_trial_start = [[-x] for x in go_cue_times]
    all_lick = dict()
    for event_type in plot_setting:
        all_lick[event_type] = []
        for i, trial_start in enumerate(all_trial_start):
            all_lick[event_type].append((lick_times[(lick_times['trial']==i+1) & (lick_times['action_event_type']==event_type)]['action_event_time'].values.astype(float) + trial_start).tolist())
    
    # -- All licking events (Ordered by trials) --
    ax1.plot([0, 0], [0, trial_num], 'k', lw=0.5)   # Aligned by go cue
    ax1.set(ylabel='Trial number', xlim=(-3, 3), xticks=[])

    # Batch plotting to speed up    
    ax1.eventplot(lineoffsets=all_trial_num, positions=all_trial_start, color='k')   # Aligned by go cue
    for event_type in plot_setting:
        ax1.eventplot(lineoffsets=all_trial_num, 
                     positions=all_lick[event_type],
                     color=plot_setting[event_type],
                     linewidth=2)   # Trial start
    
    # -- Histogram of all licks --
    for event_type in plot_setting:
        sns.histplot(np.hstack(all_lick[event_type]), binwidth=0.01, alpha=0.5, 
                     ax=ax2, color=plot_setting[event_type], label=event_type)  # 10-ms window
    
    ymax_tmp = max(ax2.get_ylim())  
    sns.histplot(-go_cue_times, binwidth=0.01, color='k', ax=ax2, label='trial start')  # 10-ms window
    ax2.axvline(x=0, color='k', lw=0.5)
    ax2.set(ylim=(0, ymax_tmp), xticks=[], title='All events') # Fix the ylim of left and right licks
    ax2.legend()
    
    # -- Histogram of reaction time (first lick after go cue) --   
    plot_setting = {'LEFT':'red', 'RIGHT':'blue'}  
    for water_port in plot_setting:
        this_RT = (foraging_analysis.TrialStats() & key_subject_id_session & (experiment.WaterPortChoice() & 'water_port="{}"'.format(water_port))).fetch('reaction_time').astype(float)
        sns.histplot(this_RT, binwidth=0.01, alpha=0.5, 
                     ax=ax3, color=plot_setting[water_port], label=water_port)  # 10-ms window
    ax3.axvline(x=0, color='k', lw=0.5)
    ax3.set(xlabel='Time to Go Cue (s)', title='First lick (reaction time)') # Fix the ylim of left and right licks
    ax3.legend()
    
        