import pandas as pd
from pipeline import lab, experiment, foraging_analysis
import numpy as np
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
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.10, 0.2), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)
        

def extract_trials(plottype = '2lickport',
                   wr_name = 'FOR01',
                   sessions = (5,11),
                   show_bias_check_trials = True,
                   kernel = np.ones(10)/10,
                   filters=None,
                   local_matching = {'calculate_local_matching':False}):
    
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
                        * foraging_analysis.TrialReactionTime                      # Reaction time
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
        if len(p_reward_this):
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
        total_trials_so_far = (foraging_analysis.SessionStats()&'subject_id = {}'.format(subject_id) &'session < {}'.format(session)).fetch('session_total_trial_num')
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
        
        
    #% calculating local matching, bias, reward rate

    kernel = np.ones(movingwindow)/movingwindow
    p1 = np.asarray(np.max([df_behaviortrial['p_reward_right'],df_behaviortrial['p_reward_left']],0),float)
    p0 = np.asarray(np.min([df_behaviortrial['p_reward_right'],df_behaviortrial['p_reward_left']],0),float)
    m_star_greedy = np.floor(np.log(1-p1)/np.log(1-p0))
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
    choice_rate_right= np.convolve(choice_right,kernel,'same')/np.convolve(choice_left+choice_middle,kernel,'same')
    reward_rate_right = np.convolve(reward_rate_right,kernel,'same')/np.convolve(reward_rate_left+reward_rate_middle,kernel,'same')
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
            slope_now, intercept_now = np.polyfit(np.log2(reward_rates_now), np.log2(choice_rates_now), 1)
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
    
        
    blockswitches = np.where(np.diff(df_behaviortrial['session'].values)>0)[0]
    if len(blockswitches)>0:
        for trialnum_now in blockswitches:
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
    
    multicolor_ylabel(ax1,('Delay (s)', ' ITI (s)','Reaction time (ms)'),('k','r','m'),axis='y',size=12)
    
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
    
    blockswitches = np.where(np.diff(df_behaviortrial['session'].values)>0)[0]
    if len(blockswitches)>0:
        for trialnum_now in blockswitches:
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
        ax2.set_xlabel('Trial #')
        if plottype == '3lickport':
            legenda = ['left','right','middle']
        else:
            legenda = ['left','right']
        ax2.legend(legenda,fontsize='small',loc = 'upper right')
        ax2.set_xlim([np.min(df_behaviortrial['trial'])-10,np.max(df_behaviortrial['trial'])+10])    
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
        total_trials_so_far = (foraging_analysis.SessionStats()&'subject_id = {}'.format(subject_id) &'session < {}'.format(session)).fetch('session_total_trial_num')
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
    ax3.set_xlim([np.min(session_switch_trial_nums)-10,np.max(session_switch_trial_nums)+10]) 
    
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
    multicolor_ylabel(ax3,('Efficiency', ' Matching '),('r','k'),axis='y',size=12)
    #%%
    return ax3


def plot_local_efficiency_matching_bias(df_behaviortrial,ax3):
    
    ax3.plot(df_behaviortrial['trial'],df_behaviortrial['local_efficiency'],'k-')     
    blockswitches = np.where(np.diff(df_behaviortrial['session'].values)>0)[0]
    if len(blockswitches)>0:
        for trialnum_now in blockswitches:
            ax3.plot([df_behaviortrial['trial'][trialnum_now],df_behaviortrial['trial'][trialnum_now]],[-.15,1.15],'b--')
     

   #%
    trialnums = df_behaviortrial.loc[~np.isnan(df_behaviortrial['local_matching_slope']),'trial']
    local_matching_slope = df_behaviortrial.loc[~np.isnan(df_behaviortrial['local_matching_slope']),'local_matching_slope']
    local_matching_bias = df_behaviortrial.loc[~np.isnan(df_behaviortrial['local_matching_slope']),'local_matching_bias']
    ax3.plot(trialnums,local_matching_slope,'ro-')
    ax3.set_ylim([-.1,1.1])
    ax33 = ax3.twinx()
    ax33.plot(trialnums,local_matching_bias,'yo-')
    ax33.set_ylim(np.asarray([-1.1,1.1])*np.nanmin([np.nanmax(np.abs(local_matching_bias)),4]))
    ax33.set_ylabel('Bias',color='y')
    ax33.spines["right"].set_color("yellow")
    ax3.set_xlim([np.min(df_behaviortrial['trial'])-10,np.max(df_behaviortrial['trial'])+10]) 
    #ax33.tick_params(axis='y', colors='yellow')
    multicolor_ylabel(ax3,('Efficiency', ' Matching '),('r','k'),axis='y',size=12)
    #%%
    return ax3