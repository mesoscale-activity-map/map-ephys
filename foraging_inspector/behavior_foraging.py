import datajoint as dj
# import pipeline.lab as lab
# import pipeline.experiment as experiment
# from pipeline.pipeline_tools import get_schema_name
from pipeline import (lab, experiment, get_schema_name)
schema = dj.schema(get_schema_name('behavior_foraging'),locals())
import numpy as np
import pandas as pd
import math
dj.config["enable_python_native_blobs"] = True
#%%
bootstrapnum = 100
minimum_trial_per_block = 30


#%%
@schema
class TrialReactionTime(dj.Computed):
    definition = """
    -> experiment.BehaviorTrial
    ---
    reaction_time = null : decimal(8,4) # reaction time in seconds (first lick relative to go cue) [-1 in case of ignore trials]
    """
    # Foraging sessions only
    key_source = experiment.BehaviorTrial & 'task LIKE "foraging%"'

    def make(self, key):
        gocue_time = (experiment.TrialEvent & key & 'trial_event_type = "go"').fetch1('trial_event_time')
        q_reaction_time = experiment.BehaviorTrial.aggr(experiment.ActionEvent & key
                                                        & 'action_event_type LIKE "%lick"'
                                                        & 'action_event_time > {}'.format(gocue_time),
                                                        reaction_time='min(action_event_time)')
        if q_reaction_time:
            key['reaction_time'] = q_reaction_time.fetch1('reaction_time') - gocue_time

        self.insert1(key)
            
@schema # TODO remove bias check?
class BlockStats(dj.Computed):
    definition = """ # All blocks including bias check
    -> experiment.SessionBlock
    ---
    block_trial_num : int # number of trials in block
    block_ignore_num : int # number of ignores
    block_reward_rate = null: decimal(8,4) # hits / (hits + misses)
    """

    def make(self, key):
        
        q_block_trials = experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key

        block_stats = {'block_trial_num': len((experiment.SessionBlock.BlockTrial & key)),
                       'block_ignore_num': len(q_block_trials & 'outcome = "ignore"')}
        try:
            block_stats['block_reward_rate'] = len(q_block_trials & 'outcome = "hit"') / len(q_block_trials & 'outcome in ("hit", "miss")')
        except:
            pass
        self.insert1({**key, **block_stats})
 
    
@schema #remove bias check trials from statistics # 03/25/20 NW added nobiascheck terms for hit, miss and ignore trial num
class SessionStats(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_total_trial_num = null : int #number of trials
    session_block_num = null : int #number of blocks, including bias check
    session_hit_num = null : int #number of hits
    session_miss_num = null : int #number of misses
    session_ignore_num = null : int #number of ignores
    session_autowater_num = null : int #number of trials with autowaters
    session_length = null : decimal(10, 4) #length of the session in seconds
    """
    # Foraging sessions only
    key_source = experiment.Session & (experiment.BehaviorTrial & 'task LIKE "foraging%"')

    def make(self, key):
    
        session_stats = {'session_total_trial_num': len(experiment.SessionTrial & key),
                'session_block_num': len(experiment.SessionBlock & key),
                'session_hit_num': len(experiment.BehaviorTrial & key & 'outcome = "hit"'),
                'session_miss_num': len(experiment.BehaviorTrial & key & 'outcome = "miss"'),
                'session_ignore_num': len(experiment.BehaviorTrial & key & 'outcome = "ignore"'),
                'session_autowater_num': len(experiment.TrialNote & key & 'trial_note_type = "autowater"')}
        
        if session_stats['session_total_trial_num'] > 0:
            session_stats['session_length'] = float(((experiment.SessionTrial() & key).fetch('stop_time')).max())
        else:
            session_stats['session_length'] = 0

        self.insert1({**key, **session_stats})
            
@schema
class SessionTaskProtocol(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_task_protocol : tinyint # the number of the dominant task protocol in the session
    session_real_foraging : bool # True if it is real foraging, false in case of pretraining
    """
    
    # Foraging sessions only
    key_source = experiment.Session & (experiment.BehaviorTrial & 'task LIKE "foraging%"')
    
    def make(self, key):
        task_protocols = (experiment.BehaviorTrial & key).fetch('task_protocol')

        is_real_foraging = bool(experiment.SessionBlock.WaterPortRewardProbability & key
                                & 'reward_probability > 0 and reward_probability < 1')

        self.insert1({**key,
                      'session_task_protocol': np.median(task_protocols),
                      'session_real_foraging': is_real_foraging})

        
@schema
class BlockFraction(dj.Computed):
    definition = """ # Block reward and choice fraction without bias check
    -> experiment.SessionBlock
    ---
    block_length: smallint #
    block_reward_per_trial: float   # This is actually "block reward rate", same as BlockStats.block_reward_rate, except only for real_training and > min_trial_length
    """

    class WaterPortFraction(dj.Part):
        definition = """
        -> master
        -> experiment.WaterPort
        ---
        block_reward_fraction=null                  : float     # lickport reward fraction from all trials
        block_choice_fraction=null                  : float
        """

    @property
    def key_source(self):
        """
        Only process the blocks with:
         1. trial-count > minimum_trial_per_block
         2. is_real_training only
        """
        # trial-count > minimum_trial_per_block
        ks_tr_count = experiment.SessionBlock.aggr(experiment.SessionBlock.BlockTrial, tr_count='count(*)') & 'tr_count > {}'.format(minimum_trial_per_block)
        # is_real_training only
        ks_real_training = ks_tr_count - (experiment.SessionBlock.WaterPortRewardProbability & 'reward_probability >= 1')
        return ks_real_training

    def make(self, key):
        # To skip bias check trial 04/02/20 NW
        q_block_trial = experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key

        block_rw = q_block_trial.proj(reward='outcome = "hit"').fetch('reward', order_by='trial')
        block_choice = (experiment.WaterPortChoice * q_block_trial).proj(
            non_null_wp='water_port IS NOT NULL').fetch('non_null_wp', order_by='trial')

        trialnum = len(block_rw)

        # ---- whole-block fraction ----
        block_reward_fraction = dict(
            block_length=trialnum,
            # block_reward_per_trial=block_rw.mean(),
            block_reward_per_trial=block_rw.sum()/block_choice.sum(),   # Exclude ignore trials
            )

        self.insert1({**key, **block_reward_fraction})

        # ---- water-port fraction ----
        wp_frac = {}
        for water_port in experiment.WaterPort.fetch('water_port'):
            # --- reward fraction ---
            rw = (experiment.WaterPortChoice * q_block_trial).proj(
                reward='water_port = "{}" AND outcome = "hit"'.format(water_port)).fetch('reward', order_by='trial').astype(float)
            wp_frac[water_port] = {
                'block_reward_fraction': rw.sum() / block_rw.sum() if block_rw.sum() else np.nan,
              }

            # --- choice fraction ---
            choice = (experiment.WaterPortChoice * q_block_trial).proj(
                choice='water_port = "{}"'.format(water_port)).fetch('choice', order_by='trial').astype(float)

            wp_frac[water_port].update(**{
                'block_choice_fraction': np.nansum(choice) / block_choice.sum() if block_choice.sum() else np.nan,
               })

        self.WaterPortFraction.insert([{**key, 'water_port': wp, **wp_data} for wp, wp_data in wp_frac.items()])
        

@schema
class SessionMatchBias(dj.Computed): # bias check removed, 
    definition = """
    -> experiment.Session
    ---
    block_r_r_ratio = null: longblob # right lickport reward ratio
    block_r_c_ratio = null: longblob # right lickport choice ratio
    match_idx_r = null: decimal(8,4) # slope of log ratio R from whole blocks
    bias_r = null: decimal(8,4) # intercept of log ratio R from whole blocks
    """
    def make(self, key):
        #key = {'subject_id': 453478, 'session': 13}
       
        bias_check_block = pd.DataFrame(SessionStats() & key)        
        # df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
        
        block_R_RewardRatio = np.ones(bias_check_block['session_block_num'])*np.nan
        block_R_ChoiceRatio = np.ones(bias_check_block['session_block_num'])*np.nan
        df_block = pd.DataFrame((experiment.SessionBlock() & key))
        
        print(key)
        if len(df_block)> 0:
            for x in range(len(df_block['block'])):
                print('block '+str(x))
                df_choices = pd.DataFrame((experiment.BehaviorTrial()*experiment.SessionBlock()) & key & 'block ='+str(x+1))
                realtraining = (df_choices['p_reward_left']<1) & (df_choices['p_reward_right']<1) & ((df_choices['p_reward_middle']<1) | df_choices['p_reward_middle'].isnull())
                if realtraining[0] == True:
                    BlockRewardFraction = pd.DataFrame(BlockRewardFractionNoBiasCheck() & key & 'block ='+str(x+1))
                    BlockChoiceFraction = pd.DataFrame(BlockChoiceFractionNoBiasCheck() & key & 'block ='+str(x+1))
                    
                    # if df_behaviortrial['task_protocol'][0] == 100: # if it's 2lp task, then only calculate R log ratio
                    #     try:
                    #         block_R_RewardRatio[x] = np.log2(float(BlockRewardFraction['block_reward_fraction_right']/BlockRewardFraction['block_reward_fraction_left']))
                    #     except:
                    #         pass
                    #     try:
                    #         block_R_ChoiceRatio[x] = np.log2(float(BlockChoiceFraction['block_choice_fraction_right']/BlockChoiceFraction['block_choice_fraction_left']))
                    #     except:
                    #         pass
                    # elif df_behaviortrial['task_protocol'][0] == 101: # if ti's 3lp task, then calculate R, L and M
                    
                    # Middle will naturally be 0 if it's a 2lp task
                    try:
                        block_R_RewardRatio[x] = np.log2(float(BlockRewardFraction['block_reward_fraction_right']/(BlockRewardFraction['block_reward_fraction_left']+BlockRewardFraction['block_reward_fraction_middle'])))
                    except:
                        pass
                    
                    try:
                        block_R_ChoiceRatio[x] = np.log2(float(BlockChoiceFraction['block_choice_fraction_right']/(BlockChoiceFraction['block_choice_fraction_left']+BlockChoiceFraction['block_choice_fraction_middle'])))
                    except:
                        pass
        
        block_R_RewardRatio = block_R_RewardRatio[~np.isinf(block_R_RewardRatio)]
        block_R_ChoiceRatio = block_R_ChoiceRatio[~np.isinf(block_R_ChoiceRatio)]

        try:
            match_idx_r, bias_r = draw_bs_pairs_linreg(block_R_RewardRatio, block_R_ChoiceRatio, size=bootstrapnum)
        except:
            pass
        
        key['match_idx_r'] = np.nanmean(match_idx_r)   
        key['bias_r'] = np.nanmean(bias_r) 
        
        key['block_r_r_ratio'] = block_R_RewardRatio
        key['block_r_c_ratio'] = block_R_ChoiceRatio
        
        self.insert1(key,skip_duplicates=True)     
        
        
        
@schema
class BlockEfficiency(dj.Computed): # bias check excluded
    definition = """
    -> experiment.SessionBlock
    ---
    block_ideal_phat_greedy = null: decimal(8,4) # denominator = Ideal-pHat-greedy
    regret_ideal_phat_greedy = null: decimal(8,4) # Ideal-pHat-greedy - reward collected
    
    """
    def make(self, key):
        
        # key = {'subject_id': 447921, 'session': 12, 'block': 17}
        keytoinsert = key
        print(keytoinsert)
        df_choices = pd.DataFrame((experiment.BehaviorTrial()*experiment.SessionBlock()) & key)
        realtraining = (df_choices['p_reward_left']<1) & (df_choices['p_reward_right']<1) & ((df_choices['p_reward_middle']<1) | df_choices['p_reward_middle'].isnull())
        
        if realtraining[0] == True and len(df_choices) > minimum_trial_per_block:
            p_reward_left,p_reward_right,p_reward_middle = (experiment.SessionBlock() & keytoinsert).fetch('p_reward_left','p_reward_right','p_reward_middle')
            p_reward_left = p_reward_left.astype(float)
            p_reward_right = p_reward_right.astype(float)
            p_reward_middle = p_reward_middle.astype(float)
        
            reward_available = pd.DataFrame((experiment.BehaviorTrial()*experiment.SessionBlock()* experiment.TrialAvailableReward() & keytoinsert))
        
            if reward_available['task_protocol'][0] == 101:
                pass    # Not defined for 3lp                
            else:
                p1 = np.nanmax([p_reward_left,p_reward_right])
                p0 = np.nanmin([p_reward_left,p_reward_right])
                if p0 != 0 :
                    m_star_greedy = math.floor(math.log(1-p1)/math.log(1-p0))
                    p_star_greedy = p1 + (1-(1-p0)**(m_star_greedy+1)-p1**2)/(m_star_greedy+1)
        
            BlockRewardRatio = pd.DataFrame(BlockRewardFractionNoBiasCheck & key)

            try:
                keytoinsert['block_ideal_phat_greedy'] = float(BlockRewardRatio['block_reward_per_trial'])/p_star_greedy
            except:
                pass 
          
        self.insert1(keytoinsert,skip_duplicates=True)
        
def draw_bs_pairs_linreg(x, y, size=1): 
    """Perform pairs bootstrap for linear regression."""#from serhan aya
    try:
        # Set up array of indices to sample from: inds
        inds = np.arange(len(x))
    
        # Initialize replicates: bs_slope_reps, bs_intercept_reps
        bs_slope_reps = np.empty(size)
        bs_intercept_reps = np.empty(shape=size)
    
        # Generate replicates
        for i in range(size):
            bs_inds = np.random.choice(inds, size=len(inds)) # sampling the indices (1d array requirement)
            bs_x, bs_y = x[bs_inds], y[bs_inds]
            bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
    
        return bs_slope_reps, bs_intercept_reps
    except:
        return np.nan, np.nan
