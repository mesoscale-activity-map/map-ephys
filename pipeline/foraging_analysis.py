import datajoint as dj
from pipeline import (experiment, get_schema_name)
schema = dj.schema(get_schema_name('foraging_analysis'),locals())
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
    session_early_lick_ratio = null: decimal(5,4)  # early lick ratio
    session_autowater_num = null : int #number of trials with autowaters
    session_length = null : decimal(10, 4) #length of the session in seconds
    session_pure_choices_num = null : int # Number of pure choices (excluding auto water)
    
    session_foraging_eff_optimal = null: decimal(5,4)   # Session-wise foraging efficiency (optimal; average sense)
    session_foraging_eff_optimal_random_seed = null: decimal(5,4)   # Session-wise foraging efficiency (optimal; random seed) #!!!
    
    session_mean_reward_sum = null: decimal(4,3)  # Median of sum of reward prob
    session_mean_reward_contrast = null: float  # Median of reward prob ratio
    session_effective_block_trans_num = null: int  # Number of effective block transitions  #!!!
    """
    # Foraging sessions only
    key_source = experiment.Session & (experiment.BehaviorTrial & 'task LIKE "foraging%"')

    def make(self, key):
        # import pdb; pdb.set_trace()
        q_all_trial = experiment.SessionTrial & key
        q_block = experiment.SessionBlock & key
        q_hit = experiment.BehaviorTrial & key & 'outcome = "hit"'
        q_miss = experiment.BehaviorTrial & key & 'outcome = "miss"'
        q_auto_water = experiment.TrialNote & key & 'trial_note_type = "autowater"'
        q_actual_finished = q_hit.proj() + q_miss.proj()  - q_auto_water.proj()   # Real finished trial = 'hit' or 'miss' but not 'autowater'
        
        session_stats = {'session_total_trial_num': len(q_all_trial),
                'session_block_num': len(q_block),
                'session_hit_num': len(q_hit),
                'session_miss_num': len(q_miss),
                'session_ignore_num': len(experiment.BehaviorTrial & key & 'outcome = "ignore"'),
                'session_early_lick_ratio': len(experiment.BehaviorTrial & key & 'early_lick="early"') / (len(q_hit) + len(q_miss)),
                'session_autowater_num': len(q_auto_water),
                'session_pure_choices_num': len(q_actual_finished)}
        
        if session_stats['session_total_trial_num'] > 0:
            session_stats['session_length'] = float(((experiment.SessionTrial() & key).fetch('stop_time')).max())
        else:
            session_stats['session_length'] = 0
            
        # -- Session-wise foraging efficiency (2lp only) --
        if len(experiment.BehaviorTrial & key & 'task="foraging"'):
            # Get reward rate (hit but not autowater) / (hit but not autowater + miss but not autowater)
            q_pure_hit_num = q_hit.proj() - q_auto_water.proj()
            reward_rate = len(q_pure_hit_num) / len(q_actual_finished)
            
            q_actual_finished_reward_prob = (experiment.SessionTrial * experiment.SessionBlock.BlockTrial  # Session-block-trial
                                           * experiment.SessionBlock.WaterPortRewardProbability  # Block-trial-p_reward
                                           & q_actual_finished)  # Select finished trials
                                
            # Get reward probability (only pure finished trials)
            p_Ls = (q_actual_finished_reward_prob & 'water_port="left"').fetch(
                'reward_probability').astype(float)
            p_Rs = (q_actual_finished_reward_prob & 'water_port="right"').fetch(
                'reward_probability').astype(float)
            
            # Recover actual random numbers
            random_number_Ls = np.empty(len(q_all_trial))
            random_number_Ls[:] = np.nan
            random_number_Rs = random_number_Ls.copy()
            
            rand_seed_starts = (experiment.TrialNote()  & key & 'trial_note_type="random_seed_start"').fetch('trial', 'trial_note', order_by='trial')
            
            for start_idx, start_seed in zip(rand_seed_starts[0], rand_seed_starts[1]):  # For each pybpod session
                # Must be exactly the same as the pybpod protocol 
                # https://github.com/hanhou/Foraging-Pybpod/blob/5e19e1d227657ed19e27c6e1221495e9f180c323/pybpod_protocols/Foraging_baptize_by_fire_new_lickport_retraction.py#L478
                np.random.seed(int(start_seed))
                random_number_L_this = np.random.uniform(0.,1.,2000).tolist()
                random_number_R_this = np.random.uniform(0.,1.,2000).tolist()
                
                # Fill in random numbers
                random_number_Ls[start_idx - 1 :] = random_number_L_this[: len(random_number_Ls) - start_idx + 1]
                random_number_Rs[start_idx - 1 :] = random_number_R_this[: len(random_number_Rs) - start_idx + 1]
                
            # Select finished trials
            actual_finished_idx = q_actual_finished.fetch('trial', order_by='trial')-1
            random_number_Ls = random_number_Ls[actual_finished_idx]
            random_number_Rs = random_number_Rs[actual_finished_idx]
            
            # Get foraging efficiency
            for_eff_optimal, for_eff_optimal_random_seed = foraging_eff(reward_rate, p_Ls, p_Rs, random_number_Ls, random_number_Rs)
             
            # Reward schedule stats
            if (SessionTaskProtocol & key).fetch1('session_real_foraging'):   # Real foraging
                p_contrast = np.max([p_Ls, p_Rs], axis=0) / np.min([p_Ls, p_Rs], axis=0)
                p_contrast[np.isinf(p_contrast)] = np.nan  # A arbitrary huge number
                p_contrast_mean = np.nanmean(p_contrast)
            else:
                p_contrast_mean = 100
                
            session_stats.update(session_foraging_eff_optimal = for_eff_optimal,
                                 session_foraging_eff_optimal_random_seed = for_eff_optimal_random_seed,
                                 session_mean_reward_sum = np.nanmean(p_Ls + p_Rs), 
                                 session_mean_reward_contrast = p_contrast_mean)
            
            
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
class SessionMatching(dj.Computed):  # bias check removed,
    definition = """# Blockwise matching of a session
    -> SessionStats
    """

    class WaterPortMatching(dj.Part):
        definition = """  # reward and choice ratio of this water-port w.r.t the sum of the other water-ports
        -> master
        -> experiment.WaterPort
        ---
        reward_ratio=null                : longblob      # lickport reward ratio (this : sum of others)
        choice_ratio=null                : longblob      # lickport choice ratio (this : sum of others)       
        match_idx=null                      : decimal(8,4)  # slope of log ratio
        bias=null                           : decimal(8,4)  # intercept of log ratio
        """

    def make(self, key):
        q_block_fraction = BlockFraction.WaterPortFraction & key

        ratio_attrs = ['reward_ratio', 
                        'choice_ratio']
        
        session_matching = {}
        
        # ratio = this / others = fraction / (1-fraction)
        q_block_ratio = q_block_fraction.proj(reward_ratio='block_reward_fraction/(1-block_reward_fraction)',
                                               choice_ratio='block_choice_fraction/(1-block_choice_fraction)')
        
        for water_port in experiment.WaterPort.fetch('water_port'):
            # # ---- compute the reward and choice fraction, across blocks ----
            # # query for this port
            # this_port = (q_block_fraction & 'water_port = "{}"'.format(water_port))
            
            # # query for other ports, take the sum of the reward and choice fraction of the other ports
            # other_ports = BlockFraction.aggr(
            #     q_block_fraction & 'water_port != "{}"'.format(water_port),
            #     rw_sum='sum(block_reward_fraction)',
            #     choice_sum='sum(block_choice_fraction)')
            
            # # merge and compute the fraction of this port over the sum of the other ports
            # wp_block_ratio = (this_port * other_ports).proj(
            #     reward_ratio='block_reward_fraction / rw_sum',
            #     choice_ratio='block_choice_fraction / choice_sum').fetch(
            #     *ratio_attrs, order_by='block')
                    
            wp_block_ratio = (q_block_ratio & 'water_port = "{}"'.format(water_port)).fetch(
                              *ratio_attrs, order_by='block')
            
            # taking the log2
            # session_matching[water_port] = {attr: np.log2(attr_value.astype(float))
            #                             for attr, attr_value in zip(ratio_attrs, wp_block_ratio)}
            session_matching[water_port] = {}
            for attr, attr_value in zip(ratio_attrs, wp_block_ratio):
                attr_value = attr_value.astype(float)
                attr_value[(attr_value==0)] = np.nan
                session_matching[water_port][attr] = np.log2(attr_value)

            # ---- compute the match index and bias ----
            # for tertile_suffix in ('', '_first_tertile', '_second_tertile', '_third_tertile'):
            tertile_suffix = ''
            reward_name = 'reward_ratio' + tertile_suffix
            choice_name = 'choice_ratio' + tertile_suffix
            # Ignore those with all NaNs or all Infs
            if (np.isfinite(session_matching[water_port][reward_name]).any()
                    and np.isfinite(session_matching[water_port][choice_name]).any()):
                match_idx, bias = draw_bs_pairs_linreg(
                    session_matching[water_port][reward_name],
                    session_matching[water_port][choice_name], size=bootstrapnum)
                session_matching[water_port]['match_idx' + tertile_suffix] = np.nanmean(match_idx)
                session_matching[water_port]['bias' + tertile_suffix] = np.nanmean(bias)

            else:
                session_matching[water_port].pop(reward_name)
                session_matching[water_port].pop(choice_name)

        # ---- Insert ----
        self.insert1(key)
        # can't do list comprehension for batch insert because "session_matching" may have different fields
        for k, v in session_matching.items():
            self.WaterPortMatching.insert1({**key, 'water_port': k, **v})
            
@schema
class BlockEfficiency(dj.Computed):  # bias check excluded
    definition = """
    -> BlockFraction
    ---
    block_effi_one_p_reward: float                       # denominator = max of the reward assigned probability (no baiting)
    block_effi_sum_p_reward: float                       # denominator = sum of the reward assigned probability (no baiting)
    block_effi_one_a_reward=null: float                  # denominator = max of the reward assigned probability + baiting)
    block_effi_sum_a_reward=null: float                  # denominator = sum of the reward assigned probability + baiting)
    block_ideal_phat_greedy=null: float                  # denominator = Ideal-pHat-greedy
    regret_ideal_phat_greedy=null: float                 # Ideal-pHat-greedy - reward collected
    """

    def make(self, key):
        
        # BlockFraction does not use all experiment.SessionBlock primary keys (bias check excluded)
        if len(BlockFraction & key) == 0:  
            return
        
        block_fraction = (BlockFraction & key).fetch1()
        
        water_ports, rewards = (experiment.SessionBlock.WaterPortRewardProbability & key).fetch(
            'water_port', 'reward_probability')
        rewards = rewards.astype(float)
        max_prob_reward = np.nanmax(rewards)
        sum_prob_reward = np.nansum(rewards)

        q_block_trial = experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key

        reward_available = pd.DataFrame({wp: (q_block_trial * experiment.TrialAvailableReward
                                              & {'water_port': wp}).fetch('reward_available', order_by='trial')
                                         for wp in water_ports})
        max_reward_available = reward_available.max(axis=1)
        sum_reward_available = reward_available.sum(axis=1)

        block_efficiency_data = dict(
            block_effi_one_p_reward=block_fraction['block_reward_per_trial'] / max_prob_reward,
            block_effi_sum_p_reward=block_fraction['block_reward_per_trial'] / sum_prob_reward,
            block_effi_one_a_reward=(block_fraction['block_reward_per_trial'] / max_reward_available.mean()
                                     if max_reward_available.mean() else np.nan),
            block_effi_sum_a_reward=(block_fraction['block_reward_per_trial'] / sum_reward_available.mean()
                                     if sum_reward_available.mean() else np.nan))

        # Ideal-pHat-greedy  - only for blocks containing "left" and "right" port only
        if not len(np.setdiff1d(['right', 'left'], water_ports)):
            p1 = np.nanmax(rewards)
            p0 = np.nanmin(rewards)
            if p0 != 0:
                m_star_greedy = math.floor(math.log(1 - p1) / math.log(1 - p0))
                p_star_greedy = p1 + (1 - (1 - p0) ** (m_star_greedy + 1) - p1 ** 2) / (m_star_greedy + 1)
            else:
                p_star_greedy = p1

            block_efficiency_data.update(
                block_ideal_phat_greedy=block_fraction['block_reward_per_trial'] / p_star_greedy,
                regret_ideal_phat_greedy=p_star_greedy - block_fraction['block_reward_per_trial'])

        self.insert1({**key, **block_efficiency_data})


   
# ====================== HELPER FUNCTIONS ==========================     
   
def draw_bs_pairs_linreg(x, y, size=1): 
    """Perform pairs bootstrap for linear regression."""#from serhan aya
    # Get rid of infs/nans
    idx = np.isfinite(x) & np.isfinite(y)
    x = x[idx]
    y = y[idx]

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
    

def foraging_eff(reward_rate, p_Ls, p_Rs, random_number_L=None, random_number_R=None):  # Calculate foraging efficiency (only for 2lp)
        
        # --- Optimal-aver (use optimal expectation as 100% efficiency) ---
        p_stars = np.zeros_like(p_Ls)
        for i, (p_L, p_R) in enumerate(zip(p_Ls, p_Rs)):   # Sum over all ps 
            p_max = np.max([p_L, p_R])
            p_min = np.min([p_L, p_R])
            if p_min > 0:
                m_star = np.floor(np.log(1-p_max)/np.log(1-p_min))
                p_stars[i] = p_max + (1-(1-p_min)**(m_star + 1)-p_max**2)/(m_star+1)
            else:
                p_stars[i] = p_max
        for_eff_optimal = reward_rate / np.nanmean(p_stars)
        
        if random_number_L is None:
            return for_eff_optimal, np.nan
            
        # --- Optimal-actual (uses the actual random numbers by simulation)
        block_trans = np.where(np.diff(np.hstack([np.inf, p_Ls, np.inf])))[0].tolist()
        reward_refills = [p_Ls >= random_number_L, p_Rs >= random_number_R]
        reward_optimal_random_seed = 0
        # Generate optimal choice pattern
        for b_start, b_end in zip(block_trans[:-1], block_trans[1:]):
            p_max = np.max([p_Ls[b_start], p_Rs[b_start]])
            p_min = np.min([p_Ls[b_start], p_Rs[b_start]])
            side_max = np.argmax([p_Ls[b_start], p_Rs[b_start]])
            
            reward_refill = np.vstack([reward_refills[1 - side_max][b_start:b_end], 
                             reward_refills[side_max][b_start:b_end]]).astype(int)  # Max = 1, Min = 0
            
            # Get choice pattern
            if p_min > 0:   
                m_star = np.floor(np.log(1-p_max)/np.log(1-p_min))
                this_choice = np.array((([1]*int(m_star)+[0]) * (1+int((b_end-b_start)/(m_star+1)))) [:b_end-b_start])
            else:
                this_choice = np.array([1] * (b_end-b_start))
                
            # Do simulation
            reward_remain = [0,0]
            for t in range(b_end - b_start):
                reward_available = reward_remain | reward_refill[:, t]
                reward_optimal_random_seed += reward_available[this_choice[t]]
                reward_remain = reward_available.copy()
                reward_remain[this_choice[t]] = 0
            
            if reward_optimal_random_seed:                
                for_eff_optimal_random_seed = reward_rate / (reward_optimal_random_seed / len(p_Ls))
            else:
                for_eff_optimal_random_seed = np.nan
        
        return for_eff_optimal, for_eff_optimal_random_seed
