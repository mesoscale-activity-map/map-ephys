# =============================================================================
#  Models for fitting behavioral data
# =============================================================================
# == Simplified from foraging_testbed_models.py
#
# = Fitting types =
#   1. Special foragers
#       1). 'LossCounting': switch to another option when loss count exceeds a threshold drawn from Gaussian [from Shahidi 2019]
#           - 3.1: loss_count_threshold = inf --> Always One Side
#           - 3.2: loss_count_threshold = 1 --> win-stay-lose-switch
#           - 3.3: loss_count_threshold = 0 --> Always switch
#
#   2. NLP-like foragers
#       1). 'LNP_softmax' (absorbs 'Corrado2005', 'Sugrue2004' and 'Iigaya2019', forget about fraction income):
#               income  ->  2-exp filter  ->  softmax ( = diff + sigmoid)  -> epsilon-Poisson (epsilon = 0 in their paper; has the same effect as tau_long??)
#
#   3. RL-like foragers
#       1). 'RW1972_epsi':  return  ->   exp filter    -> epsilon-greedy  (epsilon > 0 is essential)
#          1.1). 'RW1972_softmax':  return  ->   exp filter   ->  softmax
#       2). 'Bari2019':        return/income  ->   exp filter (both forgetting)   -> softmax     -> epsilon-Poisson (epsilon = 0 in their paper, no necessary)
#       3). 'Hattori2019':     return/income  ->   exp filter (choice-dependent forgetting, reward-dependent step_size)  -> softmax  -> epsilon-Poisson (epsilon = 0 in their paper; no necessary)
#
#   4. Choice kernel
#      Could be added to: 'RW1972_softmax_CK', 'LNP_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK'
#
#   5. Network models (abstracted from Ulises' network models)
#       1). 'CANN_iti': values decay in ITI
#       2). 'Synaptic' 
#       3). 'Synaptic_W>0': positive W
#
#
#  = Other types that can simulate but not fit =
#   1. 'Random'
#   2. 'IdealpHatGreedy'
#   3. 'pMatching'
#
# Feb 2020, Han Hou (houhan@gmail.com) @ Janelia
# Svoboda & Li lab
# =============================================================================


import numpy as np
from scipy.stats import norm
from .util import softmax, choose_ps

LEFT = 0
RIGHT = 1

global_block_size_mean = 80
global_block_size_sd = 20


class BanditModel:
    '''
    Foragers that can simulate and fit bandit models
    '''

    # @K_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations

    def __init__(self, forager=None, K_arm=2, n_trials=1000, if_baited=True,

                 epsilon=None,               # For 'RW1972_epsi'
                 # For 'LNP_softmax', 'RW1972_softmax', 'Bari2019', 'Hattori2019'
                 softmax_temperature=None,

                 # Bias terms, K-1 degrees of freedom, with constraints:
                 # 1. for those involve random:  b_K = 0 - sum_1toK-1_(b_k), -1/K < b_k < (K-1)/K. cp_k = cp + b_k (for pMatching, may be truncated)
                 # 2. for those involve softmax: b_K = 0, no constraint. cp_k = exp(Q/sigma + b_k) / sum(Q/sigma + b_k). Putting b_k outside /sigma to make it comparable across different softmax_temperatures
                 biasL=0,  # For K = 2.
                 biasR=0,  # Only for K = 3

                 # For 'LNP_softmax', up to two taus
                 tau1=None,
                 tau2=None,
                 w_tau1=None,

                 # Choice kernel
                 choice_step_size=None,
                 choice_softmax_temperature=None,

                 # For 'RW1972_epsi','RW1972_softmax','Bari2019', 'Hattori2019'
                 learn_rate=None,  # For RW and Bari
                 learn_rate_rew=None,     # For Hattori
                 learn_rate_unrew=None,     # For Hattori
                 # 'RW1972_xxx' (= 0)， 'Bari2019' (= 1-Zeta)， 'Hattori2019' ( = unchosen_forget_rate).
                 forget_rate=None,

                 # For 'LossCounting' [from Shahidi 2019]
                 loss_count_threshold_mean=None,
                 loss_count_threshold_std=0,

                 # If true, use the same random seed for generating p_reward!!
                 p_reward_seed_override='',
                 p_reward_sum=0.45,   # Gain of reward. Default = 0.45
                 p_reward_pairs=None,  # Full control of reward prob

                 # !! Important for predictive fitting !!
                 # If not None, calculate predictive_choice_probs(t) based on fit_choice_history(0:t-1) and fit_reward_history(0:t-1) for negLL calculation.
                 fit_choice_history=None,
                 fit_reward_history=None,
                 fit_iti=None,  # ITI [t] --> ITI between t and t + 1

                 # For CANN
                 tau_cann=None,
                 
                 # For synaptic network
                 rho = None,
                 I0 = None,
                 ):

        self.forager = forager
        self.if_baited = if_baited
        self.epsilon = epsilon
        self.softmax_temperature = softmax_temperature
        self.loss_count_threshold_mean = loss_count_threshold_mean
        self.loss_count_threshold_std = loss_count_threshold_std
        self.p_reward_seed_override = p_reward_seed_override
        self.p_reward_sum = p_reward_sum
        self.p_reward_pairs = p_reward_pairs

        self.fit_choice_history = fit_choice_history
        self.fit_reward_history = fit_reward_history
        self.iti = fit_iti
        # In some cases we just need fit_c to fit the model
        self.if_fit_mode = self.fit_choice_history is not None

        if self.if_fit_mode:
            self.K, self.n_trials = np.shape(
                fit_reward_history)  # Use the targeted histories

        else:  # Backward compatibility
            self.K = K_arm
            self.n_trials = n_trials

        # =============================================================================
        #   Parameter check and prepration
        # =============================================================================

        # -- Bias terms --
        # K-1 degrees of freedom, with constraints:
        # 1. for those involve random:  sum_(b_k) = 0, -1/K < b_k < (K-1)/K. cp_k = cp + b_k (for pMatching, may be truncated)
        if forager in ['Random', 'pMatching', 'RW1972_epsi']:
            if self.K == 2:
                self.bias_terms = np.array(
                    [biasL, -biasL])  # Relative to right
            elif self.K == 3:
                self.bias_terms = np.array(
                    [biasL, -(biasL + biasR), biasR])  # Relative to middle
            # Constraints (no need)
            # assert np.all(-1/self.K <= self.bias_terms) and np.all(self.bias_terms <= (self.K - 1)/self.K), self.bias_terms

        # 2. for those involve softmax: b_undefined = 0, no constraint. cp_k = exp(Q/sigma + b_i) / sum(Q/sigma + b_i). Putting b_i outside /sigma to make it comparable across different softmax_temperatures
        elif forager in ['RW1972_softmax', 'LNP_softmax', 'Bari2019', 'Hattori2019',
                         'RW1972_softmax_CK', 'LNP_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK',
                         'CANN', 'Synaptic', 'Synaptic_W>0']:
            if self.K == 2:
                self.bias_terms = np.array([biasL, 0])  # Relative to right
            elif self.K == 3:
                self.bias_terms = np.array(
                    [biasL, 0, biasR])  # Relative to middle
            # No constraints

        # -- Forager-dependent --
        if 'LNP_softmax' in forager:
            assert all(x is not None for x in (tau1, softmax_temperature))
            if tau2 == None:  # Only one tau ('Sugrue2004')
                self.taus = [tau1]
                self.w_taus = [1]
            else:                           # 'Corrado2005'
                self.taus = [tau1, tau2]
                self.w_taus = [w_tau1, 1 - w_tau1]

        elif 'RW1972' in forager:
            assert all(x is not None for x in (learn_rate,))
            # RW1972 has the same learning rate for rewarded / unrewarded trials
            self.learn_rates = [learn_rate, learn_rate]
            self.forget_rates = [0, 0]   # RW1972 does not forget

        elif 'Bari2019' in forager:
            assert all(x is not None for x in (learn_rate, forget_rate))
            # Bari2019 also has the same learning rate for rewarded / unrewarded trials
            self.learn_rates = [learn_rate, learn_rate]
            self.forget_rates = [forget_rate, forget_rate]

        elif 'Hattori2019' in forager:
            assert all(x is not None for x in (
                learn_rate_rew, learn_rate_unrew))
            if forget_rate is None:
                # Allow Hattori2019 to not have forget_rate. In that case, it is an extension of RW1972.
                forget_rate = 0

            # 0: unrewarded, 1: rewarded
            self.learn_rates = [learn_rate_unrew, learn_rate_rew]
            self.forget_rates = [forget_rate, 0]   # 0: unchosen, 1: chosen

        elif 'CANN' in forager:
            assert all(x is not None for x in (
                learn_rate, tau_cann, softmax_temperature))
            self.tau_cann = tau_cann
            self.learn_rates = [learn_rate, learn_rate]
            
        elif 'Synaptic' in forager:
            assert all(x is not None for x in (
                learn_rate, forget_rate, I0, rho, softmax_temperature))
            self.I0 = I0
            self.rho = rho
            self.learn_rates = [learn_rate, learn_rate]
            self.forget_rates = [forget_rate, forget_rate]
            
        # Choice kernel can be added to any reward-based forager
        if '_CK' in forager:
            assert choice_step_size is not None and choice_softmax_temperature is not None
            self.choice_step_size = choice_step_size
            self.choice_softmax_temperature = choice_softmax_temperature

    def reset(self):

        #  print(self)

        # Initialization
        self.time = 0

        # All latent variables have n_trials + 1 length to capture the update after the last trial (HH20210726)
        self.q_estimation = np.full([self.K, self.n_trials + 1], np.nan)
        self.q_estimation[:, 0] = 0

        self.choice_prob = np.full([self.K, self.n_trials + 1], np.nan)
        self.choice_prob[:, 0] = 1/self.K   # To be strict (actually no use)

        if self.if_fit_mode:  # Predictive mode
            self.predictive_choice_prob = np.full(
                [self.K, self.n_trials + 1], np.nan)
            # To be strict (actually no use)
            self.predictive_choice_prob[:, 0] = 1/self.K

        else:   # Generative mode
            self.choice_history = np.zeros(
                [1, self.n_trials + 1], dtype=int)  # Choice history
            # Reward history, separated for each port (Corrado Newsome 2005)
            self.reward_history = np.zeros([self.K, self.n_trials + 1])

            # Generate baiting prob in block structure
            self.generate_p_reward()

            # Prepare reward for the first trial
            # For example, [0,1] represents there is reward baited at the RIGHT but not LEFT port.
            # Reward history, separated for each port (Corrado Newsome 2005)
            self.reward_available = np.zeros([self.K, self.n_trials + 1])
            self.reward_available[:, 0] = (np.random.uniform(
                0, 1, self.K) < self.p_reward[:, self.time]).astype(int)

        # Forager-specific
        if self.forager in ['RW1972_epsi', 'RW1972_softmax', 'Bari2019', 'Hattori2019',
                            'RW1972_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK']:
            pass

        elif self.forager in ['LNP_softmax', 'LNP_softmax_CK']:
            # Compute the history filter. Compatible with any number of taus.
            # Use the full length of the session just in case of an extremely large tau.
            reversed_t = np.flipud(np.arange(self.n_trials + 1))
            self.history_filter = np.zeros_like(reversed_t).astype('float64')

            for tau, w_tau in zip(self.taus, self.w_taus):
                # Note the normalization term (= tau when n -> inf.)
                self.history_filter += w_tau * \
                    np.exp(-reversed_t / tau) / \
                    np.sum(np.exp(-reversed_t / tau))

        elif self.forager in ['LossCounting']:
            # Initialize
            self.loss_count = np.zeros([1, self.n_trials + 1])
            if not self.if_fit_mode:
                self.loss_threshold_this = np.random.normal(
                    self.loss_count_threshold_mean, self.loss_count_threshold_std)

        elif 'CANN' in self.forager:
            if not self.if_fit_mode:   # Override user input of iti
                self.iti = np.ones(self.n_trials)
                
        elif 'Synaptic' in self.forager:
            self.w = np.full([self.K, self.n_trials + 1], np.nan)
            self.w[:, 0] = 0.1

        # Choice kernel can be added to any forager
        if '_CK' in self.forager:
            self.choice_kernel = np.zeros([self.K, self.n_trials + 1])

    def generate_p_reward(self, block_size_base=global_block_size_mean,
                          block_size_sd=global_block_size_sd,
                          # (Bari-Cohen 2019)
                          p_reward_pairs=[
                              [.4, .05], [.3857, .0643], [.3375, .1125], [.225, .225]],
                          ):

        # If para_optim, fix the random seed to ensure that p_reward schedule is fixed for all candidate parameters
        # However, we should make it random during a session (see the last line of this function)
        if self.p_reward_seed_override != '':
            np.random.seed(self.p_reward_seed_override)

        if self.p_reward_pairs == None:
            p_reward_pairs = np.array(
                p_reward_pairs) / 0.45 * self.p_reward_sum
        else:  # Full override of p_reward
            p_reward_pairs = self.p_reward_pairs

        # Adapted from Marton's code
        n_trials_now = 0
        block_size = []
        n_trials = self.n_trials + 1
        p_reward = np.zeros([2, n_trials])

        # Fill in trials until the required length
        while n_trials_now < n_trials:

            # Number of trials in each block (Gaussian distribution)
            # I treat p_reward[0,1] as the ENTIRE lists of reward probability. RIGHT = 0, LEFT = 1. HH
            n_trials_this_block = np.rint(np.random.normal(
                block_size_base, block_size_sd)).astype(int)
            n_trials_this_block = min(
                n_trials_this_block, n_trials - n_trials_now)

            block_size.append(n_trials_this_block)

            # Get values to fill for this block
            # If 0, the first block is set to 50% reward rate (as Marton did)
            if n_trials_now == -1:
                p_reward_this_block = np.array(
                    [[sum(p_reward_pairs[0])/2] * 2])  # Note the outer brackets
            else:
                # Choose reward_ratio_pair
                # If we had equal p_reward in the last block
                if n_trials_now > 0 and not(np.diff(p_reward_this_block)):
                    # We should not let it happen again immediately
                    pair_idx = np.random.choice(range(len(p_reward_pairs)-1))
                else:
                    pair_idx = np.random.choice(range(len(p_reward_pairs)))

                p_reward_this_block = np.array(
                    [p_reward_pairs[pair_idx]])   # Note the outer brackets

                # To ensure flipping of p_reward during transition (Marton)
                if len(block_size) % 2:
                    p_reward_this_block = np.flip(p_reward_this_block)

            # Fill in trials for this block
            p_reward[:, n_trials_now: n_trials_now +
                     n_trials_this_block] = p_reward_this_block.T

            # Fill choice history for some special foragers with choice patterns {AmBn} (including IdealpHatOptimal, IdealpHatGreedy, and AmB1)
            if self.forager == 'IdealpHatGreedy':
                self.get_AmBn_choice_history(
                    p_reward_this_block, n_trials_this_block, n_trials_now)

            # Next block
            n_trials_now += n_trials_this_block

        self.n_blocks = len(block_size)
        self.p_reward = p_reward
        self.block_size = np.array(block_size)
        self.p_reward_fraction = p_reward[RIGHT, :] / \
            (np.sum(p_reward, axis=0))   # For future use
        self.p_reward_ratio = p_reward[RIGHT, :] / \
            p_reward[LEFT, :]   # For future use

        # We should make it random afterwards
        np.random.seed()

    def get_AmBn_choice_history(self, p_reward_this_block, n_trials_this_block, n_trials_now):

        # Ideal-p^-Greedy
        mn_star_pHatGreedy, p_star_pHatGreedy = self.get_IdealpHatGreedy_strategy(
            p_reward_this_block[0])
        mn_star = mn_star_pHatGreedy

        # For ideal optimal, given p_0(t) and p_1(t), the optimal choice history is fixed, i.e., {m_star, 1} (p_min > 0)
        S = int(np.ceil(n_trials_this_block/(mn_star[0] + mn_star[1])))
        c_max_this = np.argwhere(p_reward_this_block[0] == np.max(
            p_reward_this_block))[0]  # To handle the case of p0 = p1
        c_min_this = np.argwhere(
            p_reward_this_block[0] == np.min(p_reward_this_block))[-1]
        # Choice pattern of {m_star, 1}
        c_star_this_block = ([c_max_this] * mn_star[0] +
                             [c_min_this] * mn_star[1]) * S
        # Truncate to the correct length
        c_star_this_block = c_star_this_block[:n_trials_this_block]

        self.choice_history[0, n_trials_now: n_trials_now +
                            n_trials_this_block] = c_star_this_block  # Save the optimal sequence

    def get_IdealpHatGreedy_strategy(self, p_reward):
        '''
        Ideal-p^-greedy, only care about the current p^, which is good enough (for 2-arm task)  03/28/2020
        '''
        p_max = np.max(p_reward)
        p_min = np.min(p_reward)

        if p_min > 0:
            m_star = np.floor(np.log(1-p_max)/np.log(1-p_min))
            p_star = p_max + (1-(1-p_min)**(m_star + 1)-p_max**2) / \
                (m_star+1)  # Still stands even m_star = *

            return [int(m_star), 1], p_star
        else:
            # Safe to be always on p_max side for this block
            return [self.n_trials, 1], p_max

    def act_random(self):

        if self.if_fit_mode:
            self.predictive_choice_prob[:,
                                        self.time] = 1/self.K + self.bias_terms
            choice = None   # No need to make specific choice in fitting mode
        else:
            # choice = np.random.choice(self.K)
            choice = choose_ps(1/self.K + self.bias_terms)
            self.choice_history[0, self.time] = choice
        return choice

    def act_LossCounting(self):

        if self.time == 0:  # Only this need special initialization
            if self.if_fit_mode:
                # No need to update self.predictive_choice_prob[:, self.time]
                pass
                return None
            else:
                return np.random.choice(self.K)

        if self.if_fit_mode:
            # Retrieve the last choice
            last_choice = self.fit_choice_history[0, self.time - 1]

            # Predict this choice prob
            # To be general, and ensure that alway switch when mean = 0, std = 0
            prob_switch = norm.cdf(
                self.loss_count[0, self.time], self.loss_count_threshold_mean - 1e-6, self.loss_count_threshold_std + 1e-16)

            # Choice prob [choice] = 1-prob_switch, [others] = prob_switch /(K-1). Assuming randomly switch to other alternatives
            self.predictive_choice_prob[:,
                                        self.time] = prob_switch / (self.K - 1)
            self.predictive_choice_prob[last_choice,
                                        self.time] = 1 - prob_switch

            # Using fit_choice to mark an actual switch
            if self.time < self.n_trials and last_choice != self.fit_choice_history[0, self.time]:
                # A flag of "switch happens here"
                self.loss_count[0, self.time] = - self.loss_count[0, self.time]

            choice = None

        else:
            # Retrieve the last choice
            last_choice = self.choice_history[0, self.time - 1]

            if self.loss_count[0, self.time] >= self.loss_threshold_this:
                # Switch
                choice = LEFT + RIGHT - last_choice

                # Reset loss counter threshold
                # A flag of "switch happens here"
                self.loss_count[0, self.time] = - self.loss_count[0, self.time]
                self.loss_threshold_this = np.random.normal(
                    self.loss_count_threshold_mean, self.loss_count_threshold_std)
            else:
                # Stay
                choice = last_choice

            self.choice_history[0, self.time] = choice

        return choice

    def act_EpsiGreedy(self):

        # if np.random.rand() < self.epsilon:
        #     # Forced exploration with the prob. of epsilon (to avoid AlwaysLEFT/RIGHT in Sugrue2004...) or before some rewards are collected
        #     choice = self.act_random()

        # else:    # Greedy
        #     choice = np.random.choice(np.where(self.q_estimation[:, self.time] == self.q_estimation[:, self.time].max())[0])
        #     if self.if_fit_mode:
        #         self.predictive_choice_prob[:, self.time] = 0
        #         self.predictive_choice_prob[choice, self.time] = 1  # Delta-function
        #         choice = None   # No need to make specific choice in fitting mode
        #     else:
        #         self.choice_history[0, self.time] = choice

        # == The above is erroneous!! We should never realize any probabilistic events in model fitting!! ==
        choice = np.random.choice(np.where(
            self.q_estimation[:, self.time] == self.q_estimation[:, self.time].max())[0])

        if self.if_fit_mode:
            self.predictive_choice_prob[:, self.time] = self.epsilon * \
                (1 / self.K + self.bias_terms)
            self.predictive_choice_prob[choice, self.time] = 1 - self.epsilon + \
                self.epsilon * (1 / self.K + self.bias_terms[choice])
            choice = None   # No need to make specific choice in fitting mode
        else:
            if np.random.rand() < self.epsilon:
                choice = self.act_random()

            self.choice_history[0, self.time] = choice

        return choice

    def act_Probabilistic(self):

        # !! Should not change q_estimation!! Otherwise will affect following Qs
        # And I put softmax here
        if '_CK' in self.forager:
            self.choice_prob[:, self.time] = softmax(np.vstack([self.q_estimation[:, self.time], self.choice_kernel[:, self.time]]),
                                                     np.vstack(
                                                         [self.softmax_temperature, self.choice_softmax_temperature]),
                                                     bias=self.bias_terms)  # Updated softmax function that accepts two elements
        else:
            self.choice_prob[:, self.time] = softmax(
                self.q_estimation[:, self.time], self.softmax_temperature, bias=self.bias_terms)

        if self.if_fit_mode:
            self.predictive_choice_prob[:,
                                        self.time] = self.choice_prob[:, self.time]
            choice = None   # No need to make specific choice in fitting mode
        else:
            choice = choose_ps(self.choice_prob[:, self.time])
            self.choice_history[0, self.time] = choice

        return choice

    def step_LossCounting(self, reward):

        if self.loss_count[0, self.time - 1] < 0:  # A switch just happened
            # Back to normal (Note that this = 0 in Shahidi 2019)
            self.loss_count[0, self.time - 1] = - \
                self.loss_count[0, self.time - 1]
            if reward:
                self.loss_count[0, self.time] = 0
            else:
                self.loss_count[0, self.time] = 1
        else:
            if reward:
                self.loss_count[0,
                                self.time] = self.loss_count[0, self.time - 1]
            else:
                self.loss_count[0, self.time] = self.loss_count[0,
                                                                self.time - 1] + 1

    def step_LNP(self, valid_reward_history):

        valid_filter = self.history_filter[-self.time:]
        local_income = np.sum(valid_reward_history * valid_filter, axis=1)

        self.q_estimation[:, self.time] = local_income

    def step_RWlike(self, choice, reward):

        # Reward-dependent step size ('Hattori2019')
        if reward:
            learn_rate_this = self.learn_rates[1]
        else:
            learn_rate_this = self.learn_rates[0]

        # Choice-dependent forgetting rate ('Hattori2019')
        # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
        self.q_estimation[choice, self.time] = (1 - self.forget_rates[1]) * self.q_estimation[choice, self.time - 1]  \
            + learn_rate_this * \
            (reward - self.q_estimation[choice, self.time - 1])

        # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
        unchosen_idx = [cc for cc in range(self.K) if cc != choice]
        self.q_estimation[unchosen_idx, self.time] = (
            1 - self.forget_rates[0]) * self.q_estimation[unchosen_idx, self.time - 1]

        # --- The below three lines are erroneous!! Should not change q_estimation!! ---
        # Softmax in 'Bari2019', 'Hattori2019'
        # if self.forager in ['RW1972_softmax', 'Bari2019', 'Hattori2019']:
        #     self.q_estimation[:, self.time] = softmax(self.q_estimation[:, self.time], self.softmax_temperature)

    def step_CANN(self, choice, reward):
        """
        Abstracted from Ulises' line attractor model
        """
        if reward:
            learn_rate_this = self.learn_rates[1]
        else:
            learn_rate_this = self.learn_rates[0]
            
        # ITI[self.time] --> ITI between (self.time) and (self.time + 1)
        iti_time_minus1_to_time = self.iti[self.time - 1]

        # Choice-dependent forgetting rate ('Hattori2019')
        # Chosen:   Q(n+1) = (1- forget_rate_chosen) * Q(n) + step_size * (Reward - Q(n))
        self.q_estimation[choice, self.time] = (self.q_estimation[choice, self.time - 1]
                                                + learn_rate_this * (reward - self.q_estimation[choice, self.time - 1])
                                                ) * np.exp( -iti_time_minus1_to_time / self.tau_cann)

        # Unchosen: Q(n+1) = (1-forget_rate_unchosen) * Q(n)
        unchosen_idx = [cc for cc in range(self.K) if cc != choice]
        self.q_estimation[unchosen_idx, self.time] = self.q_estimation[unchosen_idx, self.time - 1
                                                                       ] * np.exp( -iti_time_minus1_to_time / self.tau_cann)
        
    @staticmethod
    def f(x): 
        return 0 if x <= 0 else 1 if x >= 1 else x

    def step_synaptic(self, choice, reward):
        """
        Abstracted from Ulises' mean-field synaptic model
        """
        # -- Update w --
        if reward:
            learn_rate_this = self.learn_rates[1]
        else:
            learn_rate_this = self.learn_rates[0]
            
        # Chosen side
        self.w[choice, self.time] = (1 - self.forget_rates[1]) * self.w[choice, self.time - 1] \
            + learn_rate_this * (reward - self.q_estimation[choice, self.time - 1]) * self.q_estimation[choice, self.time - 1]
        # Unchosen side
        self.w[1 - choice, self.time] = (1 - self.forget_rates[0]) * self.w[1 - choice, self.time - 1]
        
        # Rectify w if needed
        if self.forager == 'Synaptic_W>0':
            self.w[self.w[:, self.time] < 0, self.time] = 0
        
        # -- Update u --
        for side in [0, 1]:
            self.q_estimation[side, self.time] = self.f(self.I0 * (1 - self.w[1 - side, self.time]) / 
                                        (self.w[:, self.time].prod() - (1 + self.rho / 2) * self.w[:, self.time].sum() + 1 + self.rho))
            

    def step_choice_kernel(self, choice):
        # Choice vector
        choice_vector = np.zeros([self.K])
        choice_vector[choice] = 1

        # Update choice kernel (see Model 5 of Wilson and Collins, 2019)
        # Note that if chocie_step_size = 1, degenerates to Bari 2019 (choice kernel = the last choice only)
        self.choice_kernel[:, self.time] = self.choice_kernel[:, self.time - 1] \
            + self.choice_step_size * \
            (choice_vector - self.choice_kernel[:, self.time - 1])

    def act(self):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!

        # -- Predefined --
        # Foragers that have the pattern {AmBn} (not for fitting)
        if self.forager in ['IdealpHatGreedy']:
            return self.choice_history[0, self.time]  # Already initialized

        # Probability matching of base probabilities p (not for fitting)
        if self.forager == 'pMatching':
            choice = choose_ps(self.p_reward[:, self.time])
            self.choice_history[0, self.time] = choice
            return choice

        if self.forager == 'Random':
            return self.act_random()

        if self.forager == 'LossCounting':
            return self.act_LossCounting()

        if self.forager in ['RW1972_epsi']:
            return self.act_EpsiGreedy()

        if self.forager in ['RW1972_softmax', 'LNP_softmax', 'Bari2019', 'Hattori2019',
                            'RW1972_softmax_CK', 'LNP_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK',
                            'CANN', 'Synaptic', 'Synaptic_W>0']:   # Probabilistic (Could have choice kernel)
            return self.act_Probabilistic()

        print('No action found!!')

    def step(self, choice):  # Compatible with either fitting mode (predictive) or not (generative). It's much clear now!!

        if self.if_fit_mode:
            #  In fitting mode, retrieve choice and reward from the targeted fit_c and fit_r
            choice = self.fit_choice_history[0, self.time]  # Override choice
            # Override reward
            reward = self.fit_reward_history[choice, self.time]

        else:
            #  In generative mode, generate reward and make the state transition
            reward = self.reward_available[choice, self.time]
            # Note that according to Sutton & Barto's convention,
            self.reward_history[choice, self.time] = reward
            # this update should belong to time t+1, but here I use t for simplicity.

            # An intermediate reward status. Note the .copy()!
            reward_available_after_choice = self.reward_available[:, self.time].copy(
            )
            # The reward is depleted at the chosen lick port.
            reward_available_after_choice[choice] = 0

        # =================================================
        self.time += 1   # Time ticks here !!!
        # Doesn't terminate here to finish the final update after the last trial
        # if self.time == self.n_trials:
        #     return   # Session terminates
        # =================================================

        # Prepare reward for the next trial (if sesson did not end)
        if not self.if_fit_mode:
            # Generate the next reward status, the "or" statement ensures the baiting property, gated by self.if_baited.
            self.reward_available[:, self.time] = np.logical_or(reward_available_after_choice * self.if_baited,
                                                                np.random.uniform(0, 1, self.K) < self.p_reward[:, self.time]).astype(int)

        # Update value function etc.
        if self.forager in ['LossCounting']:
            self.step_LossCounting(reward)

        elif self.forager in ['RW1972_softmax', 'RW1972_epsi', 'Bari2019', 'Hattori2019',
                              'RW1972_softmax_CK', 'Bari2019_CK', 'Hattori2019_CK']:
            self.step_RWlike(choice, reward)

        elif self.forager in ['CANN']:
            self.step_CANN(choice, reward)
        
        elif self.forager in ['Synaptic']:
            self.step_synaptic(choice, reward)

        elif self.forager in ['LNP_softmax', 'LNP_softmax_CK']:
            if self.if_fit_mode:
                # Targeted history till now
                valid_reward_history = self.fit_reward_history[:, :self.time]
            else:
                # Models' history till now
                valid_reward_history = self.reward_history[:, :self.time]

            self.step_LNP(valid_reward_history)

        if '_CK' in self.forager:  # Could be independent of other foragers, so use "if" rather than "elif"
            self.step_choice_kernel(choice)

    def simulate(self):

        # =============================================================================
        # Simulate one session
        # =============================================================================
        self.reset()

        for t in range(self.n_trials):
            action = self.act()
            self.step(action)

        if self.if_fit_mode:
            # Allow the final update of action prob after the last trial (for comparing with ephys)
            action = self.act()
