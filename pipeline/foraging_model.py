import datajoint as dj
from . import experiment, get_schema_name, foraging_analysis

schema = dj.schema(get_schema_name('foraging_model'))


@schema
class ModelClass(dj.Lookup):
    definition = """
    model_class: varchar(32)  # e.g. LossCounting, RW1972, Hattori2019
    ---
    desc='': varchar(100)
    """
    contents = [
        ['LossCounting', 'Count the number of losses and switch when the number exceeds a threshold'],
        ['RW1972', 'Rescorla–Wagner model (single learnig rate)'],
        ['LNP', 'Linear-nonlinear-Possion (exponential recency-weighted average)'],
        ['Bari2019', 'Bari et al 2019 (different learning rates for chosen/unchosen)'],
        ['Hattori2019', 'Hattori et al 2019 (different learning rates for rew/unrew/unchosen)'],
    ]


@schema
class ModelParam(dj.Lookup):
    definition = """
    model_param: varchar(32)  # e.g. learn_rate, epsilon, w_tau1
    ---
    param_notation: varchar(32)  # r'$\tau_1$'
    """

    contents = [
        ['loss_count_threshold_mean', r'$\mu_{LC}$'],
        ['loss_count_threshold_std', r'$\sigma_{LC}$'],
        ['tau1', r'$\tau_1$'],
        ['tau2', r'$\tau_2$'],
        ['w_tau1', r'$w_{\tau_1}$'],
        ['learn_rate', r'$\alpha$'],
        ['learn_rate_rew', r'$\alpha_{rew}$'],
        ['learn_rate_unrew', r'$\alpha_{unr}$'],
        ['forget_rate', r'$\delta$'],
        ['softmax_temperature', r'$\sigma$'],
        ['epsilon', r'$\epsilon$'],
        ['biasL', r'$b_L$'],
        ['biasR', r'$b_R$'],
        ['choice_step_size', r'$\alpha_c$'],
        ['choice_softmax_temperature', r'$\sigma_c$'],
    ]


@schema
class Model(dj.Manual):
    definition = """
    model_id: int
    ---
    -> ModelClass
    model_notation: varchar(500)
    param_count: int
    is_bias: bool
    is_epsilon_greedy: bool
    is_softmax: bool
    is_choice_kernel: bool
    desc='': varchar(500)  # Long name
    fit_cmd: blob    # Fitting command compatible with the Dynamic-Foraing repo
    """

    class Param(dj.Part):
        definition = """
        -> master
        -> ModelParam
        ---
        param_lower_bound: float
        param_higher_bound: float
        """

    @classmethod
    def load_models(cls):
        # Original definition from the Dynamic-Foraging repo, using the format: [forager, [para_names], [lower bounds], [higher bounds], desc(optional)]
        MODELS = [
            # No bias
            ['LossCounting', ['loss_count_threshold_mean', 'loss_count_threshold_std'], 
             [0, 0], [40, 10], 'LossCounting: mean, std, no bias'],
            
            ['RW1972_epsi', ['learn_rate', 'epsilon'], 
             [0, 0], [1, 1], 'SuttonBarto: epsilon, no bias'],
            
            ['RW1972_softmax', ['learn_rate', 'softmax_temperature'], 
             [0, 1e-2], [1, 15], 'SuttonBarto: softmax, no bias'],

            ['LNP_softmax', ['tau1', 'softmax_temperature'], 
             [1e-3, 1e-2], [100, 15], 'Sugrue2004, Corrado2005: one tau, no bias'],
            
            ['LNP_softmax', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature'], 
             [1e-3, 1e-1, 0, 1e-2], [15, 40, 1, 15], 'Corrado2005, Iigaya2019: two taus, no bias'],
            
            ['Bari2019', ['learn_rate', 'forget_rate', 'softmax_temperature'], 
             [0, 0, 1e-2], [1, 1, 15], 'RL: chosen, unchosen, softmax, no bias'],
            
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature'], 
             [0, 0, 1e-2], [1, 1, 15], 'RL: rew, unrew, softmax, no bias'],
            
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature'], 
             [0, 0, 0, 1e-2], [1, 1, 1, 15], 'RL: rew, unrew, unchosen, softmax, no bias'],

            # With bias
            ['RW1972_epsi', ['learn_rate', 'epsilon', 'biasL'], 
             [0, 0, -0.5], [1, 1, 0.5], 'SuttonBarto: epsilon'],
            
            ['RW1972_softmax', ['learn_rate', 'softmax_temperature', 'biasL'], 
             [0, 1e-2, -5], [1, 15, 5], 'SuttonBarto: softmax'],
            
            ['LNP_softmax', ['tau1', 'softmax_temperature', 'biasL'], 
             [1e-3, 1e-2, -5], [100, 15, 5], 'Sugrue2004, Corrado2005: one tau'],
            
            ['LNP_softmax', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature', 'biasL'], 
             [1e-3, 1e-1, 0, 1e-2, -5], [15, 40, 1, 15, 5], 'Corrado2005, Iigaya2019: two taus'],
            
            ['Bari2019', ['learn_rate', 'forget_rate', 'softmax_temperature', 'biasL'], 
             [0, 0, 1e-2, -5], [1, 1, 15, 5], 'RL: chosen, unchosen, softmax'],
            
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature', 'biasL'], 
             [0, 0, 1e-2, -5], [1, 1, 15, 5], 'RL: rew, unrew, softmax'],
            
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature', 'biasL'], 
             [0, 0, 0, 1e-2, -5], [1, 1, 1, 15, 5], '(full Hattori) RL: rew, unrew, unchosen, softmax'],

            # With bias and choice kernel 
            ['RW1972_softmax_CK', ['learn_rate', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 1e-2, -5, 0, 1e-2], [1, 15, 5, 1, 20], 'SuttonBarto: softmax, choice kernel'],
            
            ['LNP_softmax_CK', ['tau1', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [1e-3, 1e-2, -5, 0, 1e-2], [100, 15, 5, 1, 20], 'Sugrue2004, Corrado2005: one tau, choice kernel'],
            
            ['LNP_softmax_CK', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [1e-3, 1e-1, 0, 1e-2, -5, 0, 1e-2], [15, 40, 1, 15, 5, 1, 20], 'Corrado2005, Iigaya2019: two taus, choice kernel'],
            
            ['Bari2019_CK', ['learn_rate', 'forget_rate', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 0, 1e-2, -5, 0, 1e-2], [1, 1, 15, 5, 1, 20], 'RL: chosen, unchosen, softmax, choice kernel'],
            
            ['Hattori2019_CK', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 0, 1e-2, -5, 0, 1e-2], [1, 1, 15, 5, 1, 20], 'RL: rew, unrew, softmax, choice kernel'],
            
            ['Hattori2019_CK', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 0, 0, 1e-2, -5, 0, 1e-2], [1, 1, 1, 15, 5, 1, 20], 'Hattori + choice kernel'],
            
            ['Hattori2019_CK', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 0, 0, 1e-2, -5, 1, 1e-2], [1, 1, 1, 15, 5, 1, 20], 'choice_step_size fixed at 1 --> Bari 2019: only the last choice matters'],

        ]

        # Parse and insert MODELS
        for model_id, model in enumerate(MODELS):
            # Insert Model
            model_class = [mc for mc in ModelClass.fetch("model_class") if mc in model[0]][0]
            param_count = len(model[1])
            is_bias = True if any(['bias' in param for param in model[1]]) else False
            is_epsilon_greedy = True if 'epsilon' in model[1] else False
            is_softmax = True if 'softmax_temperature' in model[1] else False
            is_choice_kernel = True if 'choice_step_size' in model[1] else False

            param_notation = ', '.join((ModelParam & f'model_param="{p}"').fetch1("param_notation") for p in model[1])
            model_notation = f'{model[0]} ({param_notation})'
            desc = model[4] if len(model) == 5 else '' # model[0] + ': ' + ', '.join(model[1])  # Use the user-defined string if exists

            Model.insert1(dict(model_id=model_id, model_class=model_class, model_notation=model_notation, param_count=param_count,
                             is_bias=is_bias, is_epsilon_greedy=is_epsilon_greedy, is_softmax=is_softmax, is_choice_kernel=is_choice_kernel,
                             desc=desc, fit_cmd=model[:4]),
                        skip_duplicates=True)

            # Insert Model.Param
            for param, lb, ub in zip(*model[1:4]):
                Model.Param.insert1(dict(model_id=model_id, model_param=param,
                                       param_lower_bound=lb, param_higher_bound=ub),
                                  skip_duplicates=True)

        return


@schema
class FittedSession(dj.Computed):
    definition = """
    -> Model
    -> experiment.Session
    ---
    aic: float
    log10_bf_aic: float
    model_weight_aic: float
    bic: float
    log10_bf_bic: float
    model_weight_bic: float 
    """

    key_source = (foraging_analysis.SessionTaskProtocol() & 'session_real_foraging') * Model()

    class FittedParam(dj.Part):
        definition = """
        -> master
        -> Model.Param
        ---
        fitted_value: float
        """

    def make(self, key):
        
        
        
        # call BanditModelComparison
        pass




def fit_each_mice(data, if_session_wise = False, if_verbose = True, file_name = '', pool = '', models = None):
    choice = data.f.choice
    reward = data.f.reward
    p1 = data.f.p1
    p2 = data.f.p2
    session_num = data.f.session
    
    # -- Formating --
    # Remove ignores
    valid_trials = choice != 0
    
    choice_history = choice[valid_trials] - 1  # 1: LEFT, 2: RIGHT --> 0: LEFT, 1: RIGHT
    reward = reward[valid_trials]
    p_reward = np.vstack((p1[valid_trials],p2[valid_trials]))
    session_num = session_num[valid_trials]
    
    n_trials = len(choice_history)
    print('Total valid trials = %g' % n_trials)
    sys.stdout.flush()
    
    reward_history = np.zeros([2,n_trials])
    for c in (0,1):  
        reward_history[c, choice_history == c] = (reward[choice_history == c] > 0).astype(int)
    
    choice_history = np.array([choice_history])
    
    results_each_mice = {}
    
    # -- Model comparison for each session --
    if if_session_wise:
        
        model_comparison_session_wise = []
        
        unique_session = np.unique(session_num)
        
        for ss in tqdm(unique_session, desc = 'Session-wise', total = len(unique_session)):
            choice_history_this = choice_history[:, session_num == ss]
            reward_history_this = reward_history[:, session_num == ss]
                
            model_comparison_this = BanditModelComparison(choice_history_this, reward_history_this, models = models)
            model_comparison_this.fit(pool = pool, plot_predictive = None, if_verbose = False) # Plot predictive traces for the 1st, 2nd, and 3rd models
            model_comparison_session_wise.append(model_comparison_this)
                
        results_each_mice['model_comparison_session_wise'] = model_comparison_session_wise
    
    # -- Model comparison for all trials --
    
    print('Pooling all sessions: ', end='')
    start = time.time()
    model_comparison_grand = BanditModelComparison(choice_history, reward_history, p_reward = p_reward, session_num = session_num, models = models)
    model_comparison_grand.fit(pool = pool, plot_predictive = None if if_session_wise else [1,2,3], if_verbose = if_verbose) # Plot predictive traces for the 1st, 2nd, and 3rd models
    print(' Done in %g secs' % (time.time() - start))
    
    if if_verbose:
        model_comparison_grand.show()
        model_comparison_grand.plot()
    
    results_each_mice['model_comparison_grand'] = model_comparison_grand    
    
    return results_each_mice



# ============= Helpers =============

def get_session_history(session_key={'subject_id': 447921, 'session': 3}):
    q_choice_outcome = (experiment.WaterPortChoice.proj(choice='water_port')
                        * experiment.BehaviorTrial.proj('outcome', 'early_lick') 
                        * experiment.SessionBlock.BlockTrial) & session_key
    q_p_reward = experiment.SessionBlock.WaterPortRewardProbability()

    for lickport in ['left', 'right']:
        pass



















