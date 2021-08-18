import datajoint as dj
import numpy as np
from . import experiment, ephys, get_schema_name, foraging_analysis
from .model.bandit_model_comparison import BanditModelComparison

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
        ['CANN', "Abstracted from Ulises' continuous attractor neural network"],
        ['Synaptic', "Abstracted from Ulises' synaptic model"]
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
        ['tau_cann', r'$\tau_{CANN}$'],
        ['I0', r'$I_0$'],
        ['rho', r'$\rho$']
    ]


@schema
class Model(dj.Manual):
    definition = """
    model_id: int
    ---
    -> ModelClass
    model_notation: varchar(500)
    n_params: int   # Effective param count
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
        param_idx: int   # To keep params the same order as the original definition in MODELS, hence `fit_result.x`
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

            ['CANN', ['learn_rate', 'tau_cann', 'softmax_temperature', 'biasL'],
             [0, 0, 1e-2, -5], [1, 1000, 15, 5], "Ulises' CANN model, ITI decay, with bias"],
            
            ['Synaptic', ['learn_rate', 'forget_rate', 'I0', 'rho', 'softmax_temperature', 'biasL'],
             [0, 0, 0, 0, 1e-2, -5], [1, 1, 10, 1, 15, 5], "Ulises' synaptic model"],
            
            ['Synaptic', ['learn_rate', 'forget_rate', 'I0', 'rho', 'softmax_temperature', 'biasL'],
             [0, 0, 0, -100, 1e-2, -5], [1, 1, 10, 100, 15, 5], "Ulises' synaptic model (unconstrained \\rho)"],
            
            ['Synaptic', ['learn_rate', 'forget_rate', 'I0', 'rho', 'softmax_temperature', 'biasL'],
             [0, 0, 0, -1e6, 1e-2, -5], [1, 1, 1e6, 1e6, 15, 5], "Ulises' synaptic model (really unconstrained I_0 and \\rho)"],
        ]

        # Parse and insert MODELS
        for model_id, model in enumerate(MODELS):
            # Insert Model
            model_class = [mc for mc in ModelClass.fetch("model_class") if mc in model[0]][0]
            is_bias = True if any(['bias' in param for param in model[1]]) else False
            is_epsilon_greedy = True if 'epsilon' in model[1] else False
            is_softmax = True if 'softmax_temperature' in model[1] else False
            is_choice_kernel = True if 'choice_step_size' in model[1] else False

            n_params = 0
            param_notation = []

            # Insert Model
            for param, lb, ub in zip(*model[1:4]):
                if lb < ub:
                    n_params += 1  # Only count effective params
                    param_notation.append((ModelParam & f'model_param="{param}"').fetch1("param_notation"))
                else:
                    param_notation.append((ModelParam & f'model_param="{param}"').fetch1("param_notation") + f'= {lb}')

            param_notation = ', '.join(param_notation)
            model_notation = f'{model[0]} ({param_notation})'
            desc = model[4] if len(model) == 5 else ''  # model[0] + ': ' + ', '.join(model[1])  # Use the user-defined string if exists

            Model.insert1(dict(model_id=model_id, model_class=model_class, model_notation=model_notation, n_params=n_params,
                               is_bias=is_bias, is_epsilon_greedy=is_epsilon_greedy, is_softmax=is_softmax, is_choice_kernel=is_choice_kernel,
                               desc=desc, fit_cmd=model[:4]),
                          skip_duplicates=True)

            # Insert Model.Param
            for idx, (param, lb, ub) in enumerate(zip(*model[1:4])):
                # The result table should save both effective and fixed params
                Model.Param.insert1(dict(model_id=model_id, model_param=param, param_idx=idx,
                                         param_lower_bound=lb, param_higher_bound=ub),
                                    skip_duplicates=True)

        return


@schema
class ModelComparison(dj.Lookup):
    # Define model comparison groups
    definition = """
    model_comparison_idx:        smallint
    ---
    desc='':     varchar(200)
    """

    class Competitor(dj.Part):
        definition = """
        -> master
        competitor_idx:          int
        ---
        -> Model
        """

    @classmethod
    def load(cls):
        model_comparisons = [
            ['all_models', Model],
            ['models_with_bias', Model & 'is_bias = 1'],
            ['models_with_bias_and_choice_kernel', Model & 'is_bias = 1' & 'is_choice_kernel']
        ]

        # Parse and insert ModelComparisonGroup
        for mc_idx, (desc, models) in enumerate(model_comparisons):
            cls.insert1(dict(model_comparison_idx=mc_idx, desc=desc), skip_duplicates=True)
            cls.Competitor.insert([
                dict(model_comparison_idx=mc_idx, competitor_idx=idx, model_id=model_id)
                for idx, model_id in enumerate(models.fetch('model_id', order_by='model_id'))
            ], skip_duplicates=True)


@schema
class FittedSessionModel(dj.Computed):
    definition = """
    -> experiment.Session
    -> Model
    ---
    n_trials: int
    n_params: int  # Number of effective params (return from the fitting function; should be the same as Model.n_params)
    log_likelihood: float  # raw log likelihood of the model
    aic: float   # AIC
    bic: float   # BIC
    lpt: float       # Likelihood-Per-Trial raw
    lpt_aic: float   # Likelihood-Per-Trial with AIC penalty 
    lpt_bic: float   # Likelihood-Per-Trial with AIC penalty 
    prediction_accuracy: float    # non-cross-validated prediction accuracy 
    cross_valid_accuracy_fit: float     # cross-validated accuracy (fitting set)
    cross_valid_accuracy_test: float    # cross-validated accuracy (testing set)
    cross_valid_accuracy_test_bias_only = NULL: float    # accuracy predicted only by bias (testing set)
    """

    key_source = ((foraging_analysis.SessionTaskProtocol() & 'session_task_protocol = 100' & 'session_real_foraging'
                  ) * Model()) # * experiment.Session() & 'session_date > "2021-01-01"' # & 'model_id > 21' # & 'subject_id = 482350'

    class Param(dj.Part):
        definition = """
        -> master
        -> Model.Param
        ---
        fitted_value: float
        """

    class TrialLatentVariable(dj.Part):
        """
        To save all fitted latent variables that will be correlated to ephys
        Notes:
        1. In the original definition (Sutton&Barto book), the updated value after choice of trial t is Q(t+1), not Q(t)!
                behavior & ephys:   -->  ITI(t-1) --> |  --> choice (t), reward(t)         --> ITI (t) -->            |
                model:           Q(t) --> choice prob(t) --> choice (t), reward(t)  | --> Q(t+1) --> choice prob (t+1)
           Therefore: the ITI of trial t -> t+1 corresponds to Q(t+1)
        2. To make it more intuitive, when they are inserted here, Q is offset by -1,
           such that ephys and model are aligned:
                behavior & ephys:   -->  ITI(t-1) --> |  --> choice (t), reward(t)         --> ITI (t) -->       |
                model:    Q(t-1) --> choice prob(t-1) | --> choice (t), reward(t)  --> Q(t) --> choice prob (t)  |
           This will eliminate the need of adding an offset=-1 whenever ephys and behavioral model are compared.
        3. By doing this, I also save the update after the last trial (useless for fitting; useful for ephys) ,
           whereas the first trial is discarded, which was randomly initialized anyway
        """
        definition = """
        -> master
        -> experiment.SessionTrial
        -> experiment.WaterPort 
        ---
        action_value=null:   float
        choice_prob=null:    float    
        choice_kernel=null:  float
        """

    def make(self, key):
        choice_history, reward_history, iti, p_reward, q_choice_outcome = get_session_history(key)
        model_str = (Model & key).fetch('fit_cmd')

        # --- Actual fitting ---
        if (Model & key).fetch1('model_class') in ['CANN']:  # Only pass ITI if this is CANN model (to save some time?)
            model_comparison_this = BanditModelComparison(choice_history, reward_history, iti=iti, model=model_str)
        else:
            model_comparison_this = BanditModelComparison(choice_history, reward_history, iti=None, model=model_str)
        model_comparison_this.fit(pool='', plot_predictive=None, if_verbose=False)  # Parallel on sessions, not on DE
        model_comparison_this.cross_validate(pool='', k_fold=2, if_verbose=False)

        # ------ Grab results ----
        fit_result = model_comparison_this.results_raw[0]
        cross_valid_result = model_comparison_this.prediction_accuracy_CV.iloc[0]

        # Insert session fitted stats
        self.insert1(dict(**key,
                          n_trials=fit_result.n_trials,
                          n_params=fit_result.k_model,
                          log_likelihood=fit_result.log_likelihood,
                          aic=fit_result.AIC,
                          bic=fit_result.BIC,
                          lpt=fit_result.LPT,
                          lpt_aic=fit_result.LPT_AIC,
                          lpt_bic=fit_result.LPT_BIC,
                          prediction_accuracy=fit_result.prediction_accuracy,
                          cross_valid_accuracy_fit=np.mean(cross_valid_result.prediction_accuracy_fit),
                          cross_valid_accuracy_test=np.mean(cross_valid_result.prediction_accuracy_test),
                          cross_valid_accuracy_test_bias_only=np.mean(cross_valid_result.prediction_accuracy_test_bias_only),
                          )
                     )

        # Insert fitted params (`order_by` is critical!)
        self.Param.insert([dict(**key, model_param=param, fitted_value=x) 
	                         for param, x in zip((Model.Param & key).fetch('model_param', order_by='param_idx'), fit_result.x)])
        
        # Insert latent variables (trial number offset -1 here!!)
        choice_prob = fit_result.predictive_choice_prob[:, 1:]  # Model must have this
        action_value = fit_result.action_value[:, 1:] if hasattr(fit_result, 'action_value') else np.full_like(choice_prob, np.nan)
        choice_kernel = fit_result.choice_kernel[:, 1:] if hasattr(fit_result, 'choice_kernel') else np.full_like(choice_prob, np.nan)

        for water_port_idx, water_port in enumerate(['left', 'right']):
            key['water_port'] = water_port

            self.TrialLatentVariable.insert(
                [{**key, 'trial': i, 'choice_prob': prob, 'action_value': value, 'choice_kernel': ck}
                 for i, prob, value, ck in zip(q_choice_outcome.fetch('trial', order_by='trial'),
                                               choice_prob[water_port_idx, :],
                                               action_value[water_port_idx, :],
                                               choice_kernel[water_port_idx, :])]
            )                

@schema
class FittedSessionModelComparison(dj.Computed):
    definition = """
    -> experiment.Session
    -> ModelComparison
    """

    key_source = (experiment.Session & FittedSessionModel) * ModelComparison  # Only include already-fitted sessions

    class RelativeStat(dj.Part):
        definition = """
        -> master
        -> Model
        ---
        relative_likelihood_aic:    float
        relative_likelihood_bic:    float
        model_weight_aic:           float
        model_weight_bic:           float
        log10_bf_aic:               float   # log_10 (Bayes factor)
        log10_bf_bic:               float   # log_10 (Bayes factor)
        """

    class BestModel(dj.Part):
        definition = """
        -> master
        ---
        best_aic:      int   # model_id of the model with the smallest aic
        best_bic:      int    # model_id of the model with the smallest bic
        best_cross_validation_test:      int   # model_id of the model with the highest cross validation test accuracy
        """

    def make(self, key):
        competing_models = ModelComparison.Competitor & key
        results = (FittedSessionModel & key & competing_models).fetch(format='frame').reset_index()
        if len(results) < len(competing_models):   # not all fitting results of competing models are ready
            return

        delta_aic = results.aic - np.min(results.aic)
        delta_bic = results.bic - np.min(results.bic)

        # Relative likelihood = Bayes factor = p_model/p_best = exp( - delta_aic / 2)
        results['relative_likelihood_aic'] = np.exp(- delta_aic / 2)
        results['relative_likelihood_bic'] = np.exp(- delta_bic / 2)

        # Model weight = Relative likelihood / sum(Relative likelihood)
        results['model_weight_aic'] = results['relative_likelihood_aic'] / np.sum(results['relative_likelihood_aic'])
        results['model_weight_bic'] = results['relative_likelihood_bic'] / np.sum(results['relative_likelihood_bic'])

        # log_10 (Bayes factor) = log_10 (exp( - delta_aic / 2)) = (-delta_aic / 2) / log(10)
        results['log10_bf_aic'] = - delta_aic / 2 / np.log(10)  # Calculate log10(Bayes factor) (relative likelihood)
        results['log10_bf_bic'] = - delta_bic / 2 / np.log(10)  # Calculate log10(Bayes factor) (relative likelihood)

        best_aic = results.model_id[np.argmin(results.aic)]
        best_bic = results.model_id[np.argmin(results.bic)]
        best_cross_validation_test = results.model_id[np.argmax(results.cross_valid_accuracy_test)]

        results['model_comparison_idx'] = key['model_comparison_idx']
        self.insert1(key)
        self.RelativeStat.insert(results, ignore_extra_fields=True, skip_duplicates=True)
        self.BestModel.insert1({**key,
                                'best_aic': best_aic,
                                'best_bic': best_bic,
                                'best_cross_validation_test': best_cross_validation_test})


# ============= Helpers =============

def get_session_history(session_key, remove_ignored=True):
    # Fetch data
    q_choice_outcome = (experiment.WaterPortChoice.proj(choice='water_port')
                        * experiment.BehaviorTrial.proj('outcome', 'early_lick')
                        * experiment.SessionBlock.BlockTrial) & session_key  # Remove ignored trials
    if remove_ignored:
        q_choice_outcome &= 'outcome != "ignore"'

    # TODO: session QC (warm-up and decreased motivation etc.)

    # -- Choice and reward --
    # 0: left, 1: right, np.nan: ignored
    _choice = q_choice_outcome.fetch('choice', order_by='trial')
    _choice[_choice == 'left'] = 0
    _choice[_choice == 'right'] = 1

    _reward = q_choice_outcome.fetch('outcome', order_by='trial') == 'hit'
    reward_history = np.zeros([2, len(_reward)])  # .shape = (2, N trials)
    for c in (0, 1):
        reward_history[c, _choice == c] = (_reward[_choice == c] == True).astype(int)
    choice_history = np.array([_choice]).astype(int)  # .shape = (1, N trials)
    
    # -- ITI --
    # All previous models has an effective ITI of constant 1.
    # For Ulises RNN model, an important prediction is that values in line attractor decay over (actual) time with time constant tau_CANN.
    # This will take into consideration (1) different ITI of each trial, and (2) long effective ITI after ignored trials.
    # Thus for CANN model, the ignored trials are also removed, but they contribute to model fitting in increasing the ITI.
    
    if (len(ephys.TrialEvent & q_choice_outcome) 
        and len(dj.U('trial') & (ephys.TrialEvent & q_choice_outcome)) == len(dj.U('trial') & (experiment.TrialEvent & q_choice_outcome))):
        # Use NI times (trial start and trial end) if (1) ephys exists (2) len(ephys) == len(behavior)
        trial_start = (ephys.TrialEvent & q_choice_outcome & 'trial_event_type = "bitcodestart"'
                       ).fetch('trial_event_time', order_by='trial').astype(float)
        trial_end = (ephys.TrialEvent & q_choice_outcome & 'trial_event_type = "trialend"'
                     ).fetch('trial_event_time', order_by='trial').astype(float)
        iti = trial_start[1:] - trial_end[:-1]  # ITI [t] --> ITI between trial t-1 and t
    else:  # If not ephys session, we can only use PC time (should be fine because this fitting is not quite time-sensitive)
        bpod_start_global = (experiment.SessionTrial & q_choice_outcome
                             ).fetch('start_time', order_by='trial').astype(float)  # This is (global) PC time
        bitcodestart_local = (experiment.TrialEvent & q_choice_outcome & 'trial_event_type = "bitcodestart"'
                              ).fetch('trial_event_time', order_by='trial').astype(float)
        trial_end_local = (experiment.TrialEvent & q_choice_outcome & 'trial_event_type = "trialend"'
                           ).fetch('trial_event_time', order_by='trial').astype(float)
        if bitcodestart_local and trial_end_local:
            iti = (bpod_start_global[1:] + bitcodestart_local[1:]) - (bpod_start_global[:-1] + trial_end_local[:-1])
        else:
            iti = bpod_start_global[1:] - bpod_start_global[:-1]
        
    iti = np.hstack([0, iti])  # First trial iti is irrelevant
    
    # -- p_reward --
    q_p_reward = q_choice_outcome.proj() * experiment.SessionBlock.WaterPortRewardProbability & session_key
    p_reward = np.vstack([(q_p_reward & 'water_port="left"').fetch('reward_probability', order_by='trial').astype(float),
                          (q_p_reward & 'water_port="right"').fetch('reward_probability', order_by='trial').astype(float)])

    return choice_history, reward_history, iti, p_reward, q_choice_outcome
