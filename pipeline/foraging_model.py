import datajoint as dj
import numpy as np
from . import experiment, get_schema_name, foraging_analysis
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
            for param, lb, ub in zip(*model[1:4]):
                # The result table should save both effective and fixed params
                Model.Param.insert1(dict(model_id=model_id, model_param=param,
                                         param_lower_bound=lb, param_higher_bound=ub),
                                    skip_duplicates=True)

        return


@schema
class FittedSessionModel(dj.Computed):
    definition = """
    -> experiment.Session
    -> Model
    ---
    n_trials: int
    n_params: int
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

    key_source = (foraging_analysis.SessionTaskProtocol() & 'session_task_protocol = 100' & 'session_real_foraging'
                  ) * Model() #& (experiment.Session & 'session_date > "2021-01-01"')

    class FittedParam(dj.Part):
        definition = """
        -> master
        -> Model.Param
        ---
        fitted_value: float
        """
    
    class PredictiveChoiceProb(dj.Part):
        definition = """
        # Could be used to compute latent value Q (for models that have Q). Ignored trial skipped.
        -> master
        -> experiment.SessionTrial
        -> experiment.WaterPort
        ---
        choice_prob: float        
        """

    def make(self, key):
        choice_history, reward_history, p_reward, q_choice_outcome = get_session_history(key)
        model_str = (Model & key).fetch('fit_cmd')

        # --- Actual fitting ---
        model_comparison_this = BanditModelComparison(choice_history, reward_history, model=model_str)
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
                          ),
                     skip_duplicates=True, allow_direct_insert=True)

        # Insert fitted params
        for param, x in zip((Model.Param & key).fetch('model_param'), fit_result.x):
            self.FittedParam.insert1(dict(**key, model_param=param, fitted_value=x))
        
        # Insert predictive choice probability
        for water_port_idx, water_port in enumerate(['left', 'right']):
            key['water_port'] = water_port
            self.PredictiveChoiceProb.insert(
                [{**key, 'trial': trial_idx, 'choice_prob': this_prob} 
                 for trial_idx, this_prob in zip(q_choice_outcome.fetch('trial'), fit_result.predictive_choice_prob[water_port_idx])],
                skip_duplicates=True
                )
        

# ============= Helpers =============

def get_session_history(session_key, remove_ignored=True):
    # Fetch data
    q_choice_outcome = (experiment.WaterPortChoice.proj(choice='water_port')
                        * experiment.BehaviorTrial.proj('outcome', 'early_lick')
                        * experiment.SessionBlock.BlockTrial) & session_key  # Remove ignored trials
    if remove_ignored:
        q_choice_outcome &= 'outcome != "ignore"'

    # TODO: session QC (warm-up and decreased motivation etc.)

    # Formatting
    _choice = (q_choice_outcome.fetch('choice') == 'right').astype(int)    # 0: left, 1: right
    _reward = q_choice_outcome.fetch('outcome') == 'hit'
    reward_history = np.zeros([2, len(_reward)])  # .shape = (2, N trials)
    for c in (0, 1):
        reward_history[c, _choice == c] = (_reward[_choice == c] == True).astype(int)
    choice_history = np.array([_choice])  # .shape = (1, N trials)

    # p_reward (optional)
    q_p_reward = q_choice_outcome.proj() * experiment.SessionBlock.WaterPortRewardProbability & session_key
    p_reward = np.vstack([(q_p_reward & 'water_port="left"').fetch('reward_probability').astype(float),
                          (q_p_reward & 'water_port="right"').fetch('reward_probability').astype(float)])

    return choice_history, reward_history, p_reward, q_choice_outcome
