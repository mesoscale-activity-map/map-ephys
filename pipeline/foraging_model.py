import datajoint as dj
from . import experiment
from . import get_schema_name

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
            # No bias (1-8)
            ['LossCounting', ['loss_count_threshold_mean', 'loss_count_threshold_std'], [0, 0], [40, 10]],
            ['RW1972_epsi', ['learn_rate', 'epsilon'], [0, 0], [1, 1], 'SuttonBarto: epsilon'],
            ['LNP_softmax', ['tau1', 'softmax_temperature'], [1e-3, 1e-2], [100, 15], 'Sugrue2004, Corrado2005: one tau'],
            ['LNP_softmax', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature'], [1e-3, 1e-1, 0, 1e-2], [15, 40, 1, 15], 'Corrado2005, Iigaya2019: two taus'],
            ['RW1972_softmax', ['learn_rate', 'softmax_temperature'], [0, 1e-2], [1, 15]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature'], [0, 0, 1e-2], [1, 1, 15], 'RL: rew, unrew, softmax'],
            ['Bari2019', ['learn_rate', 'forget_rate', 'softmax_temperature'], [0, 0, 1e-2], [1, 1, 15], 'RL: chosen, unchosen, softmax'],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature'], [0, 0, 0, 1e-2], [1, 1, 1, 15], 'RL: rew, unrew, unchosen, softmax'],

            # With bias (9-15)
            ['RW1972_epsi', ['learn_rate', 'epsilon', 'biasL'], [0, 0, -0.5], [1, 1, 0.5]],
            ['LNP_softmax', ['tau1', 'softmax_temperature', 'biasL'], [1e-3, 1e-2, -5], [100, 15, 5]],
            ['LNP_softmax', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature', 'biasL'], [1e-3, 1e-1, 0, 1e-2, -5], [15, 40, 1, 15, 5]],
            ['RW1972_softmax', ['learn_rate', 'softmax_temperature', 'biasL'], [0, 1e-2, -5], [1, 15, 5]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature', 'biasL'], [0, 0, 1e-2, -5], [1, 1, 15, 5]],
            ['Bari2019', ['learn_rate', 'forget_rate', 'softmax_temperature', 'biasL'], [0, 0, 1e-2, -5], [1, 1, 15, 5]],
            ['Hattori2019', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature', 'biasL'], [0, 0, 0, 1e-2, -5], [1, 1, 1, 15, 5]],

            # With bias and choice kernel (16-21)
            ['LNP_softmax_CK', ['tau1', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [1e-3, 1e-2, -5, 0, 1e-2], [100, 15, 5, 1, 20]],
            ['LNP_softmax_CK', ['tau1', 'tau2', 'w_tau1', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [1e-3, 1e-1, 0, 1e-2, -5, 0, 1e-2], [15, 40, 1, 15, 5, 1, 20]],
            ['RW1972_softmax_CK', ['learn_rate', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 1e-2, -5, 0, 1e-2], [1, 15, 5, 1, 20]],
            ['Hattori2019_CK', ['learn_rate_rew', 'learn_rate_unrew', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 0, 1e-2, -5, 0, 1e-2], [1, 1, 15, 5, 1, 20]],
            ['Bari2019_CK', ['learn_rate', 'forget_rate', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 0, 1e-2, -5, 0, 1e-2], [1, 1, 15, 5, 1, 20]],
            ['Hattori2019_CK', ['learn_rate_rew', 'learn_rate_unrew', 'forget_rate', 'softmax_temperature', 'biasL', 'choice_step_size', 'choice_softmax_temperature'],
             [0, 0, 0, 1e-2, -5, 0, 1e-2], [1, 1, 1, 15, 5, 1, 20]],
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
