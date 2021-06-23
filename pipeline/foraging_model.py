

@schema
ModelClass(dj.Lookup):
    definition = """
    model_class: varchar(32)  # e.g. LossCounting, RW1972_epsi
    ---
    desc='': varchar(100)
    """
@schema
ModelParam(dj.Lookup):
    definition = """
    model_param: varchar(32)  # e.g. learn_rate, epsilon, w_tau1
    ---
    param_notation: varchar(32)  # '$\\tau_1$'
    """
@schema
Model(dj.Manual):
    definition = """
    model_id: int
    ---
    -> ModelClass
    param_str: varchar(100)  # 'tau1', 'tau2', 'w_tau1', 'softmax_temperature'
    is_bias: bool
    param_count: int
    desc='': varchar(1000)
    """
    class Param(dj.Part):
        definition = """
        -> master
        -> ModelParam
        ---
        param_lower_bound: float
        param_higher_bound: float
        """
@schema
class FittedSession(dj.Compute):
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
    