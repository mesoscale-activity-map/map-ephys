import datajoint as dj
import ephys, experiment
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

schema = dj.schema(dj.config['analysis.database'], locals())

@schema
class PSTH(dj.computed):
    definition = """
    -> ephys.experiment.BehaviorTrial
    ---
    psth  : float          # psth aligned on go cue
    """