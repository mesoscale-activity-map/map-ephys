from pipeline import ephys
from pipeline import tracking
from pipeline import experiment
from pipeline.plot import behavior_plot
from pipeline.mtl_analysis import helper_functions
from pipeline import lab
import datajoint as dj

from scipy import signal
from scipy import optimize

import matplotlib.pyplot as plt
import numpy as np
#%%
water='DL004'
date='2021-03-08'
subject, session = helper_functions.water2subject(water, date)