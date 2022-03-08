import os
import logging
import datajoint as dj
import numpy as np

from pipeline import lab, experiment, ephys, ccf, histology
from pipeline.fixes import schema

os.environ['DJ_SUPPORT_FILEPATH_MANAGEMENT'] = "TRUE"

log = logging.getLogger(__name__)
log.setLevel('INFO')

"""
Some of the histology.ElectrodeCCFPosition.ElectrodePosition are flipped along the left-right axis due to an error in preprocessing
"""

@schema
class OriginalElectrodePosition(dj.Manual):
    definition = str(histology.ElectrodeCCFPosition.ElectrodePosition.heading)