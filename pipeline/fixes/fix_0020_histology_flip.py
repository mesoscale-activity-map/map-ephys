import os
import logging
import datajoint as dj
import numpy as np

from pipeline import lab, ccf, histology
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

    class ElectrodePosition(dj.Part):
        definition = """
        -> master
        -> lab.ElectrodeConfig.Electrode
        -> ccf.CCF
        ---
        mri_x=null: float  # (mm)
        mri_y=null: float  # (mm)
        mri_z=null: float  # (mm)
        """

@schema
class FixedElectrodePosition(dj.Manual):
    definition = str(histology.ElectrodeCCFPosition.ElectrodePosition.heading)

    class ElectrodePosition(dj.Part):
        definition = """
        -> master
        -> lab.ElectrodeConfig.Electrode
        -> ccf.CCF
        ---
        mri_x=null: float  # (mm)
        mri_y=null: float  # (mm)
        mri_z=null: float  # (mm)
        """

def fix_hist_key(key):
    """
    This fix applies to sessions ingested with the BehaviorIngest's make() only,
    as opposed to BehaviorBpodIngest (for Foraging Task)
    """
    
    key = (histology.Session & key).fetch1()
    FixedElectrodePosition.insert()
    
if __name__ == '__main__':
    fix_photostim_trial()