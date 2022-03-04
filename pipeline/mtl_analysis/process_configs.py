import numpy as np
import utils
import pandas as pd
def process_configs(configs, dfs):

    # if 'model_name' not in configs.keys():
    #     configs['model_name'] = ''
    # if 'mode' not in configs.keys():
    #     configs['mode'] = 'default'
    
    if 'center' not in configs.keys():
        configs['center'] = (len(configs['only_sess']) == 1)


    if 'seq_length' not in configs.keys():
        configs['seq_length'] = 1 + (configs['upper_frame']+1 - configs['lower_frame'])//configs['downsamp']
    else:
        if 'downsamp' in configs.keys():
            assert configs['seq_length'] == 1 + (configs['upper_frame']+1 - configs['lower_frame'])//configs['downsamp']
    assert configs['seq_length']>0

    fs = utils.rescale_list(np.arange(configs['lower_frame'],configs['upper_frame']+1,1), configs['seq_length'], 0, configs['upper_frame']-configs['lower_frame'],False,configs['random_frame_sampling'])
    configs['_FrameListInUse'] = [int(f) for f in fs]





    assert len(set(dfs['val'].index)&set(dfs['train'].index))==0
    if 'test' in dfs.keys():
        assert len(set(dfs['val'].index)&set(dfs['test'].index))==0
        assert len(set(dfs['test'].index)&set(dfs['train'].index))==0

    allSess = set(dfs['train'].sessID)
    allSessValTest = allSess.union(set(dfs['val'].sessID),set(dfs['test'].sessID))
    assert allSessValTest == allSess
    configs['numSess'] = len(allSess)



    # for session non-specific encoder.
    if '__model_sessID2Num' not in configs.keys():
        configs['__model_sessID2Num'] = {}
        for i,sessID in enumerate(sorted(list(allSess))):
            configs['__model_sessID2Num'][sessID] = i



