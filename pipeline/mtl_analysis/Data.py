import os, sys, glob, copy, cv2
import numpy as np
import utils
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from process_configs import process_configs
import pandas as pd
class FrameBaseDataset(Dataset):
    def __init__(self, configs, df):
        self.configs = configs
        self.df = df


    def get_single_trial(self, index):
        sequence = np.zeros((
                self.configs['seq_length'], 
                len(self.configs['chans']),
                *self.configs['image_shape']
                ))


        row = self.df.iloc[index]
        uid = row.name
        sessID = row.sessID



        frames_all = []
        
        frames_all = os.listdir(row['FramesDir'])
        # print(row['FramesDir'],frames_all)
        frames_all = sorted(frames_all, key=utils.filename2frameNum)
        frames_all = [os.path.join(row['FramesDir'], f) for f in frames_all]

        # print('same', sorted(glob.glob(os.path.join(self.prefix_frames_dir,row['FramesDir'], '*'+chan+'*'+suffix))) == frames_all) 
        len_frames_all = len(frames_all)
        assert len_frames_all == utils.filename2frameNum(frames_all[-1])
        
        frames = utils.rescale_list(
            frames_all,
            self.configs['seq_length'],
            int(row['reference_point'] + self.configs['lower_frame']),
            int(row['reference_point'] + self.configs['upper_frame']),
            True,
            self.configs['random_frame_sampling'])


        remove_info = None # This is to deal with some problem in MAP's video recording: there is a part needs to be cropped. If in your case you don't need to remove anything, set "remove_info" to be None. In MAP's case, set it to be {'coord':(22, 0), 'mean':None, 'mode':'crop'}
        sequence[:,0,:,:] = np.array([
            process_image(
                f,
                (*self.configs['image_shape'],1),
                transpose = False,
                remove_info=remove_info
                ) for f in frames])

        return sequence, uid,  sessID

def process_image(
    image,
    target_shape,
    *,
    transpose=False,
    remove_info = None
    ):
    """Given an image, process it and return the array."""
    # Load the image.



    h, w, c = target_shape
    x = cv2.imread(image,0)
    # x = mpimg.imread(image)[:,:,0]
    if x is None:
        print(image)

    if transpose:
        x = np.transpose(x,(1,0))
    h_origin, w_origin = x.shape

    if remove_info is not None:
        coord, mean, mode = remove_info['coord'], remove_info['mean'], remove_info['mode']
        if mode == 'crop':
            x = x[int(coord[0]/h*h_origin+0.5):]


    if (x.shape[0] != h) or (x.shape[1] != w):
        x = cv2.resize(x,(w,h))

    assert x.shape[0] == h
    assert x.shape[1] == w

    x = (x / 255.).astype(np.float32)

    if remove_info is not None:
        if mode != 'crop':
            x[:coord[0],coord[1]:] = mean

    return x



class MyDataset(FrameBaseDataset):
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sequence, uid,  session_code = self.get_single_trial(index)
        return sequence, uid,  session_code



class Data:

    def __init__(self, configs,if_process_configs=False,SC_uid_list_filter=None, verbose=1,dfs=None):
        self.SC_uid_list_filter = SC_uid_list_filter
        self.configs = configs

        if dfs is None:
            self.get_dfs()
        else:
            self.dfs = dfs


        self.dfs = self.deal_filter()

        if if_process_configs:
            process_configs(self.configs,self.dfs)

        for g in self.dfs.keys():
            df = self.dfs[g]
            print('{}  num trials {}'.format(g, len(df))) if verbose else None
            print('------') if verbose else None
    def get_dfs(self):
        self.df_all = pd.read_csv(self.configs['pandas_csv_path'])
        self.configs['only_sess'] = sorted(list(set(self.df_all.sessID.tolist())))


        if len(self.configs['only_sess']) == 1:
            idx = np.arange(len(self.df_all))
            seed_base = 0
            seed_offset = self.configs['random_seed_offset']
            seed = seed_base + seed_offset
            np.random.seed(seed)
            np.random.shuffle(idx)
            size = self.configs['sess_test_num']
            self.dfs = {}
            self.dfs['val'] =  self.df_all.iloc[idx[:size]]
            self.dfs['test'] =  self.df_all.iloc[idx[size:size*2]]
            self.dfs['train'] =  self.df_all.iloc[idx[size*2:]]
        else:
            self.dfs = {'train':pd.DataFrame(),'test':pd.DataFrame(),'val':pd.DataFrame()}
            self.configs['fact_only_sess'] = sorted(list(set(self.df_all.sessID)))
            for i,sess in enumerate(self.configs['fact_only_sess']): 
                df_ = self.df_all.loc[self.df_all.sessID==sess]
                idx = np.arange(len(df_))
                seed_base = i+10 #int(utils.get_sess_str(df_.sessID[0])) #rozmar - get_sess_str was missing, added this enumerate instead
                seed_offset = 42
                seed = seed_base + seed_offset
                np.random.seed(seed)
                np.random.shuffle(idx)
                print(idx[:5])
                size = self.configs['sess_test_num']
                self.dfs['val'] =  pd.concat((self.dfs['val'], df_.iloc[idx[:size]]))
                self.dfs['test'] =  pd.concat((self.dfs['test'], df_.iloc[idx[size:size*2]]))
                self.dfs['train'] =  pd.concat((self.dfs['train'], df_.iloc[idx[size*2:]]))

    def do_dataloader(self, groups =['train','val','test']):
        self.dataset = {}
        self.data_loader = {}
        if 'train' in groups:
            self.dataset['train','train'] = MyDataset(
                self.configs,
                self.dfs['train']
                )
            # self.dataset['train','train'] = MyDataset()
            self.data_loader['train','train'] = DataLoader(
                dataset=self.dataset['train','train'],
                batch_size=self.configs['bs'],
                shuffle=True,
                num_workers = self.configs['num_workers']
                )
            '''
            The first train means "training set", and the second "train" means "during the training process.".
            ''' 
        for g in groups:
            self.dataset[g,'eval'] = MyDataset(
                self.configs,
                self.dfs[g]
                )
            # self.dataset[g,'eval'] = MyDataset()
            self.data_loader[g,'eval'] = DataLoader(
                dataset=self.dataset[g,'eval'],
                batch_size=self.configs['bs']//2, 
                shuffle=False,
                num_workers = self.configs['num_workers']
                )


    def deal_filter(self, uid_list=None, only_basic=False):
        if "hold_out_list" in self.configs.keys():
            assert type(self.configs["hold_out_list"]) == list
            if any(self.configs["hold_out_list"]):
                for g in self.dfs.keys():
                    self.dfs[g] = self.dfs[g].loc[~self.dfs[g]['Mouse'].isin(self.configs["hold_out_list"])]
                    if 'num' in self.dfs[g].keys():
                        del self.dfs[g]['num']
                        self.dfs[g] = utils.add_num_col(self.dfs[g])
                    print('Mice in', g, str(set(self.dfs[g]['Mouse'])))


        if 'only_sess' in self.configs.keys():
            assert type(self.configs["only_sess"]) == list

            # Fixing an old version:
            if '/' in self.configs["only_sess"][0]:
                assert self.configs['experiment'] == 'E2'
                ss = ['_'.join(s.split('/')[-2:]) for s in self.configs["only_sess"]]
                self.configs["only_sess"] = ss
            #########################
            for g in self.dfs.keys():
                self.dfs[g] = self.dfs[g].loc[self.dfs[g]['sessID'].isin(self.configs['only_sess'])]
                if 'num' in self.dfs[g].keys():
                    del self.dfs[g]['num']
                    self.dfs[g] = utils.add_num_col(self.dfs[g])

        if uid_list is not None:
            assert type(uid_list) == list
            for g in self.dfs.keys():
                self.dfs[g] = self.dfs[g].loc[self.dfs[g].index.isin(uid_list)]
                if 'num' in self.dfs[g].keys():
                    del self.dfs[g]['num']
                    self.dfs[g] = utils.add_num_col(self.dfs[g])

        if only_basic:
            return self.dfs

        groups = list(self.dfs.keys())
        for g in groups:
            df = self.dfs[g]
            # df = df.loc[df.Success >= 0]
            # df = df.loc[df.LickRight >= 0]
            self.dfs[g] = df

        return self.dfs

