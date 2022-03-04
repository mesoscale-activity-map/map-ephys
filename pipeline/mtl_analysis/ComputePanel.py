import os,sys
import torch,datetime,pickle,glob,yaml
import numpy as np
from models import ModelPanel
from Data import Data
import pandas as pd
import copy
import torch.nn as nn
import time



def load_c_panel(train_name, iteration, configs_to_update={},del_configs_key=['seq_length']):
    configs_path = os.path.join('.\\log\\train_log\\',train_name,'configs.yaml')
    configs = yaml.load(open(configs_path,'r'), Loader=yaml.FullLoader)
    for k in del_configs_key:
        if k in configs.keys():
            del configs[k]
    
    configs.update({'save_log':0,'iterstart':iteration,'_device_type':'cpu','gpus':None})
    configs.update(configs_to_update)
    c_panel = ComputePanel(configs, auto_mode=False)
    c_panel.load_trained_model()
    c_panel.initiate_data()
    c_panel.initiate_center(no_file_then_cal=False)
    return c_panel

class ComputePanel(object):
    def __init__(self, configs = {}, data_dicts = None, auto_mode=True):


        configs_default = {}

        data_configs = {
            "pandas_csv_path":"/mnt/fs6/zwang71/FrameAutoEncoder/dataframe/SC026_080519.csv",
            "chans":["side"],
            #"lower_frame":-847,
            "lower_frame":-735,
            #"upper_frame":317,
            "upper_frame":735,
            "downsamp":10,
            "image_shape":[120, 112],
            "sess_test_num":32,
            'random_frame_sampling':True
        }

        train_configs = {
            'verbose':2,
            'do_sess_input':True,
            'node':'09',
            "bs":12,
            'mu_l2':0,
            'eval_gs':['val'],
            'recon_loss':'MSE',
            'num_workers':3,
            'gpus':'0',
            'save_log':0,
            'saved_model':None,
            'random_seed_offset':42,
            "lr":1e-4,
            "iterstart":0,
            "iterend":30000,
            "step_report":5000,
            "l2":0.0,
            "train_name_add":""
            }
        

        model_configs = {
            'model_name':'AeLd',
            "embed_size":16,
            "num_res_cnn":3,
            "num_res_chan":16,
            "num_res_chan2":32,
            "pool":4,
            "_linear1out":128,
            'd_hidden_size_list':[256,512]
            }

        
        
        configs_default.update(data_configs)
        configs_default.update(train_configs)
        configs_default.update(model_configs)





        self.configs = configs_default
        self.configs.update(configs)


        self.data_dicts = data_dicts
        if '_device_type' not in self.configs.keys():
            self.configs['_device_type'] = 'cuda'




        if auto_mode:
            self.initiate_data()
            # for x in self.data.data_loader['train','eval']:
            #     print('haha1')
            self.initiate_model()
            # for x in self.data.data_loader['train','eval']:
            #     print('haha2')
            if self.configs['save_log']:
                self.redirect()
            self.initiate_center()
            sys.stdout.flush()



    def initiate_model(self):
        self.m_panel = ModelPanel(self.configs)


    def redirect(self, main_log_dir=None):

        if main_log_dir is None:
            main_log_dir = os.path.join('log','train_log')
        if 'train_num' not in self.configs.keys():
            self.configs['train_num'] = -1

        if self.configs['train_num'] < 0:
            fn_list = sorted(
                        glob.glob(os.path.join(main_log_dir, '[0-9]*')),
                        key=lambda x: int(x.split('\\')[-1].split('-')[0])
                        )
            if len(fn_list) > 0:
                latest_fn = fn_list[-1].split('\\')[-1]
                self.configs['train_num'] = int(latest_fn.split('-')[0]) + 1
            else:
                self.configs['train_num'] = 0



        #self.configs['train_name'] = '-'.join((
        #    str(self.configs['train_num']),
        #    self.configs['train_name_add'],
        #    str(datetime.datetime.now()).split(' ')[0]
        #    ))
        self.configs['train_name'] = '-'.join((
            str(self.configs['train_num']),
            self.configs['train_name_add']
            ))
        self.configs['log_dir'] = os.path.join(main_log_dir, self.configs['train_name'])
        assert not os.path.isdir(self.configs['log_dir'])
        os.makedirs(self.configs['log_dir'])
        os.makedirs(os.path.join(self.configs['log_dir'],'checkpoints'))
        os.makedirs(os.path.join(self.configs['log_dir'],'quick_log'))
        print(self.configs['train_name'])
        sys.stdout = open(os.path.join(self.configs['log_dir'],'cmd_log.txt'), "w")
        sys.stderr = open(os.path.join(self.configs['log_dir'],'cmd_err.stderr'), "w")

        print(self.configs['train_name'])

        yaml.dump(
            self.configs,
            open(os.path.join(self.configs['log_dir'],'configs.yaml'),'w'),
            default_flow_style=False
            )

    def load_trained_model(self):

        iteration = self.configs['iterstart']
        log_dir_ = self.configs['log_dir'] = os.path.join('log','train_log',self.configs['train_name'])


        assert ' ' not in log_dir_
        fn_list = glob.glob(os.path.join(
            log_dir_, 'checkpoints','iter-'+str(iteration)+'-*'
            ))
        # print(log_dir_, 'checkpoints','iter-'+str(iteration)+'-*')
        assert len(fn_list) == 1

        self.configs['saved_model'] = fn_list[0]
        print(self.configs['saved_model'])
        self.m_panel = ModelPanel(self.configs)
        

    def initiate_data(self,SC_uid_list_filter = None, verbose=1, dfs=None):
        # SC_uid_list_filter is for further study on the firing rates. Some trial may not have good firing rates recorded, so we may need to drop those trials at that time. If it's just auto-encoder training then it probably doesn't matter. This parameter is used in generate_GLMHMM_data.py.
        self.data = Data(self.configs, if_process_configs=True,SC_uid_list_filter = SC_uid_list_filter,verbose=verbose, dfs=dfs)
        self.data.do_dataloader()

    def initiate_center(self, no_file_then_cal=True):

        if 'center' not in self.configs.keys():
            self.configs['center'] = False

        if not self.configs['center']:
            self.x_mean = 0
            return

        if self.configs['center']:
            if 'log_dir' in self.configs.keys():
                x_mean_path = os.path.join(self.configs['log_dir'],'x_mean.numpy')
                if os.path.isfile(x_mean_path):
                    print('loading the center')
                    self.x_mean = np.load(open(x_mean_path,'rb'))
                    print('center loaded')
                    return
            if no_file_then_cal:
                print('calculating the center')
                # self.get_xIn(g='train')
                # self.x_mean = self.xIn['train'].mean(axis=(0,1),keepdims=True)
                self.x_mean = self.calculate_x_mean()
                print(self.x_mean.shape)
                if self.configs['save_log']:
                    np.save(open(x_mean_path,'wb'),self.x_mean)
                print()
                print('center calculated')
            else:
                raise Exception("Can't find center file")

    def get_sess_num(self,sessionCodes):
        sessNum = []
        for sess in sessionCodes:
            sessNum.append(self.configs['__model_sessID2Num'][sess])
        sessNum = torch.tensor(sessNum).to(device=self.configs['_device_type'], dtype=torch.int)
        return sessNum
    def do_batch(self,backprop=True):
        # for x in self.data.data_loader['train','eval']:
        #     print('haha')
        try:
            xB, _, sessionCodes = self.data_load_iter.next()

        except StopIteration:
            self.data_load_iter = iter(self.data.data_loader['train','train'])
            xB, _, sessionCodes = self.data_load_iter.next()
        xB -= torch.tensor(self.x_mean)

        Input = xB.to(device=self.configs['_device_type'], dtype=torch.float)
        bs = Input.shape[0]
        bs_eff = bs*self.configs['seq_length']

        sessNum = self.get_sess_num(sessionCodes)
        Inputs = [
                Input,
                sessNum
            ]

        """ Reconstruction phase """
        X_out, mu = self.m_panel.model(*Inputs)

        recon_loss = nn.MSELoss()(X_out.view(bs_eff,-1), Input.view(bs_eff,-1))
        mu_loss = mu.view(bs_eff,-1).pow(2).sum(1).mean(0)
        if backprop:
            if self.configs['recon_loss'] == 'MSE':
                all_loss = recon_loss
            elif self.configs['recon_loss'] == 'BCE':
                all_loss = F.binary_cross_entropy(X_out.view(bs_eff,-1), Input.view(bs_eff,-1))
            if self.configs['mu_l2']:
                all_loss += self.configs['mu_l2']*mu_loss
            all_loss.backward()
            self.m_panel.opt.step() #regular optimizer from pytorch
            self.m_panel.reset_grad()



        return recon_loss.detach().cpu().numpy(), mu_loss.detach().cpu().numpy(), 0, 0




        


    def do_train(self):
        for g in ['train','val','test']:
            print(g, len(self.data.dfs[g]))
        self.data_load_iter = iter(self.data.data_loader['train','train'])

        recon_loss_list = []
        mu_loss_list = []
        D_loss_list = []
        G_loss_list = []

        self.m_panel.train()

        print('format: [recon_loss, mu_loss, D_loss, G_loss]')

        for i in range(self.configs['iterstart'],self.configs['iterend']+1):
            # if not i%self.configs['step_report']:
            #     utils.quickwrite(self.configs,'batch training for ' + str(self.configs['step_report']) +' steps: [loss]\n')
            self.m_panel.iteration = i
            self.m_panel.train()


            recon_loss, mu_loss, D_loss, G_loss = self.do_batch()


            print("[{:.3e}__{:.3e}__{:.3e}__{:.3e}]".format(
                float(recon_loss),
                float(mu_loss),
                float(D_loss),
                float(G_loss)
                ),end = ',')


            sys.stdout.flush()

            recon_loss_list.append(float(recon_loss))
            mu_loss_list.append(float(mu_loss))
            D_loss_list.append(float(D_loss))
            G_loss_list.append(float(G_loss))


            if not i%self.configs['step_report']:
                with torch.no_grad():
                    self.m_panel.eval()
                    o = '\n********************************************\n'
                    o += 'iteration='+str(i) + '\n'
                    result_dfs = {}

                    loss_str = {}
                    self.m_panel.eval()

                    for g in self.configs['eval_gs']:
                        loss_str[g] = self.evaluation(g, store_output=False)[0]
                        o += loss_str[g] + '\n'


                    loss_str['real-time-train'] = "rt-train[{:.3e}__{:.3e}__{:.3e}__{:.3e}]".format(
                        np.mean(recon_loss_list),
                        np.mean(mu_loss_list),
                        np.mean(D_loss),
                        np.mean(G_loss_list)
                        )
                      
                    o += loss_str['real-time-train'] + '\n'




                    print(o)
                    print('********************************************\n')

                    self.m_panel.train()



                    recon_loss_list = []
                    mu_loss_list = []
                    D_loss_list = []
                    G_loss_list = []
                    if self.configs['save_log']:
                        if self.configs['gpus'] is not None:
                            if len(self.configs['gpus']) > 1:
                                state_dict = self.m_panel.model.module.state_dict()
                            else:
                                state_dict = self.m_panel.model.state_dict()
                        else:
                            state_dict = self.m_panel.model.state_dict()

                        raw_fn = 'iter-{}-'.format(i) + loss_str['real-time-train'] + '-' + loss_str['val'] + '-' + self.configs['train_name']
                        model_fn = raw_fn + '.pt'
                        path = os.path.join(self.configs['log_dir'],'checkpoints',model_fn)
                        assert not os.path.isfile(path)
                        torch.save(state_dict,path)

                        path = os.path.join(self.configs['log_dir'],'quick_log',raw_fn+'.txt')
                        f= open(path,"w")
                        f.write('')
                        f.close() 

    def get_embed(self,g='val'):
        gs = ['train','val','test']
        for g in gs:
            self.evaluation(g=g,store_output=g,embed_only=True,eval_verbose = True)
        self.embed_all = np.concatenate([self.mu[g] for g in gs])
        self.uid_all = np.concatenate([self.uid[g] for g in gs])


    def get_xIn_xOut(self,g='val'):
        self.evaluation(g=g,store_output=g,embed_only=False,store_xOut=True, store_xIn=True)

    def evaluation(self, 
        g='val',
        store_output = None, 
        store_xOut=False, 
        store_xIn=False,
        eval_verbose = False,
        embed_only = False):

        self.m_panel.eval()

        if store_output:
            try:
                self.xOut
                self.mu
                self.xIn
                self.uid
            except:
                self.mu = {}
                self.xOut = {}
                self.xIn = {}
                self.uid = {}
            self.mu[store_output] = np.zeros((
                len(self.data.dfs[g]),
                self.configs['seq_length'],
                self.configs['embed_size']
                ))
            self.uid[store_output] = []
            if not embed_only:
                if store_xOut:
                    self.xOut[store_output] = np.zeros((
                        len(self.data.dfs[g]),
                        self.configs['seq_length'],
                        len(self.configs['chans']),
                        self.configs['image_shape'][0],
                        self.configs['image_shape'][1]
                        ))
                if store_xIn:
                    self.xIn[store_output] = np.zeros((
                        len(self.data.dfs[g]),
                        self.configs['seq_length'],
                        len(self.configs['chans']),
                        self.configs['image_shape'][0],
                        self.configs['image_shape'][1]
                        ))



        
         
        i_this = 0
        count = 0
        recon_loss_list = []
        mu_loss_list = []
        D_loss_list = []
        G_loss_list = []
        for xB, trialID, sessionCodes_b in self.data.data_loader[g,'eval']:
            time.sleep(0.5)
            xB -= torch.tensor(self.x_mean)
            if eval_verbose:
                print(count, end = ',')
                sys.stdout.flush()
            count+=1

            Input = xB.to(device=self.configs['_device_type'], dtype=torch.float)
            
            bs = xB.shape[0]
            bs_eff = bs*self.configs['seq_length']

            sessNum = self.get_sess_num(sessionCodes_b)
            if not embed_only:
                Inputs = [
                        Input,
                        sessNum
                    ]
                X_out, mu = self.m_panel.model(*Inputs)
                recon_loss = nn.MSELoss()(X_out.view(bs_eff,-1), Input.view(bs_eff,-1)).detach().cpu().numpy()
                mu_loss = mu.view(bs_eff,-1).pow(2).sum(1).mean(0)
                z_real = torch.randn(bs_eff, self.configs['embed_size']).to(dtype=mu.dtype,device=mu.device)
                z_fake = mu.view(bs_eff,-1)
                # z_fake = mu.view(bs_eff, self.configs['embed_size'])

                D_loss, G_loss = 0, 0
            
                recon_loss_list.append(float(recon_loss))
                mu_loss_list.append(float(mu_loss))
                D_loss_list.append(float(D_loss))
                G_loss_list.append(float(G_loss))
            else:
                mu = self.m_panel.model.encoder(Input)


            if store_output:
                self.mu[store_output][i_this:i_this+bs] = mu.detach().cpu().numpy()
                trialID_numpy = trialID.detach().cpu().numpy()
                for i in range(len(trialID_numpy)):
                    self.uid[store_output].append('-'.join([sessionCodes_b[i],str(trialID_numpy[i])]))
                if not embed_only:
                    if store_xOut:
                        self.xOut[store_output][i_this:i_this+bs] = X_out.detach().cpu().numpy() + self.x_mean
                    if store_xIn:
                        self.xIn[store_output][i_this:i_this+bs] = Input.detach().cpu().numpy() + self.x_mean
            i_this += bs
            assert bs == len(xB)
            if not embed_only:
                del X_out, mu, Input, Inputs


        if store_output:
            assert i_this == len(self.mu[store_output])

        if not embed_only:
            recon_loss, mu_loss, D_loss, G_loss = np.mean(recon_loss_list), np.mean(mu_loss_list), np.mean(D_loss_list), np.mean(G_loss_list)

            o = g + "[{:.3e}__{:.3e}__{:.3e}__{:.3e}]".format(recon_loss, mu_loss, D_loss, G_loss)


            return o, recon_loss, D_loss, G_loss
        else:
            return '',np.nan,np.nan,np.nan

    def get_xIn(self, g='val', store_output = None):
        try:
            self.xIn
        except:
            self.xIn = {}
        if store_output is None:
            store_output = g
        self.xIn[store_output] = np.zeros((
            len(self.data.dfs[g]),
            self.configs['seq_length'],
            len(self.configs['chans']),
            self.configs['image_shape'][0],
            self.configs['image_shape'][1]
            ))
        i_this = 0
        count = 0
        for xB, _, sessionCodes_b in self.data.data_loader[g,'eval']:
            count += 1
            print(count,end=',')
            Input = xB
            bs = xB.shape[0]
            bs_eff = bs*self.configs['seq_length']
            self.xIn[store_output][i_this:i_this+bs] = Input.detach().cpu().numpy()
            i_this += bs
            assert bs == len(xB)


    def calculate_x_mean(self):
        bs_list = []
        mean_list = []
        count = 0
        for xB, _, sessionCodes_b in self.data.data_loader['train','train']:
            count += 1
            print(count,end=',')
            sys.stdout.flush()
            xIn = xB.detach().cpu().numpy()
            bs = xIn.shape[0]
            mean_list.append(xIn.mean(axis=(0,1),keepdims=True))
            bs_list.append(bs)
            if np.sum(bs_list) > 20*len(self.configs['only_sess']):
                # may be enough to take these samples.
                break
            time.sleep(1)
        x_mean = np.average(np.array(mean_list),weights=bs_list,axis=0)
        if len(self.configs['only_sess']) == 1:
            return x_mean
        else:
            return np.mean(x_mean)