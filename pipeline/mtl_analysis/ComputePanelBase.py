import os,sys,glob,datetime




class ComputePanelBase():


    def redirect(self, configs, main_log_dir, affix):
        if self.configs[affix+'save_log']:
            if (affix+'train_num') not in configs.keys():
                configs[affix+'train_num'] = -1

            if configs[affix+'train_num'] < 0:
                fn_list = sorted(
                            glob.glob(os.path.join(main_log_dir, '[0-9]*')),
                            key=lambda x: int(x.split('/')[-1].split('-')[0])
                            )
                if len(fn_list) > 0:
                    latest_fn = fn_list[-1].split('/')[-1]
                    configs[affix+'train_num'] = int(latest_fn.split('-')[0]) + 1
                else:
                    configs[affix+'train_num'] = 0



            configs[affix+'train_name'] = '-'.join((
                str(configs[affix+'train_num']),
                configs[affix+'train_name_add'],
                str(datetime.datetime.now()).split(' ')[0]
                ))
            configs[affix+'log_dir'] = os.path.join(main_log_dir, configs[affix+'train_name'])
            assert not os.path.isdir(configs[affix+'log_dir'])
            os.makedirs(configs[affix+'log_dir'])
            os.makedirs(os.path.join(configs[affix+'log_dir'],'checkpoints'))
            os.makedirs(os.path.join(configs[affix+'log_dir'],'quick_log'))
            sys.stdout = open(os.path.join(configs[affix+'log_dir'],'cmd_log.txt'), "w")
            sys.stderr = open(os.path.join(configs[affix+'log_dir'],'cmd_err.stderr'), "w")

            print(configs[affix+'train_name'])

    def do_train(self,train_data_loader=None):
        if train_data_loader is None:
            train_data_loader = self.data_loader['train-shuffle']
        self.data_load_iter = iter(train_data_loader)

    def get_batch_loader_output(self,train_data_loader=None):

        if train_data_loader is None:
            train_data_loader = self.data_loader['train-shuffle']
        try:
            batch_loader_output = self.data_load_iter.next()
        except StopIteration:
            self.data_load_iter = iter(train_data_loader)
            batch_loader_output = self.data_load_iter.next()
            self.epoch += 1
        return batch_loader_output