import os

import pandas as pd
import numpy as np

from ws.shared.read_cfg import read_hyperparam_config
from ws.shared.logger import *

LOOKUP_DIR = './lookup/'
def check_lookup_existed(name, lookup_dir=LOOKUP_DIR):
    if name.endswith('.json'):
        name = name[:-4]
    for csv in os.listdir(lookup_dir):
        if str(csv) == '{}.csv'.format(name):
            return True
    return False
def load(data_type, lookup_dir=LOOKUP_DIR, config_folder='hp_conf/', grid_order=None):
    grid_shuffle = False
    if grid_order == 'shuffle':
        grid_shuffle = True
    
    csv_path = lookup_dir + str(data_type) + '.csv'
    csv_data = pd.read_csv(csv_path)

    cfg_path = config_folder + str(data_type) + '.json'
    #debug("lookup load: {} config path: {}".format(data_type, cfg_path))
    cfg = read_hyperparam_config(cfg_path)

    num_epochs = 15
    if data_type == 'CIFAR10-VGG' or data_type == 'CIFAR100-VGG':
        num_epochs = 50
    
    if hasattr(cfg, 'num_epoch'):
        num_epochs = cfg.num_epoch
    
    if data_type == 'CIFAR10-ResNet':
        loader = CifarResnetSurrogateLoader(data_type, csv_data, cfg)
    else:
        loader = LookupDataLoader(data_type, csv_data, cfg, num_epochs, grid_shuffle)
    return loader


class LookupDataLoader(object):
    ''' Load lookup table data as pandas DataFrame object'''
    def __init__(self, dataset_type, surrogate, hp_cfg, num_epochs, grid_shuffle):
        self.data = surrogate
        if grid_shuffle is True:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            
        self.hp_config = hp_cfg
        self.data_type = dataset_type
        
        self.begin_index = 1
        self.num_epochs = num_epochs
        self.num_hyperparams = len(hp_cfg.get_param_names())

    def get_all_hyperparam_vectors(self):
        start_index = self.begin_index
        end_index = start_index + self.num_hyperparams
        hp_grid = self.data.ix[:, start_index:end_index].values  # hyperparameter vectors

        return hp_grid

    def get_all_test_acc_per_epoch(self, end_epoch=None):
        
        start_index = self.begin_index + self.num_hyperparams
        if end_epoch is None or end_epoch > self.num_epochs:            
            end_epoch = self.num_epochs
        end_index = start_index + end_epoch 
        accs = self.data.ix[:, start_index:end_index]  # accuracy at each epoch

        if hasattr(self.hp_config, 'metric'):
            if self.hp_config.metric == 'perplexity':
                # XXX:handle perplexity metric
                max_perplexity = 1000.0
                perplexities = accs
                accs = (max_perplexity - perplexities) / max_perplexity       
        return accs

    def get_all_test_errors(self, end_epoch=None):
        vals = self.get_all_test_acc_per_epoch(end_epoch)
        sorted_vals = np.sort(vals)  # sorted accuracies
        test_error_index = self.num_epochs - 1
        if end_epoch is not None: 
            if end_epoch > 0 and end_epoch <= self.num_epochs:
                test_error_index = end_epoch - 1

        fin_vals = sorted_vals[:, test_error_index]  # accuracy when training finished
        fin_loss = 1 - fin_vals  # final error
        return fin_loss

    def get_all_exec_times(self, end_epoch=None):

        if end_epoch is None or end_epoch > self.num_epochs:
            end_epoch = self.num_epochs        
        time_col_index = self.begin_index + self.num_hyperparams + end_epoch  #25
        dur = self.data.ix[:, time_col_index].values  # elapsed time
        dur = dur / self.num_epochs * end_epoch 
        return dur

    def get_all_sobol_vectors(self): 
        start_index = self.begin_index + self.num_hyperparams + self.num_epochs + 1
        end_index = start_index + self.num_hyperparams
        sobol_grid = self.data.ix[:, start_index:end_index].values 
        return sobol_grid

    def get_best_acc_of_trial(self):
        return np.max(self.get_all_test_acc_per_epoch().values, axis=1)


class CifarResnetSurrogateLoader(LookupDataLoader):
    ''' Load lookup table data as pandas DataFrame object'''
    def __init__(self, dataset_type, surrogate, hp_cfg):
        super(CifarResnetSurrogateLoader, self).__init__(dataset_type, surrogate, hp_cfg, 100, False)

    def get_all_test_acc_per_epoch(self, end_epoch=None):
        
        start_index = self.begin_index + self.num_hyperparams + 1 # skip durations + 0 epoch
        if end_epoch is None or end_epoch > self.num_epochs:            
            end_epoch = self.num_epochs
        end_index = start_index + end_epoch 
        accs = self.data.ix[:, start_index:end_index]  # accuracy at each epoch
        return accs

    def get_all_sobol_vectors(self): 
        start_index = self.begin_index + self.num_hyperparams + 202
        end_index = start_index + self.num_hyperparams
        sobol_grid = self.data.iloc[:, start_index:end_index].values 
        return sobol_grid

    def get_all_exec_times(self, end_epoch=None):
    
        if end_epoch is None or end_epoch > self.num_epochs:
            end_epoch = self.num_epochs        
        time_col_index = self.begin_index + self.num_hyperparams + 101
        dur = self.data.ix[:, time_col_index].values  # elapsed time
        dur = dur / self.num_epochs * end_epoch 
        return dur


