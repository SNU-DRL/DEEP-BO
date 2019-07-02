import numpy as np
import time
from ws.shared.logger import *

from ws.shared.hp_cfg import HyperparameterConfiguration
from ws.hpo.utils.converter import VectorGridConverter
from ws.hpo.utils.grid_gen import *
from ws.hpo.utils.one_hot_grid import OneHotVectorTransformer

class SearchHistory(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.reset()       

    def reset(self):
        self.complete = np.arange(0)
        self.candidates = np.setdiff1d(np.arange(self.num_samples), self.complete)
        
        self.observed_errors = np.ones(self.num_samples)
        self.terminal_record = np.zeros(self.num_samples)

    def get_candidates(self, use_interim=True):
        if use_interim:
            return self.candidates
        else:
            # select candidates by elements which have 0 of terminal record
            candidates = np.where(self.terminal_record == 0)[0]
            return candidates

    def get_completes(self, use_interim=True):
        if use_interim:
            return self.complete
        else:
            completes = np.where(self.terminal_record == 1)[0]
            return completes

    def update_error(self, model_index, test_error, use_interim=False):
        if not model_index in self.complete:
            self.candidates = np.setdiff1d(self.candidates, model_index)
            self.complete = np.append(self.complete, model_index)
            
        self.observed_errors[model_index] = test_error

        if use_interim == False:
            self.terminal_record[model_index] = 1
        else:
            self.terminal_record[model_index] = 0

    def get_errors(self, type_or_id, use_interim=True):
        
        if type_or_id == "completes":
            c = self.get_completes(use_interim)
            return self.observed_errors[c]
        elif type_or_id == "all":
            return self.observed_errors
        else:
            return self.observed_errors[type_or_id]

    def expand(self, hpv):
        # TODO: check hyperparams are existed
        model_index = self.num_samples # assign new model index
        self.num_samples += 1
        self.complete = np.append(self.complete, [model_index], axis=0)
        self.observed_errors = np.append(self.observed_errors, [1.0], axis=0)
        self.terminal_record = np.append(self.terminal_record, [0], axis=0) 
        #debug("Error space expanded: {}".format(len(self.observed_errors)))
        return model_index


class GridSamplingSpace(SearchHistory):

    def __init__(self, name, grid, hpv, hp_config, one_hot=False):

        self.name = name
        if type(hp_config) == dict:
            self.hp_config = HyperparameterConfiguration(hp_config)
        else:
            self.hp_config = hp_config

        self.hpv = hpv
        if one_hot == True:
            self.grid = self.get_one_hot_grid()
        else:
            self.grid = np.asarray(grid)
        
        super(GridSamplingSpace, self).__init__(len(hpv))

    def get_size(self):
        return len(self.hpv)

    def get_name(self):
        return self.name

    def get_hp_config(self):
        return self.hp_config

    def get_params(self):
        return self.hp_config.param_order

    def get_grid_dim(self):
        return self.grid.shape[1]

    def get_grid(self, type_or_index='all', use_interim=False):
        if type_or_index == "completes":
            completes = self.get_completes(use_interim)
            #debug("index of completes: {}".format(completes))
            return self.grid[completes, :]
        elif type_or_index == "candidates":
            candidates = self.get_candidates(use_interim)
            #debug("index of candidates: {}".format(candidates))
            return self.grid[candidates, :]        
        elif type_or_index != 'all':
            return self.grid[type_or_index]
        else:
            return self.grid

    def get_hpv(self, index=None):
        if index != None:
            params = self.hp_config.param_order
            args = self.hpv[index]
            hpv = {}
            for i in range(len(params)):
                p = params[i]
                hpv[p] = args[i]
            return hpv
        else:
            return self.hpv

    def expand(self, hpv):
        cvt = VectorGridConverter(self.hpv, self.get_candidates(), self.hp_config)
        grid_vec = cvt.to_grid_vector(hpv)
        
        self.hpv = np.append(self.hpv, [hpv], axis=0)
        self.grid = np.append(self.grid, [grid_vec], axis=0)
        debug("Sampling space expanded: {}".format(len(self.hpv))) 
        return super(GridSamplingSpace, self).expand(hpv)

    def get_one_hot_grid(self):
        grid = []
        num_samples = self.get_size()
        for i in range(num_samples):
            s = self.get_hpv(i)
            c = self.get_hp_config()
            t = OneHotVectorTransformer(c)
            e = t.transform(s)
            grid.append(np.asarray(e))
        return np.asarray(grid) 


class SurrogateSamplingSpace(GridSamplingSpace):

    def __init__(self, lookup, one_hot=False):

        self.grid = lookup.get_all_sobol_vectors()
        self.hpv = lookup.get_all_hyperparam_vectors()

        super(SurrogateSamplingSpace, self).__init__(lookup.data_type, 
                                                    self.grid, self.hpv, 
                                                    lookup.hp_config,
                                                    one_hot=one_hot)
        # preloaded results
        self.test_errors = lookup.get_all_test_errors()
        self.exec_times = lookup.get_all_exec_times()
        self.lookup = lookup

    # For search history 
    def update_error(self, model_index, test_error=None, use_interim=False):
        if test_error is None:
            test_error = self.test_errors[model_index]
        super(GridSamplingSpace, self).update_error(model_index, test_error, use_interim)

    def get_errors(self, type_or_id, use_interim=False):
        if type_or_id == "completes":
            c = self.get_completes(use_interim)
            return self.test_errors[c]
        elif type_or_id == "all":
            return self.test_errors
        else:
            return self.test_errors[type_or_id]

    def get_exec_time(self, index=None):
        if index != None:
            return self.exec_times[index]
        else:
            return self.exec_times

    def expand(self, hpv):
        # return approximated index instead of newly created index
        cvt = VectorGridConverter(self.hpv, self.get_candidates(), self.hp_config)
        idx, err_distance = cvt.get_nearby_index(hpv)

        debug("Distance btw selection and surrogate: {}".format(err_distance))
        return idx


class RemoteSamplingSpace(SearchHistory):
    def __init__(self, name, proxy):
        self.space = proxy
        
        num_samples = proxy.get_num_samples()
        self.name = "remote_{}".format(name)

        return super(RemoteSamplingSpace, self).__init__(num_samples)

    def get_name(self):
        return self.name

    def get_grid_dim(self):
        return len(self.get_params())

    def get_hp_config(self):
        return self.space.hp_config

    def get_params(self):
        return self.space.hp_config["param_order"]

    def get_grid(self, type_or_index, use_interim=False):
        return np.asarray(self.space.get_grid(type_or_index, use_interim))

    def get_hpv(self, index=None):
        if index == None:
            return self.space.get_vector('all')
        else:
            return self.space.get_vector(index)

    # For history
    def get_candidates(self, use_interim=True):
        return np.asarray(self.space.get_candidates(use_interim))

    def get_completes(self, use_interim=True):
        return np.asarray(self.space.get_completes(use_interim))

    def update_error(self, model_index, test_error, interim=False):
        self.space.update_error(model_index, test_error, interim)

    def get_errors(self, type_or_id, interim=False):
        return np.asarray(self.space.get_error(type_or_id, interim))

    def expand(self, hpv):
        self.space.expand(hpv)
        