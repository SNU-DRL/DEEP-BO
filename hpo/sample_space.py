import numpy as np
import os
import copy
import json
import time
from ws.shared.logger import *
from ws.shared.hp_cfg import HyperparameterConfiguration

from hpo.utils.converter import *
from hpo.utils.grid_gen import *

from hpo.connectors.remote_space import RemoteParameterSpaceConnector


class ParameterSpace(object):
    def __init__(self, parameters):
        self.param_vectors = parameters
        num_samples = len(parameters)
        self.completions = np.arange(0)
        self.candidates = np.setdiff1d(np.arange(num_samples), self.completions)
        self.observed_errors = np.ones(num_samples)
        self.train_epochs = np.zeros(num_samples)
        
    def reset(self):
        num_samples = len(self.param_vectors)
        self.completions = np.arange(0)
        self.candidates = np.setdiff1d(np.arange(num_samples), self.completions)        
        self.observed_errors = np.ones(num_samples)
        self.train_epochs = np.zeros(num_samples)
        
    def get_candidates(self, use_interim=True):
        if use_interim:
            return self.candidates
        else:
            max_epochs = np.max(self.train_epochs)
            # select candidates by elements which have 0 of terminal record
            candidates = np.where(self.train_epochs < max_epochs)[0]
            return candidates

    def get_completions(self, use_interim=True):
        if use_interim:
            return self.completions
        else:
            max_epochs = np.max(self.train_epochs)
            completions = np.where(self.train_epochs == max_epochs)[0]
            return completions

    def get_search_order(self, sample_index):
        if sample_index in self.completions:
            search_order = self.completions.tolist()
            return search_order.index(sample_index)
        else:
            return None

    def get_train_epoch(self, sample_index):
        return self.train_epochs[sample_index]

    def update_error(self, sample_index, test_error, num_epochs=None):
        if not sample_index in self.completions:
            self.candidates = np.setdiff1d(self.candidates, sample_index)
            self.completions = np.append(self.completions, sample_index)

        if sample_index < len(self.observed_errors):    
            self.observed_errors[sample_index] = test_error
        else:
            raise ValueError("Invaid index: {}, size: {}".format(sample_index, len(self.observed_errors)))

        if num_epochs != None:
            self.train_epochs[sample_index] = num_epochs

    def get_errors(self, type_or_id="completions"):
        if type_or_id == "completions":
            c = self.get_completions()
            return self.observed_errors[c]
        elif type_or_id == "all":
            return self.observed_errors
        else:
            return self.observed_errors[type_or_id]

    def expand(self, indices):
        self.candidates = np.append(self.candidates, indices, axis=0)
        self.observed_errors = np.append(self.observed_errors, 
                                    [ None for i in range(len(indices))], axis=0)
        self.train_epochs = np.append(self.train_epochs, 
                                    [ 0 for i in range(len(indices))], axis=0) 
        return indices        

    def remove(self, sample_index):
        # check index in candidates or completions
        if not sample_index in self.completions:
            if not sample_index in self.candidates:
                debug('{} is not in candidate list'.format(sample_index))
            else:
                self.candidates = np.setdiff1d(self.candidates, sample_index)
        else:
            debug('Completed sample id {} can not be removed.'.format(sample_index))


class HyperParameterSpace(ParameterSpace):

    def __init__(self, name, hp_config_dict, hpv_list, 
                 space_setting=None):

        self.name = name
        self.hp_config = HyperparameterConfiguration(hp_config_dict)

        self.space_setting = space_setting
        self.prior_history = None
        if 'prior_history' in space_setting:
            self.prior_history = space_setting['prior_history']
        self.priors = []

        self.hp_vectors = copy.copy(hpv_list)
        self.initial_hpv = hpv_list

        super(HyperParameterSpace, self).__init__(self.one_hot_encode(self.hp_vectors))

    def reset(self):
        self.hp_vectors = copy.copy(self.initial_hpv)
        self.param_vectors = self.one_hot_encode(self.hp_vectors)
        super(HyperParameterSpace, self).reset()

        if self.prior_history != None:
            try:
                if len(self.priors) == 0:
                    hp_vectors = self.load('spaces', self.prior_history)
                    if len(hp_vectors) == 0:
                        raise IOError("No space information retrieved: {}".format(self.prior_history))
                    self.priors = self.extract_prior(hp_vectors, 'results', self.prior_history)
                
                self.preset()
                debug("The # of prior observations: {}".format(len(self.completions)))

            except Exception as ex:
                warn("Use of prior history failed: {}".format(ex))

    def save(self, save_type='npz'):
        # save hyperparameter vectors
        space = {}
        
        try:
            if not os.path.isdir("./spaces/"):
                os.mkdir("./spaces/")
            space['name'] = self.name
            if save_type == 'json':
                file_name = "spaces/{}.json".format(self.name)

                if type(self.hp_vectors) == list:
                    space['hpv'] = self.hp_vectors
                else:
                    space['hpv'] = self.hp_vectors.tolist()

                with open(file_name, 'w') as json_file:
                    json_file.write(json.dumps(space))
                
            elif save_type == 'npz':
                file_name = "spaces/{}.npz".format(self.name)
                np.savez_compressed(file_name, hpv=np.array(self.hp_vectors))
            else:
                raise ValueError("Unsupported save format: {}".format(save_type))
            debug("{} saved properly.".format(file_name))
        except Exception as ex:
            warn("Unable to save {}: {}".format(file_name, ex))

    def load(self, space_folder, space_name):
        if space_folder[-1] != '/':
            space_folder += '/'
        if not os.path.isdir(space_folder):
            raise IOError("{} folder not existed".format(space_folder))
        json_file = "{}{}.json".format(space_folder, space_name)
        npz_file = "{}{}.npz".format(space_folder, space_name)
        hp_vectors = []
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                space = json.load(f)
                if 'hpv' in space:
                    hp_vectors = space['hpv']
                else:
                    warn("Invalid space format!")
        elif os.path.exists(npz_file):
            space = np.load(npz_file)
            if 'hpv' in space:
                hp_vectors = space['hpv']
            else:
                warn("Invalid space format!")           
            space.close()
        else:
            raise IOError("{}{} file not exist".format(space_folder, space_name))
        return hp_vectors

    def preset(self):
        
        try:
            for k in self.priors:
                c = self.priors[k]
                if 'hyperparams' in c:
                    hpv = self.hp_config.to_typed_list(c['hyperparams'])
                    indices = self.expand(hpv) # XXX: expand() returns array
                    self.update_error(indices[0], c['observed_error'], c['train_epoch'])

        except Exception as ex:
            warn("Preset previous history failed:{}".format(ex))

    def extract_prior(self, hp_vectors, result_folder, result_name):
        completions = {}
        if result_folder[-1] != '/':
            result_folder += '/'
        result_path = "{}{}".format(result_folder, result_name)
        if not os.path.isdir(result_path):
            raise IOError("{} not found".format(result_path))

        for dirpath, dirnames, filenames in os.walk(result_path):
            for filename in [f for f in filenames if f.endswith(".json")]:
                result_file = os.path.join(dirpath, filename)
                debug("Priors will be from {}.".format(result_file))
                with open(result_file, 'r') as json_file:
                    results = json.load(json_file)
                    for k in results.keys():
                        r = results[k]
                        if 'model_idx' in r:
                            for i in range(len(r['model_idx'])):
                                idx = r['model_idx'][i]
                                if idx < len(hp_vectors):
                                    completions[idx] = {
                                        "hyperparams": hp_vectors[idx],
                                        "observed_error": r['error'][i],
                                        "train_epoch": r['train_epoch'][i]
                                    }
                        else:
                            raise ValueError("Invalid prior result format: {}".format(result_file))
                            
        return completions

    def get_size(self):
        return len(self.hp_vectors)

    def get_name(self):
        return self.name

    def get_hp_config(self):
        return self.hp_config

    def get_params_dim(self):
        return self.param_vectors.shape[1]

    def get_param_vectors(self, type_or_index='all', use_interim=False):
        if type(type_or_index) == str: 
            if type_or_index == "completions":
                completions = self.get_completions(use_interim)
                #debug("index of completions: {}".format(completions))
                return self.param_vectors[completions, :]
            elif type_or_index == "candidates":
                candidates = self.get_candidates(use_interim)
                #debug("index of candidates: {}".format(candidates))
                return self.param_vectors[candidates, :]        
            elif type_or_index == 'all':
                return self.param_vectors            
        else:
            return self.param_vectors[type_or_index]
 
    def get_hp_vectors(self):
            return self.hp_vectors # XXX:return array value

    def get_hpv_dict(self, index):
        params = self.hp_config.get_param_list()
        args = self.hp_vectors[index]
        hpv = {}
        for i in range(len(params)):
            p = params[i]
            hpv[p] = args[i]
        return hpv # XXX:return dictionary value

    def expand(self, hpv):
        # TODO:check hpv is valid. 
        
        # check dimensions
        hpv_list = hpv
        dim = len(np.array(hpv).shape)
        if dim == 1:
            hpv_list = [ hpv ]
        elif dim != 2:
            raise TypeError("Invalid hyperparameter vector")

        # XXX:assumed that hpv consisted with valid values.
        self.hp_vectors = np.append(self.hp_vectors, hpv_list, axis=0)

        cvt = VectorGridConverter(self.hp_config)
        param_list = []
        vec_indices = []
        vec_index = len(self.param_vectors) # get new model index
        for hpv in hpv_list:
            param_vec = cvt.to_norm_vector(hpv)            
            param_list.append(param_vec)
            vec_indices.append(vec_index)
            vec_index += 1
        self.param_vectors = np.append(self.param_vectors, param_list, axis=0)
        
        #debug("Search space expanded: {}".format(len(self.hp_vectors))) 
        return super(HyperParameterSpace, self).expand(vec_indices)

    def remove(self, sample_index):
        # remove item in self.hp_vectors and self.param_vectors
        if len(self.hp_vectors) > sample_index:
            # XXX:below makes that the index will be reset
            # FIXME:however, the size of those vectors can be largely increased.
            #self.hp_vectors = np.delete(self.hp_vectors, sample_index)
            #self.param_vectors = np.delete(self.param_vectors, sample_index)

            return super(HyperParameterSpace, self).remove(sample_index)
        else:
            warn("Index {} can not be removed.".format(sample_index))

    def one_hot_encode(self, hpv_list):
        # TODO:below loop can be faster using parallelization.
        params = []
        for i in range(len(hpv_list)):
            t = OneHotVectorTransformer(self.hp_config)
            e = t.transform(self.get_hpv_dict(i))
            params.append(np.asarray(e))
        return np.asarray(params) 


class SurrogatesSpace(HyperParameterSpace):

    def __init__(self, lookup):

        hpv_list = lookup.get_all_hyperparam_vectors()

        super(SurrogatesSpace, self).__init__(lookup.data_type,                                               
                                              lookup.hp_config.get_dict(),
                                              hpv_list)
        # preloaded results
        self.test_errors = lookup.get_all_test_errors()
        self.exec_times = lookup.get_all_exec_times()
        self.lookup = lookup
        self.num_epochs = lookup.num_epochs

    # For search history 
    def update_error(self, sample_index, test_error=None, num_epochs=None):
        if test_error is None:
            test_error = self.test_errors[sample_index]
        if num_epochs is None:
            num_epochs = self.num_epochs
        super(HyperParameterSpace, self).update_error(sample_index, test_error, num_epochs)

    def get_errors(self, type_or_id):
        if type_or_id == "completions":
            c = self.get_completions()
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
        cvt = VectorGridConverter(self.hp_config)
        idx, err_distance = cvt.get_nearby_index(self.get_candidates(), self.hp_vectors, hpv)

        debug("Distance btw selection and surrogate: {}".format(err_distance))
        return idx


class RemoteParameterSpace(ParameterSpace):
    def __init__(self, space_url, cred):
        self.space = RemoteParameterSpaceConnector(space_url, credential=cred)
        self.name = "remote_{}".format(self.space.get_space_id())
        self.params_dim = None
        param_vectors = self.space.get_param_vectors('all')
        super(RemoteParameterSpace, self).__init__(param_vectors)

    def get_name(self):
        return self.name

    def get_params_dim(self):
        if self.params_dim == None:
            param_vectors = self.get_param_vectors('candidates')
            self.params_dim = param_vectors.shape[1]
        return self.params_dim
    
    def get_hp_config(self):
        return self.space.hp_config

    def get_param_vectors(self, type_or_index, use_interim=False):
        return np.asarray(self.space.get_param_vectors(type_or_index, use_interim))

    def get_hpv_dict(self, index):
            return self.space.get_hpv_dict(index)

    def get_hp_vectors(self):
            return self.space.get_hp_vectors()


    # For history
    def get_candidates(self, use_interim=True):
        self.candidates = np.asarray(self.space.get_candidates(use_interim))
        return self.candidates

    def get_completions(self, use_interim=True):
        self.completions = np.asarray(self.space.get_completions(use_interim))
        return self.completions

    def update_error(self, sample_index, test_error, num_epochs=None):
        return self.space.update_error(sample_index, test_error, num_epochs)

    def get_errors(self, type_or_id):
        self.observed_errors, self.search_order = self.space.get_error(type_or_id)
        return np.asarray(self.observed_errors)

    def expand(self, hpv):
        return self.space.expand(hpv)
        