import os
import copy
import json
import time
import numpy as np
import pandas as pd

from ws.shared.logger import *
from ws.shared.hp_cfg import HyperparameterConfiguration

from hpo.connectors.remote_space import RemoteParameterSpaceConnector


class ParameterSpace(object):
    def __init__(self, parameters):
        self.param_vectors = parameters
        num_samples = len(parameters)
        self.completions = np.arange(0)
        self.candidates = np.setdiff1d(np.arange(num_samples), self.completions)
        self.observed_errors = np.ones(num_samples)
        self.loss_curves = {}
        self.train_epochs = np.zeros(num_samples)
        self.min_train_epoch = None
        
    def reset(self):
        num_samples = len(self.param_vectors)
        self.completions = np.arange(0)
        self.candidates = np.setdiff1d(np.arange(num_samples), self.completions)        
        self.observed_errors = np.ones(num_samples)
        self.train_epochs = np.zeros(num_samples)
        
        self.loss_curves = {}
        self.min_train_epoch = None

    def set_min_train_epoch(self, epoch): # XXX:For truncate warm up results
            # select candidates by elements which have 0 of terminal record
        self.min_train_epoch = epoch
    def get_candidates(self):
        return self.candidates

    def get_completions(self):
        if self.min_train_epoch == None: 
            return self.completions
        else:
           completions = np.where(self.train_epochs > self.min_train_epoch)[0]
           return completions

    def get_incumbent(self):

        min_i = np.argmin(self.get_errors("completions"))
        i = self.completions[min_i]
        return i

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
            if not sample_index in self.loss_curves:
                self.loss_curves[sample_index] = {}    
            self.loss_curves[sample_index][num_epochs] = test_error

    def get_errors(self, type_or_id="completions"):
        if type_or_id == "completions":
            c = self.get_completions()
            try:
                return self.observed_errors[c]
            except Exception as ex:
                debug("Exception on get errors: {}".format(ex))
                return []
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

    def remove(self, indices):
        # check index in candidates or completions
        diff = np.setdiff1d(indices, self.candidates)
        if len(diff) > 0:
            debug('{} points to be removed are not existed in {} candidates'.format(len(diff), len(self.candidates)))
        self.candidates = np.setdiff1d(self.candidates, indices)


class HyperParameterSpace(ParameterSpace):

    def __init__(self, name, hp_config_dict, hpv_list, 
                 space_setting=None):

        self.name = name
        self.hp_config = HyperparameterConfiguration(hp_config_dict)
        
        self.spec = space_setting
        self.prior_history = None
        if 'prior_history' in space_setting:
            self.prior_history = space_setting['prior_history']

        self.resampled = False
        if 'resample_steps' in space_setting:
            if space_setting['resample_steps'] > 0:
                self.resampled = True

        self.priors = []
        self.backups = {} # XXX: it may take large size memory

        self.hp_vectors = copy.copy(hpv_list)
        self.initial_hpv = hpv_list
        self.schemata = np.zeros(np.array(hpv_list).shape)
        self.gen_counts = np.zeros(len(hpv_list)) # generation counts for evolutionary sampling

        super(HyperParameterSpace, self).__init__(self.one_hot_encode(self.hp_vectors))

    def reset(self):
        self.hp_vectors = copy.copy(self.initial_hpv)
        self.param_vectors = self.one_hot_encode(self.hp_vectors)
        super(HyperParameterSpace, self).reset()

        if self.prior_history != None:
            try:
                if len(self.priors) == 0:
                    if self.prior_history.lower().endswith('.csv'):
                        self.priors = self.load_prior_from_table(self.prior_history)
                    else:
                        hp_vectors = self.load('spaces', self.prior_history)
                        if len(hp_vectors) == 0:
                            raise IOError("No space information retrieved: {}".format(self.prior_history))
                        self.priors = self.extract_prior(hp_vectors, 'results', self.prior_history)
                
                self.preset()
                debug("The # of prior observations: {}".format(len(self.completions)))

            except Exception as ex:
                warn("Use of prior history failed: {}".format(ex))
   
    def archive(self, run_index):
        
        if run_index == 0:
            k_hpv = "hpv"
            k_schemata = "schemata"
            k_gen_count = "gen_count"
        else:
            k_hpv = "hpv{}".format(run_index)
            k_schemata = "schemata{}".format(run_index)
            k_gen_count = "gen_count{}".format(run_index)
        
        if self.resampled == True:
            self.backups[k_hpv] = np.array(copy.copy(self.hp_vectors))
            self.backups[k_schemata] = np.array(copy.copy(self.schemata))
            self.backups[k_gen_count] = np.array(copy.copy(self.gen_counts))  

    def update_history(self, run_index, folder='temp/'):
        # FIXME: below works stupidly because it refreshs from scratch.
        # save current experiment to csv format
        hpv_dict_list = []
        for c in self.completions:
            h = self.get_hpv_dict(c)
            e = self.get_errors(c)
            t = self.get_train_epoch(c)
            h['_error_'] = e
            h['_epoch_'] = t
            hpv_dict_list.append(h)

        # create dictionary type results
        if len(hpv_dict_list) > 0:
            df = pd.DataFrame.from_dict(hpv_dict_list)
            path = "{}{}-{}.csv".format(folder, self.name, run_index)
            csv = df.to_csv(path, index=False)
            debug("Current progress updated at {}".format(path))

    def save(self):
        # save hyperparameter vectors & schemata when no backup available
        if not "hpv" in self.backups:
            self.backups["hpv"] = np.array(copy.copy(self.hp_vectors))

        if not "schemata" in self.backups:
            self.backups["schemata"] = np.array(copy.copy(self.schemata))        
        
        if not "gen_count" in self.backups:
            self.backups["gen_count"] = np.array(copy.copy(self.gen_counts))

        try:
            if not os.path.isdir("./spaces/"):
                os.mkdir("./spaces/")

            file_name = "spaces/{}.npz".format(self.name)
            np.savez_compressed(file_name, **self.backups)
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
                    hpv = self.hp_config.convert("arr", "list", c['hyperparams'])
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
    def load_prior_from_table(self, csv_file, csv_dir='temp/'):
        completions = {}
        csv_path = csv_dir + csv_file
        try:            
            hist = pd.read_csv(csv_path)
            hp_params = self.hp_config.get_param_names()
            errors = hist['_error_'].tolist()
            epochs = None
            if '_epoch_' in hist:
                epochs = hist['_epoch_'].tolist()
            for i in range(len(errors)):
                hp_vector = hist[hp_params].iloc[i].tolist()
                train_epoch = 0
                if epochs != None:
                    train_epoch = epochs[i]
                completions[i] = {
                    "hyperparams": hp_vector,
                    "observed_error": errors[i],
                    "train_epoch": train_epoch 
                } 
        except Exception as ex:
            warn("Exception on loading prior history from table: {}".format(ex))
            raise ValueError("Invalid prior table file: {}".format(csv_path))
        return completions

    def get_size(self):
        return len(self.hp_vectors)

    def get_name(self):
        return self.name

    def get_hp_config(self):
        return self.hp_config

    def get_schema(self, index):
        if len(self.schemata) <= index:
            raise ValueError("Invalid index: {}".format(index))

        return self.schemata[index]

    def get_generation(self, index):
        if len(self.gen_counts) < index:
            raise ValueError("Invalid index: {}".format(index))
        return self.gen_counts[index]

    def get_hpv_dim(self):
        return len(self.hp_vectors[0])

    def get_params_dim(self):
        return self.param_vectors.shape[1]

    def get_param_vectors(self, type_or_index='all'): 
        if type(type_or_index) == str: 
            if type_or_index == "completions":
                completions = self.get_completions() 
                #debug("index of completions: {}".format(completions))
                return self.param_vectors[completions, :]
            elif type_or_index == "candidates":
                candidates = self.get_candidates()
                #debug("index of candidates: {}".format(candidates))
                return self.param_vectors[candidates, :]        
            elif type_or_index == 'all':
                return self.param_vectors            
        else:
            return self.param_vectors[type_or_index]
 
    def get_hp_vectors(self):
        return self.hp_vectors # XXX:return array value

    def get_hpv(self, index):
        return self.hp_vectors[index]

    def get_hpv_dict(self, index, k=None):
        hpv_list = self.hp_vectors
        if k != None:
            if k == 0:
                if 'hpv' in self.backups: 
                    hpv_list = self.backups['hpv']
            else:
                key = 'hpv{}'.format(k)
                if key in self.backups:
                    hpv_list = self.backups[key]
                else:
                    error("No backup of hyperparamter vectors: {}".format(k))
        hp_arr = hpv_list[index]
        hpv = self.hp_config.convert('arr', 'dict', hp_arr)
        return hpv # XXX:return dictionary value

    def set_schema(self, index, schema):
        if len(schema) != self.get_hpv_dim():
            raise ValueError("Invalid schema dimension.")

        if len(self.schemata) <= index:
            raise ValueError("Invalid index: {}".format(index))
        # TODO:validate input
        self.schemata[index] = schema

    def expand(self, hpv, schemata=[], gen_counts=[]):
        # check dimensions
        if type(hpv) == dict:
            hpv = self.hp_config.convert('dict', 'arr', hpv)
        hpv_list = hpv
        dim = len(np.array(hpv).shape)
        if dim == 1:
            hpv_list = [ hpv ]
        elif dim != 2:
            raise TypeError("Invalid hyperparameter vector")

        # XXX:assumed that hpv_list are consisted with valid hyperparameter vectors.
        param_list = []
        vec_indices = []
        vec_index = len(self.param_vectors) # starts with the last model index
        for h in hpv_list:
            param_list.append(self.hp_config.convert('arr', 'one_hot', h))
            vec_indices.append(vec_index)
            vec_index += 1
        self.param_vectors = np.append(self.param_vectors, param_list, axis=0)
        self.hp_vectors = np.append(self.hp_vectors, hpv_list, axis=0)
        if len(gen_counts) == 0:
            gen_counts = np.zeros(len(hpv_list))
        if len(schemata) == 0:
            schemata = np.zeros(np.array(hpv_list).shape)
        
        if len(hpv_list) != len(schemata):
            raise ValueError("Invalid schemata: {}".format(schemta))
        elif len(schemata) != len(gen_counts):
            raise ValueError("Size mismatch: schema {} != generation {}".format(schemata.shape, gen_counts.shape))
        
        self.schemata = np.append(self.schemata, schemata, axis=0)
        self.gen_counts = np.append(self.gen_counts, gen_counts, axis=0)
        
        return super(HyperParameterSpace, self).expand(vec_indices)

    def one_hot_encode(self, hpv_list):
        # TODO:below loop can be faster using parallelization.
        encoded = []
        for hpv in hpv_list:
            e = self.hp_config.convert('arr', 'one_hot', hpv)
            encoded.append(np.asarray(e))
        return np.asarray(encoded) 


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
        idx, dist = self.hp_config.get_nearby_index(self.get_candidates(), 
                                                    self.hp_vectors, 
                                                    hpv)

        debug("Distance btw selection and surrogate: {}".format(dist))
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

    def get_param_vectors(self, type_or_index):
        return np.asarray(self.space.get_param_vectors(type_or_index))

    def get_hpv_dict(self, index):
            return self.space.get_hpv_dict(index)

    def get_hp_vectors(self):
            return self.space.get_hp_vectors()

    # For history
    def get_candidates(self):
        self.candidates = np.asarray(self.space.get_candidates())
        return self.candidates

    def get_completions(self):
        self.completions = np.asarray(self.space.get_completions())
        return self.completions

    def update_error(self, sample_index, test_error, num_epochs=None):
        return self.space.update_error(sample_index, test_error, num_epochs)

    def get_errors(self, type_or_id):
        self.observed_errors, self.search_order = self.space.get_error(type_or_id)
        return np.asarray(self.observed_errors)

    def expand(self, hpv):
        return self.space.expand(hpv)
        