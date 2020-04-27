import os
import time
import json
import copy
import sys
import random

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 

import ws.shared.lookup as lookup
from hpo.utils.hpv_gen import HyperparameterVectorGenerator
from hpo.search_space import *


def connect_remote_space(space_url, cred):
    try:
        debug("Connecting remote space: {}".format(space_url))
        return RemoteParameterSpace(space_url, cred)
    except Exception as ex:
        warn("Fail to get remote samples: {}".format(ex))
        return None  

def create_space_from_table(surrogate_name, grid_order=None):
    
    l = lookup.load(surrogate_name, grid_order=grid_order)
    s = SurrogatesSpace(l)
    debug("Surrogate model created: {}".format(surrogate_name))
    return s


def create_surrogate_space(hp_cfg_dict, space_setting={}):

    if not 'num_samples' in space_setting:
        space_setting['num_samples'] = 20000

    if not 'sample_method' in space_setting:
        space_setting['sample_method'] = 'Sobol'

    if not 'seed' in space_setting:
        space_setting['seed']  = 1 # basic seed number
    else:
        if type(space_setting["seed"]) is int:
            grid_seed = space_setting["seed"]
        elif space_setting["seed"] == 'random':
            max_bound = 100000 # XXX: tentatively designed bound
            space_setting["seed"] = random.randint(1, max_bound)

    if not 'prior_history' in space_setting:
        space_setting['prior_history'] = None

    if 'dataset' in hp_cfg_dict and 'model' in hp_cfg_dict:
        prefix = "{}-{}".format(hp_cfg_dict['dataset'], hp_cfg_dict['model'])
    else:
        prefix = "{}-{}".format(space_setting['sample_method'], space_setting['seed'])
    name = "{}-{}".format(prefix, time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))

    with_default = False
    if 'starts_from_default' in space_setting:
        with_default = space_setting['starts_from_default']
    hvg = HyperparameterVectorGenerator(hp_cfg_dict, space_setting, with_default)
    hvg.generate()
    s = HyperParameterSpace(name, hp_cfg_dict, hvg.get_hp_vectors(),
                           space_setting=space_setting)
    debug("Search space created: {}".format(name))
    return s


def append_samples(space, num_samples):

    space.space_setting['num_samples'] = num_samples
    # Randomizes Sobol sequences  
    if space.space_setting['sample_method'] == 'Sobol':
        space.space_setting['seed'] += space.get_size() 
    
    hvg = HyperparameterVectorGenerator(space.get_hp_config(), space.space_setting)
    hvg.generate()    
    hpvs = hvg.get_hp_vectors()
    space.expand(hpvs)


def intensify_samples(space, num_samples, best_candidate, num_gen):
 
    try:
        space.space_setting['num_samples'] = num_samples
        space.space_setting['sample_method'] = 'local'
        space.space_setting['best_candidate'] = best_candidate # XXX:should be normalized value
        space.space_setting['generation'] = num_gen 

        hvg = HyperparameterVectorGenerator(space.get_hp_config(), space.space_setting)
        hvg.generate()    
        hpvs = hvg.get_hp_vectors()
        schemata = hvg.get_schemata()
        gen_counts = hvg.get_generations()
        if len(hpvs) > 0:
            space.expand(hpvs, schemata, gen_counts)
    except Exception as ex:
        warn("Exception raised on intensifying samples: {}".format(ex))


def evolve_samples(space, num_samples, current_best, best_candidate, mutation_ratio=.1):
    try:
        space.space_setting['num_samples'] = num_samples
        space.space_setting['sample_method'] = 'genetic'
        space.space_setting['current_best'] = current_best
        space.space_setting['best_candidate'] = best_candidate # XXX:should be normalized value
        space.space_setting['mutation_ratio'] = mutation_ratio

        hvg = HyperparameterVectorGenerator(space.get_hp_config(), space.space_setting)
        hvg.generate()    
        hpvs = hvg.get_hp_vectors()
        schemata = hvg.get_schemata()
        gen_counts = hvg.get_generations()
        if len(hpvs) > 0:
            space.expand(hpvs, schemata, gen_counts)
        else:
            error("Evolution failed - #s: {}, #c: {}, #b: {}".format(num_samples, current_best, best_candidate))
    except Exception as ex:
        warn("Exception raised on evolving samples: {}".format(ex))     
def remove_samples(space, method, estimates):
    if method == 'all_candidates':
        cands = space.get_candidates()
        space.remove(cands)
        return
    
    if estimates == None or not 'candidates' in estimates or not 'acq_funcs' in estimates:
        warn("Samples can not be removed without estimated values")
        return
    
    if not '[' in method or not ']' in method:
        warn("Invalid method format: {}".format(method))
        return
    else:
        o_i = method.find('[')
        e_i = method.find(']')

        try:
            method_type = method[:o_i]
            number = int(method[o_i+1:e_i])
        
        except Exception as ex:
            warn("Invalid method name: {}".format(method))
            return

        try:
            start_t = time.time()
            cands = np.array(estimates['candidates']) # has index
            est_values = np.array(estimates['acq_funcs']) # estimated performance by acquistion function
            if len(cands) < number:
                raise ValueError("Invalid method - removing {} samples can not performed over {} candidates".format(number, len(cands)))

            # supported method type: worst, except_top 
            if method_type == 'worst':
                # find worst {number} items to delete
                worst_k = est_values.argsort()[:number][::1]

                space.remove(cands[worst_k])
            elif method_type == 'except_top':
                # find top {number} items to be saved
                top_k = est_values.argsort()[-1 * number:][::-1]
                remains = np.setdiff1d(cands, cands[top_k]) # remained items
                space.remove(remains)
            else:
                raise ValueError("Invalid method type: {}".format(method))

        except Exception as ex:
            warn("Removing sample failed: {}".format(ex))



class SearchSpaceManager(ManagerPrototype):

    def __init__(self, *args, **kwargs):
        super(SearchSpaceManager, self).__init__(type(self).__name__)
        self.spaces = {} 

    def create(self, space_spec):
        if "surrogate" in space_spec:
            surrogate = space_spec["surrogate"]
            grid_order = None

            if "grid_order" in space_spec:
                grid_order = space_spec["grid_order"]
            s = create_space_from_table(surrogate, grid_order)
            cfg = surrogate
        else:
            if not "hp_config" in space_spec:
                raise ValueError("No hp_config in parameter space spec: {}".format(space_spec))
            
            hp_cfg = space_spec['hp_config']
    
            if not "num_samples" in space_spec:
                space_spec["num_samples"] = 20000
            
            if "grid_seed" in space_spec:
                grid_seed = space_spec["grid_seed"]
            else:
                space_spec["grid_seed"] = 1

            s = create_surrogate_space(hp_cfg, space_spec)
            cfg = hp_cfg
        
        space_id = s.name
        space_obj = {"id" : space_id, "config": cfg, "samples": s }
        space_obj["created"] = time.strftime('%Y%m%dT%H:%M:%SZ',time.gmtime())
        space_obj["status"] = "created"    
        
        self.spaces[space_id] = space_obj

        return space_id

    def get_available_spaces(self):
        return list(self.spaces.keys())

    def get_active_space_id(self):
        for s in self.spaces:
            if self.spaces[s]['status'] == "active":
                return s
        debug("No space is active now")
        return None                  

    def set_space_status(self, space_id, status):
        if space_id == "active":
            space_id = get_active_space_id()
                                
        elif space_id in self.spaces:
            self.spaces[space_id]['status'] = status
            self.spaces[space_id]['updated'] = time.strftime('%Y%m%dT%H:%M:%SZ',time.gmtime())
            return True
        else:
            debug("No such space {} existed".format(space_id))
            return False

    def get_samples(self, space_id):
        if space_id == "active":
            for s in self.spaces:
                if self.spaces[s]['status'] == "active":
                    return self.spaces[s]['samples']
            return None
        elif space_id in self.spaces:
            return self.spaces[space_id]['samples']
        else:
            debug("No such {} space existed".format(space_id))
            return None

    def get_space_config(self, space_id):
        if space_id == "active":
            space_id = get_active_space_id()
                    
        if space_id in self.spaces:
            return self.spaces[space_id]['config']
        else:
            debug("No such space space {} existed".format(space_id))
            return None        


