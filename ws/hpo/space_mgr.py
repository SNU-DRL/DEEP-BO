import os
import time
import json
import copy

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 

import ws.shared.lookup as lookup

from ws.hpo.sample_space import *


def connect_remote_space(space_url, cred):
    try:
        debug("Connecting remote space: {}".format(space_url))
        return RemoteParameterSpace(space_url, cred)
    except Exception as ex:
        warn("Fail to get remote samples: {}".format(ex))
        return None  


def create_surrogate_space(surrogate, grid_order=None, one_hot=False):
    l = lookup.load(surrogate, grid_order=grid_order)
    s = SurrogatesSpace(l, one_hot=one_hot)
    debug("Surrogate model created: {}".format(surrogate))
    return s


def create_grid_space(hp_cfg_dict, num_samples=20000, grid_seed=1, one_hot=False):
    if 'config' in hp_cfg_dict:
        if 'num_samples' in hp_cfg_dict['config']:
            num_samples = hp_cfg_dict['config']['num_samples']

        if 'grid_seed' in hp_cfg_dict['config']:
            grid_seed = hp_cfg_dict['config']['grid_seed']
    prefix = 'Sobol'
    if 'dataset' in hp_cfg_dict and 'model' in hp_cfg_dict:
        prefix = "{}-{}".format(hp_cfg_dict['dataset'], hp_cfg_dict['model'])
    name = "{}-{}".format(prefix, time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    
    hvg = HyperparameterVectorGenerator(hp_cfg_dict, num_samples, grid_seed)
    s = GridParameterSpace(name, 
                        hvg.get_grid(), hvg.get_hpv(), 
                        hp_cfg_dict, 
                        one_hot=one_hot)
    debug("Sampling space created: {}".format(name))
    return s


class SamplingSpaceManager(ManagerPrototype):

    def __init__(self, *args, **kwargs):
        super(SamplingSpaceManager, self).__init__(type(self).__name__)
        self.spaces = {} 

    def create(self, space_spec):
        if "surrogate" in space_spec:
            surrogate = space_spec["surrogate"]
            grid_order = None

            if "grid_order" in space_spec:
                grid_order = space_spec["grid_order"]
            s = create_surrogate_space(surrogate, grid_order)
            cfg = surrogate
        else:
            if not "hp_config" in space_spec:
                raise ValueError("No hp_config in parameter space spec: {}".format(space_spec))
            
            hp_cfg = space_spec['hp_config']
            num_samples = 20000
            grid_seed = 1               
            if "num_samples" in hp_cfg:
                num_samples = hp_cfg["num_samples"]
            if "grid_seed" in space_spec:
                grid_seed = space_spec["grid_seed"]

            s = create_grid_space(hp_cfg, num_samples, grid_seed)
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


