import os
import numpy as np
import pandas as pd
import random
import copy
import time

from ws.shared.logger import *
from ws.shared.hp_cfg import HyperparameterConfiguration
from hpo.utils.sample_gen import *


class HyperparameterVectorGenerator(object):
    def __init__(self, config, spec):
        if type(config) == dict:
            self.config = HyperparameterConfiguration(config)
        else:
            self.config = config
        self.params = self.config.get_param_names()
        self.spec = spec
        self.grid = np.array([])
        self.hpvs = np.array([])
        self.schemata = np.array([])
        self.generations = np.array([])
        
    def get_param_vectors(self):
        return self.grid

    def get_hp_vectors(self):        
        return self.hpvs
    def get_schemata(self):        
        return self.schemata
    def get_generations(self):
        return self.generations

    def generate(self):
        #debug("Sampling hyperparameter configurations...")
        s_t = time.time()
        try:
            spec = self.spec
            seed = spec['seed']
            n_s = spec['num_samples']
            if 'num_skips' in spec:
                seed += spec['num_skips']

            if not 'mutation_ratio' in spec:
                spec['mutation_ratio'] = .1 # default mutation ratio
            g = SobolSequenceGenerator(self.config, spec['num_samples'], seed)
            
            if 'sample_method' in spec:
                if spec['sample_method'] == 'uniform':
                    g = UniformRandomGenerator(self.config, n_s, seed)
                elif spec['sample_method'] == 'latin':
                    g = LatinHypercubeGenerator(self.config, n_s, seed)
                elif spec['sample_method'] == 'local':
                    g = LocalSearchGenerator(self.config, n_s, 
                                             spec['best_candidate'], spec['generation'], seed)
                elif spec['sample_method'] == 'genetic':
                    g = EvolutionaryGenerator(self.config, n_s, spec['current_best'],
                                             spec['best_candidate'], seed, spec['mutation_ratio'])                                             
                elif spec['sample_method'] != 'Sobol':
                    warn("Not supported sampling method: {}. We utilize Sobol sequences as default.".format(spec['sample_method']))

            self.grid = np.asarray(g.generate())
            self.schemata = g.get_schemata()
            self.generations = g.get_generations()
            # TODO:speeding up required
            self.hpvs = []
            for i in range(len(self.grid)):
                vec = self.grid[i]
                hpv = []
                for j in range(len(vec)):
                    param_name = self.params[j]
                    value = vec[j]
                    hp_cfg = getattr(self.config.hyperparams, param_name)
                    arg = self.to_param_value(hp_cfg, value)
                    hpv.append(arg)
                self.hpvs.append(hpv)

        except Exception as ex:
            warn("Failed to generate space: {}".format(ex))
        finally:
            if len(self.grid) > 1:
                debug("{} samples have been populated. ({:.1f} sec)".format(len(self.grid), time.time() - s_t))


    def to_param_value(self, hp_cfg, value):
        result = None
        range_list = hp_cfg.range
        range_list.sort()

        if hp_cfg.value_type == "categorical" or hp_cfg.value_type == 'preordered':
            size = len(range_list)
            index = int(value * size)
            if index == size: # handle terminal condition
                index = size - 1 
            result = range_list[index]
        else:
            max_value = max(range_list)
            min_value = min(range_list)

            if hp_cfg.type == 'int':
                result = min_value + int(value * (max_value - min_value)) 
                #XXX:to include max value
                if value == 1.0:
                    result = max_value
                if hasattr(hp_cfg, 'power_of'):
                    result = int(np.power(hp_cfg.power_of, result))

            elif hp_cfg.type == 'float':
                result = min_value + (value * (max_value - min_value)) 
                #XXX:to include max value
                if value == 1.0:
                    result = max_value

                if hasattr(hp_cfg, 'power_of'):
                    result = np.power(hp_cfg.power_of, result)
        
        if hp_cfg.type == 'int':
            result = int(result)
        elif hp_cfg.type == 'bool':
            result = bool(result)
        elif hp_cfg.type == 'str':
            result = str(result)
        return result

