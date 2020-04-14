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
    def __init__(self, config, spec, use_default=False):
        if type(config) == dict:
            self.config = HyperparameterConfiguration(config)
        else:
            self.config = config
        self.params = self.config.get_param_names()
        self.spec = spec
        self.use_default = use_default
        if use_default:
            default = self.config.get_default_vector()
            debug("Default value setting: {}".format(default))
            norm_vec = self.config.convert("arr", "norm_arr", default)
            self.grid = np.array([norm_vec])
            self.hpvs = np.array([default])
            self.schemata = np.zeros(self.hpvs.shape)
            self.generations = np.array([0,])
        else:
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

            grid = g.generate()
            schemata = g.get_schemata()
            gen = g.get_generations()
            # TODO:speeding up required
            if self.use_default:
                self.grid = np.concatenate((self.grid, grid))
                self.schemata = np.concatenate((self.schemata, schemata))
                self.generations = np.concatenate((self.generations, gen))
            else:
                self.grid = grid
                self.schemata = schemata
                self.generations = gen

            self.hpvs = self.config.convert('grid', 'hpv_list', self.grid)
            

        except Exception as ex:
            warn("Failed to generate space: {}".format(ex))
        finally:
            if len(self.grid) > 1:
                debug("{} samples have been populated. ({:.1f} sec)".format(len(self.grid), time.time() - s_t))

