import os
import numpy as np
import pandas as pd
import random
import copy
from itertools import combinations_with_replacement as cwr

from ws.shared.logger import *

from hpo.utils.sobol_lib import i4_sobol_generate
from hpo.utils.converter import *

class GridGenerator(object):
    def __init__(self, config, num_samples, seed):
        self.seed = seed
        self.config = config
        self.params = config.get_param_list()
        self.num_dim = len(self.params)
        self.num_samples = num_samples

    def validate(self, candidate):
        # candidate is dict type
        cand = {}
        try:
            # Type forcing
            for k in candidate:
                if not k in self.config.get_param_list():
                    raise ValueError("{} is not in {}".format(k, self.params))
                v = candidate[k]
                t = eval(self.config.get_type(k))
                v = t(v)
                # Value check
                r_k = self.config.get_range(k)
                vt = self.config.get_value_type(k)
                if vt == 'categorical' or vt == 'preordered':
                    if not v in r_k:
                        raise ValueError("{} is not in {}".format(v, r_k)) 
                else:
                    if v < r_k[0] or v > r_k[-1]:
                        raise ValueError("{} is not in {}".format(v, r_k))
                cand[k] = v 

            return cand            
        
        except Exception as ex:
            raise ValueError("Invalid candidate:{}".format(ex))

    def get_schemata(self):
        '''return empty schemata'''
        return np.array(np.zeros((self.num_samples, self.num_dim)))

    def generate(self):
        ''' returns M * N normalized vectors '''
        raise NotImplementedError("This method should return given samples # * parameters # dimensional normalized array.")


class SobolSequenceGenerator(GridGenerator):
    def __init__(self, config, num_samples, seed):
        
        super(SobolSequenceGenerator, self).__init__(config, num_samples, seed)

    def generate(self):
        sobol_grid = np.transpose(i4_sobol_generate(self.num_dim, self.num_samples, self.seed)) 
        return sobol_grid
        

class UniformRandomGenerator(GridGenerator):
    def __init__(self, config, num_samples, seed):        
        super(UniformRandomGenerator, self).__init__(config, num_samples, seed)

    def generate(self):
        np.random.seed(self.seed)
        random_grid = np.transpose(np.random.rand(self.num_dim, self.num_samples))
        
        return random_grid
        

class LatinHypercubeGenerator(GridGenerator):
    def __init__(self, config, num_samples, seed):
        
        super(LatinHypercubeGenerator, self).__init__(config, num_samples, seed)

    def generate(self):
        from pyDOE import lhs
        random.seed(self.seed)
        
        hypercube_grid = np.array(lhs(self.num_dim, samples=self.num_samples))        
        return hypercube_grid


class EvolutionaryGenerator(GridGenerator):
    def __init__(self, config, num_samples, current_best, best_candidate, seed, m_ratio=.1):
        # current_best is {"hpv":[], "schema": []}
        self.converter = RepresentationConverter(config)
        self.male = current_best['hpv']
        self.m_schema = [ int(f) for f in current_best['schema'] ] # type forcing
        self.female = self.converter.to_vector(best_candidate, False)
        debug("Incumbent genotype: {}, phenotype: {}".format(self.male, self.m_schema))
        debug("Candidate genotype: {}".format(self.female))
        self.mutation_ratio = m_ratio
        self.schemata = []
        super(EvolutionaryGenerator, self).__init__(config, num_samples, seed)

    def generate(self):
        random.seed(self.seed)
        evol_grid = []
        self.schemata = []
        offsprings = self.cross_over(self.num_samples)
        mutated = self.mutate(offsprings, self.mutation_ratio)
        for m in mutated:
            # XXX:normalized vector will be used
            g = self.converter.to_norm_vector(m['hpv'], one_hot=False)
            evol_grid.append(g)
            self.schemata.append(m['schema'])
        return np.array(evol_grid)
    
    def get_schemata(self):
        return np.array(self.schemata)

    def get_random_mask(self):
        schemeta = []
        masks = list(cwr([0, 1], self.num_dim))
        masks = masks[1:-1] # remove all zero or all one mask
        s = np.random.randint(len(masks))
        schema = list(masks[s])
        np.random.shuffle(schema)

        return schema

    def cross_over(self, num_child):
        # populate offsprings from parents
        offsprings = [] 
        for c in range(num_child):
            m = self.get_random_mask()# randomized schema            
            o_schema = np.bitwise_xor(self.m_schema, m) # make offspring's schema
            o_hpv = [] # child hyperparam vector
            for i in range(len(o_schema)):
                bit = o_schema[i]
                if bit == 0: # inherit from male
                    o_hpv.append(self.male[i])
                elif bit == 1: # inherit from female
                    o_hpv.append(self.female[i])
                else:
                    raise ValueError("Invalid child schema: {}".format(o_schema))
            # validate new parameter
            hpv_dict = self.converter.to_typed_dict(o_hpv)
            self.validate(hpv_dict)
            
            child = {"hpv": o_hpv, "schema": o_schema }
            #debug("Child: {}".format(child))
            offsprings.append(child)
        return offsprings # contains {"hpv": [], "schema": []}

    def mutate(self, candidates, threshold):
        ''' returns [{"hpv": [], "schema": []}, ] '''
        
        mutated = []
        # candidates consist of {"hpv": [], "schema": []}
        for cand in candidates:
            if np.random.rand() <= threshold:
                n_i = random.randint(0, self.num_dim - 1) # choose param index
                # mutate this candidate
                hpv_dict = self.converter.to_typed_dict(cand['hpv'])
                lsg = LocalSearchGenerator(self.config, 1, hpv_dict, self.seed)
                hp_dict = lsg.perturb(n_i) # return dict type
                r_schema = cand['schema']
                # XOR operation in n_i
                if r_schema[n_i] == 1:
                    r_schema[n_i] = 0
                elif r_schema[n_i] == 0:
                    r_schema[n_i] = 1
                else:
                    raise ValueError("Invalid schema: {}".format(r_schema))
                m_cand = { "hpv": self.converter.to_vector(hp_dict, False), # XXX: hpv is normalized
                           "schema": r_schema } 
                mutated.append(m_cand)
            else:
                mutated.append(cand)
        return mutated


class LocalSearchGenerator(GridGenerator):
    def __init__(self, config, num_samples, best_candidate, seed, sd=0.2):        
               
        super(LocalSearchGenerator, self).__init__(config, num_samples, seed)
        self.candidate = self.validate(best_candidate) # XXX:validate requires __init__ first
        self.sd = sd
        self.converter = RepresentationConverter(config)
 
    def generate(self):
        np.random.seed(self.seed)         
        nc_list = []
        
        conv = RepresentationConverter(self.config)
        try:
            for i in range(self.num_samples):            
                n_i = random.randint(0, self.num_dim - 1) # choose param index
                nc = self.perturb(n_i)                            
                nc2 = self.converter.to_vector(nc)  
                nc_list.append(nc2)
        except Exception as ex:
            warn("Local search sampling failed:{}".format(ex))
        finally:
            return np.array(nc_list)

    def perturb(self, i):
        ''' returns perturbed dictionary ''' 
        ovt = OneHotVectorTransformer(self.config)
        hp_name = self.params[i] # hyperparameter name

        vt = self.config.get_value_type(hp_name)
        t = self.config.get_type(hp_name)
        r = self.config.get_range(hp_name)

        p_val = self.candidate[hp_name] # the value of choosen param
        np_val = None

        n_val = ovt.encode(vt, t, r, p_val)
        if vt == 'categorical': 
            # choose any others                
            ot_opts = np.delete(r, n_val.index(1.0), 0)
            np_val = np.random.choice(ot_opts)
        else:
            while True: # force to one exchange neighbourhood
                r_val = np.random.normal(n_val, self.sd) # random draw from normal
                if r_val < 0.:
                    r_val = 0.0
                elif r_val > 1.:
                    r_val = 1.0                
                un_val = ovt.decode(vt, t, r, r_val)
                # Value check                        
                if vt == 'categorical' or vt == 'preordered':
                    if not un_val in r:
                        warn("{} is not in {}".format(un_val, r))
                        continue 
                else:
                    if un_val < r[0] or un_val > r[-1]:
                        warn("{} is not in {}".format(un_val, r))
                        continue

                if p_val != un_val: # check parameter changed
                    np_val = un_val
                    break

        nc = copy.copy(self.candidate)
        nc[hp_name] = np_val
        return nc


