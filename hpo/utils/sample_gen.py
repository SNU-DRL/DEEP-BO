import os
import numpy as np
import pandas as pd
import random
import copy
from itertools import combinations
from itertools import combinations_with_replacement as cwr

from ws.shared.logger import *

from hpo.utils.sobol_lib import i4_sobol_generate
from ws.shared.converter import OneHotVectorTransformer

class GridGenerator(object):
    def __init__(self, config, num_samples, seed):
        self.seed = seed
        self.config = config
        self.params = config.get_param_names()
        self.num_dim = len(self.params)
        self.num_samples = num_samples

    def get_name(self):
        if self.name:
            return self.name
        else:
            return "Undefined"
    def validate(self, candidate):
        # candidate is dict type
        cand = {}
        try:
            # Type forcing
            for k in candidate:
                if not k in self.config.get_param_names():
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
    def get_generations(self):
        '''return all zeros'''
        return np.array(np.zeros(self.num_samples))

    def generate(self):
        ''' returns M * N normalized vectors '''
        raise NotImplementedError("This method should return given samples # * parameters # dimensional normalized array.")


class SobolSequenceGenerator(GridGenerator):
    def __init__(self, config, num_samples, seed):
        self.name = 'Sobol sequences'
        super(SobolSequenceGenerator, self).__init__(config, num_samples, seed)

    def generate(self):
        sobol_grid = np.transpose(i4_sobol_generate(self.num_dim, self.num_samples, self.seed)) 
        return sobol_grid
        

class UniformRandomGenerator(GridGenerator):
    def __init__(self, config, num_samples, seed):        
        self.name = 'uniform random sampling'        
        super(UniformRandomGenerator, self).__init__(config, num_samples, seed)

    def generate(self):
        np.random.seed(self.seed)
        random_grid = np.transpose(np.random.rand(self.num_dim, self.num_samples))
        
        return random_grid
        

class LatinHypercubeGenerator(GridGenerator):
    def __init__(self, config, num_samples, seed):
        self.name = 'Latin Hypercube sampling' 
        super(LatinHypercubeGenerator, self).__init__(config, num_samples, seed)

    def generate(self):
        from pyDOE import lhs
        random.seed(self.seed)
        
        hypercube_grid = np.array(lhs(self.num_dim, samples=self.num_samples))        
        return hypercube_grid


class EvolutionaryGenerator(GridGenerator):
    def __init__(self, config, num_samples, current_best, best_candidate, seed, m_ratio=.1):
        # current_best is {"hpv":[], "schema": []}
        self.male = current_best['hpv']
        self.generation = current_best['gen'] + 1 # set offsprings' generation
        self.m_schema = [ int(f) for f in current_best['schema'] ] # type forcing
        self.female = config.convert("dict", "arr", best_candidate)
        debug("Incumbent genotype: {}, phenotype: {}".format(self.male, self.m_schema))
        debug("Candidate genotype: {}".format(self.female))
        self.mutation_ratio = m_ratio
        self.schemata = []
        self.generations = []
        self.name = 'evolutionary sampling'
        super(EvolutionaryGenerator, self).__init__(config, num_samples, seed)

    def generate(self):
        random.seed(self.seed)
        evol_grid = []
        self.schemata = []
        candidates = []
        n_dim = len(self.male)
        candidates = self.cross_over_full(self.num_samples)
        n_remains = self.num_samples - len(candidates)
        offsprings = []
        if n_remains <= 0:
            for cand in candidates:
                if np.random.rand() <= self.mutation_ratio:
                    offsprings.append(self.mutate(cand))
                else:
                    offsprings.append(cand)
        else:
            # XXX:normalized vector will be used
            offsprings = candidates
            for n in range(n_remains):
                m = random.sample(candidates, 1)[0]
                offsprings.append(self.mutate(m))
        for o in offsprings:            
            g = self.config.convert('arr', 'norm_arr', o['hpv'])
            evol_grid.append(g)
            self.schemata.append(o['schema'])
            self.generations.append(o['gen'])
        return np.array(evol_grid)
    
    def get_schemata(self):
        return np.array(self.schemata)
    def get_generations(self):
        return np.array(self.generations)

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
            hpv_dict = self.config.convert("arr", "dict", o_hpv)
            self.validate(hpv_dict)
            
            child = {"hpv": o_hpv, "schema": o_schema,"gen": self.generation }
            offsprings.append(child)
        return offsprings # contains {"hpv": [], "schema": []}
    def cross_over_full(self, num_child):
        offsprings = []
        o_schemata = self.create_schemata(num_child) 
        n_offsprings = len(o_schemata)
        if n_offsprings < num_child:
            debug("The # of possible offsprings is less then {}: {}".format(num_child, n_offsprings))
        else:
            o_schemata = random.sample(o_schemata, num_child)       
        for o_schema in o_schemata:
            o_hpv = [] # child hyperparam vector
            for i in range(len(o_schema)):
                bit = o_schema[i]
                if bit == 0: # inherit from male
                    o_hpv.append(self.male[i])
                elif bit == 1: # inherit from female
                    o_hpv.append(self.female[i])
                else:
                    raise ValueError("Invalid child schema: {}".format(o_schema))
            hpv_dict = self.config.convert("arr", "dict", o_hpv)
            self.validate(hpv_dict)
            child = {"hpv": o_hpv, "schema": o_schema,"gen": self.generation }
            offsprings.append(child)
        return offsprings # contains {"hpv": [], "schema": []}

    def create_schemata(self, num_child):
        o_schemata = [] 
        n_params = len(self.m_schema)
        
        for i in range(1, n_params):
            for c in self.create_schema_list(n_params, i):
                o_schemata.append(c)
                if len(o_schemata) >= num_child:
                    return o_schemata
        return o_schemata
        # candidates consist of {"hpv": [], "schema": []}
    def create_schema_list(self, n_p, n_on):
        arr = []
        combi = combinations([ i for i in range(n_p)], n_on)
        for c in combi:
            a = [0 for i in range(n_p)]
            for i in c:
                a[i] = 1
            arr.append(a)
        return arr
    def mutate(self, cand):
                # mutate this candidate
        hpv_dict = self.config.convert("arr", "dict", cand['hpv'])
        lsg = LocalSearchGenerator(self.config, 1, hpv_dict, self.generation, self.seed)
        hp_dict, n_i = lsg.perturb(self.num_dim) # return dict type
        r_schema = cand['schema']
                # XOR operation in n_i
        if r_schema[n_i] == 1:
            r_schema[n_i] = 0
        elif r_schema[n_i] == 0:
            r_schema[n_i] = 1
        else:
            raise ValueError("Invalid schema: {}".format(r_schema))
        m_cand = { "hpv": self.config.convert("dict", "arr", hp_dict), 
                "schema": r_schema, "gen": self.generation } 
        return m_cand
        


class LocalSearchGenerator(GridGenerator):
    def __init__(self, config, num_samples, best_candidate, best_gen, seed, sd=0.2):        
        self.name = 'Local sampling'       
        super(LocalSearchGenerator, self).__init__(config, num_samples, seed)
        self.candidate = self.validate(best_candidate) # XXX:validate requires __init__ first
        self.sd = sd
        self.generation = best_gen + 1 # inherits from best_candidate
        self.schemata = []
        self.generations = []
 
    def generate(self):
        np.random.seed(self.seed)         
        nc_list = []
        
        try:
            for i in range(self.num_samples):            
                schema = np.zeros(self.num_dim)            
                nc, n_i = self.perturb(self.num_dim)
                schema[n_i] = 1
                nc2 = self.config.convert("dict", "norm_arr", nc)  
                nc_list.append(nc2)
                self.schemata.append(schema)
                self.generations.append(self.generation)
        except Exception as ex:
            warn("Local search sampling failed:{}".format(ex))
        finally:
            return np.array(nc_list)
    def get_schemata(self):
        return self.schemata
    def get_generations(self):
        return self.generations

    def perturb(self, num_dim, excluded_index=None):

        i = random.randint(0, num_dim - 1) # choose param index
        if excluded_index != None:
            while i == excluded_index:
                i = random.randint(0, num_dim - 1) 
        ''' returns perturbed value as dictionary type ''' 
        ovt = OneHotVectorTransformer(self.config)
        hp_name = self.params[i] # hyperparameter name

        vt = self.config.get_value_type(hp_name)
        t = self.config.get_type(hp_name)
        r = self.config.get_range(hp_name)

        p_val = self.candidate[hp_name] # the value of choosen param
        np_val = None

        n_val = ovt.encode(vt, t, r, p_val)
        if vt == 'categorical': 
            try:
                # choose any others                
                ot_opts = np.delete(r, n_val.index(1.0), 0)
                np_val = np.random.choice(ot_opts)
            except Exception as ex:
                return self.perturb(num_dim, excluded_index=i)
        elif vt == 'preordered':
            try:
                ot_opts = np.delete(r, r.index(p_val), 0)
                np_val = np.random.choice(ot_opts)
            except Exception as ex:
                return self.perturb(num_dim, excluded_index=i)            
        else:
            while True: # force to one exchange neighbourhood
                r_val = np.random.normal(n_val, self.sd) # random draw from normal
                if r_val < 0.:
                    r_val = 0.0
                elif r_val > 1.:
                    r_val = 1.0                
                un_val = ovt.decode(vt, t, r, r_val)
                # Value check                        
                if un_val < r[0] or un_val > r[-1]:
                    warn("{} is not in {}".format(un_val, r))
                    continue

                if p_val != un_val: # check parameter changed
                    np_val = un_val
                    break

        nc = copy.copy(self.candidate)
        nc[hp_name] = np_val
        return nc, i


