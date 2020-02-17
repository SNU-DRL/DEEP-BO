import os
import numpy as np
import pandas as pd
import random
import copy

from ws.shared.logger import *

from hpo.utils.sobol_lib import i4_sobol_generate
from hpo.utils.converter import *

class GridGenerator(object):
    def __init__(self, num_dim, num_samples, seed):
        self.seed = seed
        self.num_dim = num_dim
        self.num_samples = num_samples

    def generate(self):
        raise NotImplementedError("This method should return given samples # * parameters # dimensional array.")


class SobolSequenceGenerator(GridGenerator):
    def __init__(self, params, num_samples, seed):
        self.params = params
        num_dim = len(self.params)
        super(SobolSequenceGenerator, self).__init__(num_dim, num_samples, seed)

    def generate(self):
        sobol_grid = np.transpose(i4_sobol_generate(self.num_dim, self.num_samples, self.seed)) 
        return sobol_grid
        

class UniformRandomGenerator(GridGenerator):
    def __init__(self, params, num_samples, seed):
        self.params = params
        np.random.seed(seed)
        num_dim = len(self.params)
        super(UniformRandomGenerator, self).__init__(num_dim, num_samples, seed)

    def generate(self):
        random_grid = np.transpose(np.random.rand(self.num_dim, self.num_samples))
        
        return random_grid
        

class LatinHypercubeGenerator(GridGenerator):
    def __init__(self, params, num_samples, seed):
        self.params = params
        random.seed(seed)

        num_dim = len(self.params)
        super(LatinHypercubeGenerator, self).__init__(num_dim, num_samples, seed)

    def generate(self):
        from pyDOE import lhs
        
        hypercube_grid = np.array(lhs(self.num_dim, samples=self.num_samples))        
        return hypercube_grid


class LocalSearchGenerator(GridGenerator):
    def __init__(self, params, config, num_samples, best_candidate, seed, sd=0.2):
        self.params = params
        self.config = config
        self.candidate = self.validate(best_candidate)
        self.sd = sd
        np.random.seed(seed)

        num_dim = len(self.candidate)
        super(LocalSearchGenerator, self).__init__(num_dim, num_samples, seed)

    def validate(self, candidate):
        cand = {}
        try:
            # Type forcing
            for k in candidate:
                if not k in self.params:
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

    def generate(self):        
        nc_list = []
        ovt = OneHotVectorTransformer(self.config)
        vgc = VectorGridConverter(self.config)
        try:
            for i in range(self.num_samples):            
                n_i = random.randint(0, self.num_dim-1) # choose param index
                hp_name = self.params[n_i] # hyperparameter name

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
                            
                nc2 = vgc.to_grid_array(nc)  
                nc_list.append(nc2)
        except Exception as ex:
            warn("Local search sampling failed:{}".format(ex))
        finally:
            return np.array(nc_list)


