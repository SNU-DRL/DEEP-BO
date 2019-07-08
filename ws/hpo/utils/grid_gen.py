import os
import numpy as np
import pandas as pd


from ws.shared.logger import *
from ws.shared.hp_cfg import HyperparameterConfiguration

from ws.hpo.utils.sobol_lib import i4_sobol_generate

class GridGenerator(object):
    def __init__(self, num_dim, num_samples):
        self.num_dim = num_dim
        self.num_samples = num_samples
        

    def generate(self):
        raise NotImplementedError("This method should return given N *M dimensional array.")


class SobolGridGenerator(GridGenerator):
    def __init__(self, params, num_samples, seed=1):
        self.seed = seed
        self.params = params
        num_dim = len(self.params)
        super(SobolGridGenerator, self).__init__(num_dim, num_samples)

    def generate(self, return_type='array'):
        sobol_grid = np.transpose(i4_sobol_generate(self.num_dim, self.num_samples, self.seed)) 
        if return_type == 'array':
            return sobol_grid
        elif return_type == 'table':
            table = pd.DataFrame(data=sobol_grid, columns=self.params)            
            return table 


class HyperparameterVectorGenerator(object):
    def __init__(self, config_dict, num_samples, seed=1, grid_type='Sobol'):
        self.config = HyperparameterConfiguration(config_dict)
        self.params = self.config.get_param_list()
                
        sobol = None
        if grid_type == 'Sobol':            
            sobol = SobolGridGenerator(self.params, num_samples, seed)
        else:
            raise ValueError("Not supported grid type: {}".format(grid_type))
        
        self.grid = np.asarray(sobol.generate())
        self.hpvs = self.generate()
    
    def get_grid(self):
        return self.grid

    def get_hpv(self):
        #debug("HPV-0: {}".format(self.hpvs[0]))
        return self.hpvs

    def generate(self, return_type='array'):
        debug("Sampling hyperparameter configurations...")
        hps = self.config.hyperparams
        hpv_list = []
        if return_type == 'array':
            for i in range(len(self.grid)):
                vec = self.grid[i]
                hpv = []
                for j in range(len(vec)):
                    param_name = self.params[j]
                    value = vec[j]
                    hp_cfg = getattr(hps, param_name)
                    arg = self.to_param_value(hp_cfg, value)
                    hpv.append(arg)
                #debug("hp{}:{}".format(i, hpv)) 
                hpv_list.append(hpv)
            return hpv_list            
        else: 
            for i in range(len(self.grid)):
                vec = self.grid[i]
                hpv = {}
                for j in range(len(vec)):
                    param_name = self.params[j]
                    value = vec[j]
                    hp_cfg = getattr(hps, param_name)
                    arg = self.to_param_value(hp_cfg, value)
                    hpv[param_name] = arg
                hpv_list.append(hpv)
        
            if return_type == 'table':
                table = pd.DataFrame(data=hpv_list)
                return table
            elif return_type == 'dict':
                return hpv_list
            else:
                raise ValueError("Not supported format: {}".format(return_type))

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
                result = min_value + int(value * (max_value - min_value + 1)) #XXX:to include max value
                if hasattr(hp_cfg, 'power_of'):
                    result = int(np.power(hp_cfg.power_of, result))

            elif hp_cfg.type == 'float':
                result = min_value + (value * (max_value - min_value)) #FIXME:float type can't access max_value.

                if hasattr(hp_cfg, 'power_of'):
                    result = np.power(hp_cfg.power_of, result)
        
        if hp_cfg.type == 'int':
            result = int(result)
        elif hp_cfg.type == 'bool':
            result = bool(result)
        elif hp_cfg.type == 'str':
            result = str(result)
        return result
