
import json
import os
import sys
import traceback

import numpy as np
from collections import namedtuple
from ws.shared.logger import *
from ws.shared.converter import OneHotVectorTransformer

class HyperparameterConfigurationReader(object):
    def __init__(self, cfg_file_name, config_path=""):
        self._dict = {}
        if not cfg_file_name.endswith('.json'):
            cfg_file_name += '.json'
        path ="{}{}".format(config_path, cfg_file_name) 
        if os.path.exists(path):
            self._dict = self.read_json(path)
        else:
            error("hyperparam config not found: {}".format(path))

    def read_json(self, cfg_file_name):
        with open(cfg_file_name) as json_cfg:
            json_dict = json.load(json_cfg)
            return json_dict

    def get_config(self):
        try:
            hc = HyperparameterConfiguration(self._dict)
            if self.validate(hc):
                return hc
        except Exception as ex:
            raise ValueError("Invalid configuration: {}".format(self._dict))
 
    def validate(self, cfg):
        if not hasattr(cfg, 'hyperparams'):
            error('json object does not contain hyperparams attribute: {}'.format(cfg))
            return False

        for hyperparam, conf in cfg.hyperparams.__dict__.items():

            # attribute existence test
            if not hasattr(conf, 'type'):
                error(hyperparam + " has not type attribute.")
                return False
            else:
                supported_types = ['int', 'float', 'str', 'bool', 'unicode']
                if not conf.type in supported_types:
                    return False

            if not hasattr(conf, 'value_type'):
                error(hyperparam + " has not value_type attribute.")
                return False
            else:
                supported_value_types = ['discrete', 'continuous', 'preordered', 'categorical']
                if not conf.value_type in supported_value_types:
                    return False

            if not hasattr(conf, 'range'):
                error(hyperparam + " has not range attribute.")
                return False
            else:
                range_list = conf.range
                if len(range_list) is 0:
                    error(hyperparam + " has no range values")
                    return False

                for value in range_list:
                    value_type_name = type(value).__name__
                    if value_type_name == 'unicode':
                        value_type_name = 'str'
                    if value_type_name != conf.type:                    
                        if not hasattr(conf, 'power_of'):
                            error(hyperparam + " has invalid type item.")
                            return False

        return True


class DictionaryToObject(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [DictionaryToObject(x) 
                    if isinstance(
                        x, dict) else x for x in b])
            else:
                setattr(self, a, DictionaryToObject(b) 
                    if isinstance(b, dict) else b)


class HyperparameterConfiguration(DictionaryToObject):
    def __init__(self, d):
        self._dict = d
        super(HyperparameterConfiguration, self).__init__(d)
    
    def get_param_names(self):
        if 'param_order' in self._dict:
            return self._dict['param_order']
        else:
            # return hyperparameters alphabetical order
            param_list = [ p for p in self.hyperparams.__dict__.keys() ]
            param_list.sort()
            return param_list

    def get_type(self, name):
        if name in self.get_param_names():
            hyperparam = getattr(self.hyperparams, name)
            if hyperparam.type == 'unicode':
                return "str"
            else:
                return hyperparam.type
        raise ValueError("Invalid hyperparameter name: {}".format(name))

    def get_value_type(self, name):
        if name in self.get_param_names():
            hyperparam = getattr(self.hyperparams, name)
            return hyperparam.value_type
        raise ValueError("Invalid hyperparameter name: {}".format(name))

    def get_range(self, name):
        if name in self.get_param_names():
            hyperparam = getattr(self.hyperparams, name)
            r = hyperparam.range
            
            if hasattr(hyperparam, 'power_of'):
                base = hyperparam.power_of
                r = []
                for power in hyperparam.range:
                    r.append(base**power)

            if hyperparam.type == 'unicode':
                r = []
                for item in hyperparam.range:
                    r.append(item.encode('ascii', 'ignore'))
            return r
        else:
            raise ValueError("Invalid hyperparameter name: {}".format(name))

    def get_default_vector(self):
        vec = []
        for name in self.get_param_names():
            hyperparam = getattr(self.hyperparams, name)
            if hasattr(hyperparam, 'default'):
                vec.append(hyperparam.default) 
            else:
                min_val = self.get_range(name)[0]
                debug("No default value setting. Use {} as a minimum value of the range.".format(min_val))
                vec.append(min_val)
        return vec
			
    def get_dict(self):
        return self._dict

    def convert(self, source_type, target_type, value):
        if source_type == 'grid' and target_type == 'hpv_list':
            return self.grid_to_hpv_list(value)
        elif source_type == 'dict' and target_type == 'arr':
            return self.dict_to_array(value, False)
        elif source_type == 'dict' and target_type == 'norm_arr':
            return self.dict_to_array(value, True)
        elif source_type == 'arr' and target_type == 'norm_arr':
            return self.arr_to_norm_vec(value)
        elif source_type == 'arr' and target_type == 'list':
            return self.arr_to_list(value)
        elif source_type == 'arr' and target_type == 'dict':
            return self.arr_to_dict(value)
        elif target_type == 'one_hot':
            return self.to_one_hot_vector(value)
        else:
            raise TypeError("Invalid type.")            
			
    def grid_to_hpv_list(self, grid_list):
        hpvs = []
        p_names = self.get_param_names()
        for i in range(len(grid_list)):
            g = grid_list[i]
            hpv = []
            for j in range(len(g)):
                arg = self.unnormalize(p_names[j], g[j])
                hpv.append(arg)
            hpvs.append(hpv)
        return hpvs
    def dict_to_array(self, hp_dict, normalize):
        arr = []
        for p in self.get_param_names():
            arr.append(hp_dict[p])
        if normalize == True:
            arr = self.arr_to_norm_vec(arr)
        return arr
    def arr_to_list(self, arr):
        typed_list = []
        p_list = self.get_param_names()
        if len(p_list) != len(arr):
            raise TypeError("Invalid hyperparameter vector: {}".format(arr))
        for i in range(len(p_list)):
            p = p_list[i]
            t = self.get_type(p)
            if t == 'int':
                v = int(float(arr[i])) # FIX: float type string raises ValueError
            else:
                v = eval(t)(arr[i])
            typed_list.append(v)
        return typed_list
    def arr_to_dict(self, arr):
        typed_dict = {}
        p_list = self.get_param_names()
        if len(p_list) != len(arr):
            raise TypeError("Invalid hyperparameter vector: {}".format(arr))
        for i in range(len(p_list)):
            p = p_list[i]
            t = self.get_type(p)
            v = eval(t)(arr[i])
            typed_dict[p] = v
        return typed_dict
    def to_one_hot_vector(self, vector):
        one_hot = []
        vector_dict = {}
        p_list = self.get_param_names()
        if isinstance(vector, dict):
            vector_dict = vector
        else:
            vector = self.arr_to_list(vector)
            for i in range(len(vector)):
                k = p_list[i]
                v = vector[i]
                vector_dict[k] = v
        t = OneHotVectorTransformer(self)
        one_hot = t.transform(vector_dict)
        return one_hot
    def arr_to_norm_vec(self, vector):
        value_types = []
        ranges = []
        types = []
        for param in self.get_param_names():
            value_types.append(self.get_value_type(param))        
            ranges.append(self.get_range(param))
            types.append(self.get_type(param))
        if isinstance(vector, dict):
            vector = self.dict_to_list(vector)
        vector = self.arr_to_list(vector)
        p_list = self.get_param_names()
        normalized = []
        for i in range(0, len(vector)):
            param_name = p_list[i]
            value_type = value_types[i]
            type = types[i]
            value = vector[i]
            param_range = ranges[i]
            if value_type != 'categorical' and value_type != 'preordered':
                max_val = param_range[-1]
                min_val = param_range[0]
                denominator = max_val - min_val
                numerator = float(value) - min_val
                normalized.append(float(numerator) / float(denominator))
            else:
                n_v = float(param_range.index(value)) / float(len(param_range))
                normalized.append(n_v)
        return np.array(normalized)
    def get_nearby_index(self, candidates, hpv, params):
        vec = params
        p_list = self.get_param_names()
        if type(params) == dict:
            vector_list = []            
            for i in range(len(p_list)):
                vector_list.append(params[p_list[i]])
            vec = np.array(vector_list)
        closest_dist = 9999999
        nearby_idx = -1
        for i in np.nditer(candidates):
            compare_vec = hpv[i, :]
            norm_vec1 = self.arr_to_norm_vec(vec)
            norm_vec2 = self.arr_to_norm_vec(compare_vec)  
            distance = np.linalg.norm(norm_vec1 - norm_vec2) 
            if distance < closest_dist:
                closest_dist = distance
                nearby_idx = i.tolist()
        return nearby_idx, closest_dist
    def unnormalize(self, param_name, norm_value):
        result = None
        hp_cfg = getattr(self.hyperparams, param_name)
        range_list = hp_cfg.range
        if hp_cfg.value_type == "categorical" or hp_cfg.value_type == 'preordered':
            size = len(range_list)
            index = int(norm_value * size)
            if index == size: # handle the terminal condition
                index = size - 1 
            result = range_list[index]
        else:
            max_value = max(range_list)
            min_value = min(range_list)
            if hp_cfg.type == 'int':
                result = min_value + int(norm_value * (max_value - min_value)) 
                if norm_value == 1.0:
                    result = max_value
                if hasattr(hp_cfg, 'power_of'):
                    result = int(np.power(hp_cfg.power_of, result))
            elif hp_cfg.type == 'float':
                result = min_value + (norm_value * (max_value - min_value)) 
                if norm_value == 1.0:
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
