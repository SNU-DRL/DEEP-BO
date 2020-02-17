import os
import time

import numpy as np
from ws.shared.logger import *

class TimestringConverter(object):
    def __init__(self, *args, **kwargs):
        pass

    def convert(self, time_string):
        secs = 0
        try:
            ts = str(time_string)
            # convert to integer
            if ts.endswith('m'): # minutes
                ts = time_string[:-1]
                secs = int(ts) * 60
            elif ts.endswith('h'): # hours
                ts = time_string[:-1]
                secs = int(ts) * 60 * 60
            elif ts.endswith('d'): # hours
                ts = time_string[:-1]
                secs = int(ts) * 60 * 60 * 24
            elif ts.endswith('w'): # weeks
                ts = time_string[:-1]
                secs = int(ts) * 60 * 60 * 24 * 7                
            else:
                secs = int(time_string)                               

        except ValueError:
            raise ValueError('Invaild --exp_time argument: {}.'.format(time_string))
        
        return secs    


class OneHotVectorTransformer(object):
    
    def __init__(self, config):
        self.config = config

    def transform(self, hpv):
        #debug("sample: {}".format(hpv))
        encoded = []
        for param in self.config.get_param_list():
            vt = self.config.get_value_type(param)
            t = self.config.get_type(param)            
            r = self.config.get_range(param)
            v = hpv[param]
            e = self.encode(vt, t, r, v)
            #debug("{}: {} ({})".format(param, vt, r))
            #debug("{} -> {}".format(v, e))
            encoded += e
        return encoded

    def encode(self, value_type, type, range, value):
        encoded = None
        if value_type == 'discrete' or value_type == 'continuous':
            min = range[0]
            max = range[1]
            encoded = [ ((float(value) - min) / (float(max) - min)) ] # normalized
        elif value_type == 'categorical': 
            num_items = len(range)        
            index = self.get_cat_index(type, range, value)
            vecs = self.create_vectors(num_items)
            encoded = vecs[index]
        elif value_type == 'preordered':
            base = len(range) - 1
            span = float(1.0 / base)
            index = self.get_cat_index(type, range, value)
            encoded = [index * span]

        return encoded

    def decode(self, value_type, type, range, value):
        decoded = None
        if value_type == 'discrete' or value_type == 'continuous':
            min = range[0]
            max = range[-1]
            decoded = (max - min) * value + min
            # fix rounding up error
            if decoded > max:
                decoded = max
              
        elif value_type == 'categorical': 
            num_item = int(len(range) * value)       
            decoded = range[num_item]
            
        elif value_type == 'preordered':
            item_id = int(len(range) * value) - 1      
            decoded = range[item_id]

        return decoded


    def create_vectors(self, size):
        vecs = []
        for i in range(size):
            vec = [ 0.0 for v in range(size) ]
            vec[i] = 1.0
            vecs.append(vec)
        return vecs

    def cast(self, type, value):
        if type == "bool":
            value = bool(value)        
        elif type == "int":
            value = int(value)
        elif type == 'float':
            value = float(value)
        elif type == 'str':
            value = str(value)
        else:
            raise TypeError("Invalid type {}, value {}".format(type, value))
        return value

    def get_cat_index(self, type, category, value):
        num_cat = len(category)
        value = self.cast(type, value)        
        for i in range(num_cat):
            c = self.cast(type, category[i])
            if c == value:
                return i
            
        raise ValueError("No value {} in category {}".format(value, category))


class VectorGridConverter(object):

    def __init__(self, config):        
        self.config = config
        self.param_order = config.get_param_list()

    def get_param_spec(self):
        value_types = []
        ranges = []
        types = []
        for param in self.param_order:
            value_types.append(self.config.get_value_type(param))        
            ranges.append(self.config.get_range(param))
            types.append(self.config.get_type(param))

        return ranges, types, value_types    

    def to_grid_array(self, hp_dict):
        # transform Sobol-like grid
        arr = []
        for p in self.param_order:
            arr.append(hp_dict[p])
        arr = self.to_norm_vector(arr, one_hot=False)
        return arr

    def to_norm_vector(self, vector, ranges=None, types=None, one_hot=True):
        if ranges == None or types == None:
            ranges, types, value_types = self.get_param_spec()
        
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()        
        
        normalized = []
        if one_hot == True:
            # one-hot encoding
            t = OneHotVectorTransformer(self.config)
            vector_dict = {}
            for i in range(len(vector)):
                k = self.param_order[i]
                v = vector[i]
                vector_dict[k] = v
            normalized = t.transform(vector_dict)
        else:
            # min-max normalization
            for i in range(0, len(vector)):
                param_name = self.param_order[i]
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
                    #debug("Categorical/preordered type in config: {}({})".format(param_name, value_type))
                    n_v = float(param_range.index(value) / len(param_range))
                    normalized.append(n_v)


        return np.array(normalized)

    def get_nearby_index(self, candidates, hpv, params):
        vec = params
        if type(params) == dict:
            vector_list = []            
            for i in range(len(self.param_order)):
                vector_list.append(params[self.param_order[i]])
            vec = np.array(vector_list)

        closest_dist = 9999999
        nearby_idx = -1
        ranges, types, _ = self.get_param_spec()

        for i in np.nditer(candidates):
            compare_vec = hpv[i, :]
            norm_vec1 = self.to_norm_vector(vec, ranges, types)
            norm_vec2 = self.to_norm_vector(compare_vec, ranges, types)  
            distance = np.linalg.norm(norm_vec1 - norm_vec2) 
            if distance < closest_dist:
                closest_dist = distance
                nearby_idx = i.tolist()
        
        return nearby_idx, closest_dist

