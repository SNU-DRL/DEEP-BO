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
        # hpv is dict type
        
        encoded = []
        for param in self.config.get_param_names():
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

