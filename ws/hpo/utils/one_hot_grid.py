import os
import numpy as np
from ws.shared.logger import *

def create_one_hot_grid(samples):
    grid = []
    num_samples = samples.get_size()
    for i in range(num_samples):
        s = samples.get_hpv(i)
        c = samples.get_hp_config()

        t = OneHotVectorTransformer(c)
        e = t.transform(s)
        grid.append(np.asarray(e))
    return np.asarray(grid)        


class OneHotVectorTransformer(object):
    
    def __init__(self, config):
        self.config = config

    def transform(self, hpv):
        #debug("sample: {}".format(hpv))
        encoded = []
        for param in self.config.get_param_list():
            vt = self.config.get_value_type(param)

            r = self.config.get_range(param)
            v = hpv[param]
            e = self.encode(vt, r, v)
            #debug("{}: {} ({})".format(param, vt, r))
            #debug("{} -> {}".format(v, e))
            encoded += e
        return encoded

    def encode(self, type, range, value):
        encoded = None
        if type == 'discrete' or type == 'continuous':
            min = range[0]
            max = range[1]
            encoded = [ (float(value - min) / float(max - min)) ] # normalized
        elif type == 'categorical': 
            num_items = len(range)        
            index = self.get_cat_index(range, value)
            vecs = self.create_vectors(num_items)
            encoded = vecs[index]
        elif type == 'preordered':
            base = len(range) - 1
            span = float(1.0 / base)
            index = self.get_cat_index(range, value)
            encoded = [index * span]

        return encoded

    def create_vectors(self, size):
        vecs = []
        for i in range(size):
            vec = [ 0.0 for v in range(size) ]
            vec[i] = 1.0
            vecs.append(vec)
        return vecs

    def get_cat_index(self, category, value):
        num_cat = len(category)
        for i in range(num_cat):
            c = category[i]
            if c == value:
                return i
        raise ValueError("No value {} in category {}".format(value, category))