import os
import time

import numpy as np

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
            elif ts.endswith('d'): # days
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


class VectorGridConverter(object):

    def __init__(self, hpv, candidates, config):
        self.hpv = hpv
        self.candidates = candidates
        self.config = config
        self.param_order = config.get_param_list()

    def get_param_ranges_types(self):
        ranges = []
        types = []
        for param in self.param_order:        
            ranges.append(self.config.get_range(param))
            types.append(self.config.get_type(param))

        return ranges, types    

    def to_grid_vector(self, vector, ranges=None, types=None):
        if ranges == None or types == None:
            ranges, types = self.get_param_ranges_types()
        
        normalized = []
        for i in range(0, len(ranges)):
            value = vector.tolist()[i]
            param_range = ranges[i]
            
            if types[i] == 'str':
                # string type -> index / total items
                index = param_range.index(value)
                normalized.append(float(index + 1) / float(len(param_range)))
            elif types[i] == 'bool':
                if value == True:
                    normalized.append(1.0)
                else:
                    normalized.append(0.0)  
            else:
                # other types
                denominator = param_range[-1] - param_range[0]
                numerator = float(value) - param_range[0]
                normalized.append(float(numerator) / float(denominator))

        return np.array(normalized)

    def get_nearby_index(self, params):
        vec = params
        if type(params) == dict:
            vector_list = []            
            for i in range(len(self.param_order)):
                vector_list.append(params[self.param_order[i]])
            vec = np.array(vector_list)

        closest_dist = 9999999
        nearby_idx = -1
        ranges, types = self.get_param_ranges_types()

        for i in np.nditer(self.candidates):
            compare_vec = self.hpv[i, :]
            norm_vec1 = self.to_grid_vector(vec, ranges, types)
            norm_vec2 = self.to_grid_vector(compare_vec, ranges, types)  
            distance = np.linalg.norm(norm_vec1 - norm_vec2) 
            if distance < closest_dist:
                closest_dist = distance
                nearby_idx = i.tolist()
        
        return nearby_idx, closest_dist

