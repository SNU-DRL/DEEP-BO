import numpy as np
import pandas as pd

import json
import os
import sys
import traceback

from collections import namedtuple
from ws.shared.logger import *


def read_config(cfg_file):
    hcr = HyperparameterConfigurationReader(cfg_file)
    
    return hcr.get_config()


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
    
    def get_hyperparams(self):
        return self._dict['param_order']

    def get_type(self, name):
        t = None
        hyperparams = self.hyperparams
        if name in hyperparams.__dict__.keys():
            hyperparam = getattr(hyperparams, name)
            if hyperparam.type == 'unicode':
                return "str"
            else:
                return hyperparam.type
        
        return t

    def get_value_type(self, name):
        t = None
        hyperparams = self.hyperparams
        if name in hyperparams.__dict__.keys():
            hyperparam = getattr(hyperparams, name)
            return hyperparam.value_type
        
        return t

    def get_range(self, name):
        range = []
        hyperparams = self.hyperparams
        if name in hyperparams.__dict__.keys():
            hyperparam = getattr(hyperparams, name)
            range = hyperparam.range
            
            if hasattr(hyperparam, 'power_of'):
                base = hyperparam.power_of
                range = []
                for power in hyperparam.range:
                    range.append(base**power)

            if hyperparam.type == 'unicode':
                range = []
                for item in hyperparam.range:
                    range.append(item.encode('ascii', 'ignore'))
                
        return range

    def get_dict(self):
        return self._dict
