import json

from ws.shared.logger import *
from ws.shared.hp_cfg import HyperparameterConfigurationReader


def read_run_config(cfg_file_name, path='run_conf/'):
    try:
        if not cfg_file_name.endswith('.json'):
            cfg_file_name += ".json"
        with open(path + cfg_file_name) as json_cfg:
            json_dict = json.load(json_cfg)
            return json_dict   
    except Exception as ex:
        error('Exception on read json: {}'.format(path + cfg_file_name))
        raise ValueError('config file not found.')


def read_hyperparam_config(cfg_file):
    hcr = HyperparameterConfigurationReader(cfg_file)
    
    return hcr.get_config()

