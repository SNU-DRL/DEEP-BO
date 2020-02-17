import os

from ws.shared.logger import *
from ws.shared.hp_cfg import HyperparameterConfigurationReader


def read_run_config(cfg_file_name, path='run_conf/'):
    try:
        if not os.path.exists(path):
            raise ValueError("Run config path not existed: {}".format(path))
			
        if not os.path.exists(path + cfg_file_name):
            if os.path.exists(path + cfg_file_name + ".yaml"):
                cfg_file_name += ".yaml"
            elif os.path.exists(path + cfg_file_name + ".yml"):
                cfg_file_name += ".yml"
            elif os.path.exists(path + cfg_file_name + ".json"):
                cfg_file_name += ".json"
				
        with open(path + cfg_file_name) as cfg_file:
            if ".yaml" in cfg_file_name or ".yml" in cfg_file_name:
                import yaml 
                json_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
            elif ".json" in cfg_file_name:
                import json
                json_dict = json.load(cfg_file)
            else:
                raise ValueError("Invalid run config format")
            return json_dict   
			
    except Exception as ex:
        error('Exception on loading run configuration: {}'.format(ex))
        raise ValueError('{} file not found.'.format(cfg_file_name))


def read_hyperparam_config(cfg_file):
    hcr = HyperparameterConfigurationReader(cfg_file)
    
    return hcr.get_config()

