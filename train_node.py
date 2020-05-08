import os
import time

from ws.apis import *
from ws.shared.read_cfg import *
from ws.shared.logger import *

import argparse
import validators as v

from samples.targets import * # load object functions


def main(run_config):    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        master_node = None
        if "master_node" in run_config:
            if type(run_config['master_node']) == dict and 'url' in run_config['master_node']:
                master_node = run_config['master_node']['url']
            elif type(run_config['master_node']) == str:
                master_node = run_config['master_node']
            if master_node != None and v.url(master_node):
                if master_node.endswith('/'):
                    master_node += master_node[:-1]
            else:
                raise ValueError("Invalid master URL: {}".format(master_node))



        debug_mode = False
        if "debug_mode" in run_config:
            if run_config["debug_mode"]:
                debug_mode = True
                set_log_level('debug')
                print_trace()

        hp_config_dir = "./hp_conf/"
        if "hp_config_dir" in run_config:
            hp_config_dir = run_config["hp_config_dir"]         
       
        hp_cfg_file = run_config["hp_config"]
        hp_cfg_path = '{}{}.json'.format(hp_config_dir, hp_cfg_file)
        hp_cfg = read_hyperparam_config(hp_cfg_path)
        resource_type = "cpu"
        resource_id = ""

        log_file_name = 'train.log'
        credential = None
        if "credential" in run_config:
            credential = run_config['credential']
        else:
            raise ValueError("No credential info in run configuration")
        if not 'train_node' in run_config:
            raise ValueError("No train node configuarion")
        self_conf = run_config['train_node']
        port = 6000
        if "port" in self_conf:
            port = self_conf["port"]
        
        if "resource_type" in self_conf:
            resource_type = self_conf["resource_type"]  
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        if resource_type == "gpu":            
            if "resource_id" in self_conf:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self_conf["resource_id"])
        else: 
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # visible CPU only
        
        if "resource_id" in self_conf:
            resource_id = self_conf["resource_id"]

        if 'log' in self_conf:
            log_file_name = self_conf['log']
        else:
            log_file_name = "train_{}{}.log".format(resource_type, resource_id)
        set_log_file(log_file_name)
        r_id = "{}{}".format(resource_type, resource_id)
        get_resource().set_id(r_id) # to indentify the worker's resource

        eval_func = eval(self_conf["eval_func"])
        log("Evaluation will be performed via {}...".format(self_conf["eval_func"]))

        wait_train_request(eval_func, 
                           hp_cfg, 
                           debug_mode=debug_mode,
                           device_type=resource_type,
                           device_index=resource_id,
                           master_node=master_node,
                           credential=credential, 
                           port=port)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    run_conf_path = './run_conf/'
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--rconf_dir', default=run_conf_path, type=str,
                        help='Run configuration directory.\n'+\
                        'Default setting is {}'.format(run_conf_path))  
    parser.add_argument('run_config', type=str, help='run configuration name.') 
    args = parser.parse_args()
    run_cfg = read_run_config(args.run_config, args.rconf_dir)       
    
    main(run_cfg)

