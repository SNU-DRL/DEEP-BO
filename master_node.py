from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

import ws.shared.hp_cfg as hconf
from ws.shared.logger import *
from ws.shared.read_cfg import * 

from ws.apis import create_master_server


HP_CONF_PATH = './hp_conf/'
ALL_LOG_LEVELS = ['debug', 'warn', 'error', 'log']

def main(run_config):
    try:
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

        port = 5000
        if "port" in run_config:
            port = run_config["port"]

        credential = None
        if "credential" in run_config:
            credential = run_config['credential']
        else:
            raise ValueError("No credential info in run configuration")

        debug("Master node will be ready to serve...")
        create_master_server(hp_cfg, 
                             debug_mode=debug_mode,
                             credential=credential, 
                             port=port)
    
    except KeyboardInterrupt as ki:
        warn("Terminated by Ctrl-C.")
        sys.exit(-1) 

    except Exception as ex:
        error("Exception ocurred: {}".format(ex))


if __name__ == "__main__":
    run_conf_path = './run_conf/'    
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--rconf_dir', default=run_conf_path, type=str,
                        help='Run configuration directory.\n'+\
                        'Default setting is {}'.format(run_conf_path))
    parser.add_argument('run_config', type=str, help='Run configuration name.')
    args = parser.parse_args()
    run_cfg = read_run_config(args.run_config, args.rconf_dir)    
    
    main(run_cfg)
