from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

import ws.hpo.bandit_config as bconf
from ws.shared.read_cfg import *
from ws.shared.logger import * 

from ws.apis import wait_hpo_request


RUN_CONF_PATH = './run_conf/'
HP_CONF_PATH = './hp_conf/'
ALL_LOG_LEVELS = ['debug', 'warn', 'error', 'log']


def main():
    try:
        default_run_config = 'arms.json'
        default_log_level = 'warn'
        default_port = 5001
        default_name_server = "http://127.0.0.1:5000"    
        parser = argparse.ArgumentParser()
        
        # Optional argument configuration
        parser.add_argument('-rd', '--rconf_dir', default=RUN_CONF_PATH, type=str,
                            help='Run configuration directory.\n'+\
                            'Default setting is {}'.format(RUN_CONF_PATH))
        parser.add_argument('-hd', '--hconf_dir', default=HP_CONF_PATH, type=str,
                            help='Hyperparameter configuration directory.\n'+\
                            'Default setting is {}'.format(HP_CONF_PATH))
        parser.add_argument('-rc', '--rconf', default=default_run_config, type=str,
                            help='Run configuration file name existed in {}.\n'.format(RUN_CONF_PATH)+\
                            'Default setting is {}'.format(default_run_config))                                 
        parser.add_argument('-ns', '--ns_url', default=default_name_server, type=str,
                            help='Name server URL to register. Default setting is {}'.format(default_name_server))   

        # Debugging option
        parser.add_argument('-l', '--log_level', default=default_log_level, type=str,
                            help='Print out log level.\n'+\
                            '{} are available. default is {}'.format(ALL_LOG_LEVELS, default_log_level))

        # Mandatory options
        parser.add_argument('hp_config', type=str, help='hyperparameter configuration name.')
        parser.add_argument('port', type=int, help='Port number.')

        args = parser.parse_args()
        set_log_level(args.log_level)
        
        enable_debug = False
        if args.log_level == 'debug':
            enable_debug = True

        run_cfg = read_run_config(args.rconf, path=args.rconf_dir)
        if not bconf.validate(run_cfg):
            error("Invalid run configuration. see {}".format(args.rconf))
            raise ValueError('invaild run configuration.')    

        # Check hyperparameter configuration file
        hp_cfg_path = args.hconf_dir + args.hp_config
        hp_cfg = read_hyperparam_config(hp_cfg_path)
        if hp_cfg is None:
            raise ValueError('Invaild hyperparam config : {}'.format(hp_cfg_path))

        debug("HPO node will be ready to serve...")
        wait_hpo_request(run_cfg, hp_cfg, 
                        debug_mode=enable_debug, port=args.port, register_url=args.ns_url)
    
    except KeyboardInterrupt as ki:
        warn("Terminated by Ctrl-C.")
        sys.exit(-1) 

    except Exception as ex:
        error("Exception ocurred: {}".format(ex))


if __name__ == "__main__":
    main()