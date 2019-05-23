from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

import ws.shared.hp_cfg as hconf
from ws.shared.logger import * 

from ws.apis import create_name_server


HP_CONF_PATH = './hp_conf/'
ALL_LOG_LEVELS = ['debug', 'warn', 'error', 'log']


def main():
    try:
        default_log_level = 'warn'
        default_port = 5000    
        parser = argparse.ArgumentParser()
        
        # Optional argument configuration
        parser.add_argument('-hd', '--hconf_dir', default=HP_CONF_PATH, type=str,
                            help='Hyperparameter configuration directory.\n'+\
                            'Default setting is {}'.format(HP_CONF_PATH))

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

        # Check hyperparameter configuration file
        hp_cfg_path = args.hconf_dir + args.hp_config
        hp_cfg = hconf.read_config(hp_cfg_path)
        if hp_cfg is None:
            raise ValueError('Invaild hyperparam config : {}'.format(hp_cfg_path))

        debug("HPO node will be ready to serve...")
        create_name_server(hp_cfg, debug_mode=enable_debug, port=args.port)
    
    except KeyboardInterrupt as ki:
        warn("Terminated by Ctrl-C.")
        sys.exit(-1) 

    except Exception as ex:
        error("Exception ocurred: {}".format(ex))


if __name__ == "__main__":
    main()