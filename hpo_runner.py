from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import validators as valid

from ws.shared.logger import *
import ws.shared.hp_cfg as hconf

import ws.hpo.bandit_config as rconf
import ws.hpo.bandit as bandit
import ws.hpo.batch_sim as batch
import ws.hpo.space_mgr as space

ALL_OPT_MODELS = ['SOBOL', 'GP', 'RF', 'TPE', 'GP-NM', 'GP-HLE', 'RF-HLE', 'TPE-HLE']
ACQ_FUNCS = ['RANDOM', 'EI', 'PI', 'UCB']
DIV_SPECS = ['SEQ', 'RANDOM']
ALL_MIXING_SPECS = ['HEDGE', 'BO-HEDGE', 'BO-HEDGE-T', 'BO-HEDGE-LE', 'BO-HEDGE-LET', 
                    'EG', 'EG-LE', 'GT', 'GT-LE',
                    'SKO']

BATCH_SPECS = ['SYNC', 'ASYNC']
ALL_LOG_LEVELS = ['debug', 'warn', 'error', 'log']
    
LOOKUP_PATH = './lookup/'
RUN_CONF_PATH = './run_conf/'
HP_CONF_PATH = './hp_conf/'


def validate_args(args):

    surrogate = None
    valid = {}
    
    hp_cfg_path = args.hconf_dir
    if args.hp_config.endswith('.json'):
        # hp_config is a json file
        hp_cfg_path += args.hp_config
    else:
        # hp_config is surrogate name, check whether same csv file exists in lookup folder.
        if not check_lookup_existed(args.hp_config):
            debug('Invaild arguments: No {} in {}'.format(args.hp_config, LOOKUP_PATH))            
        else:
            surrogate = args.hp_config
            hp_cfg_path += (args.hp_config + ".json")
    
    hp_cfg = hconf.read_config(hp_cfg_path)
    if hp_cfg is None:
        raise ValueError('Invaild hyperparam config : {}'.format(hp_cfg_path))
              
    run_cfg = rconf.read(args.rconf, path=args.rconf_dir)
    if not rconf.validate(run_cfg):
        error("Invalid run configuration. see {}".format(args.rconf))
        raise ValueError('invaild run configuration.')    
    
    valid['surrogate'] = surrogate
    valid['hp_cfg'] = hp_cfg
    #valid['run_cfg'] = run_cfg
    
    valid['exp_time'] = args.exp_time

    for attr, value in vars(args).items():
        valid[str(attr)] = value        
           
    return run_cfg, valid


def check_lookup_existed(name):
    for csv in os.listdir(LOOKUP_PATH):
        if str(csv) == '{}.csv'.format(name):
            return True
    return False


def execute(run_cfg, args, save_results=False):
    try:
        if run_cfg is None:
            if "run_cfg" in args:
                run_cfg = args['run_cfg']
            else:
                raise ValueError("No run configuration found.")
        
        num_resume = 0
        save_internal = False
        if 'rerun' in args:
            num_resume = args['rerun']
        if 'save_internal' in args:
            save_internal = args['save_internal']
        
        result = []
        
        if args['mode'].upper() != 'BATCH':
            m = None
            hp_cfg = None 
            use_surrogate = None

            trainer = None
            samples = None

            grid_order = None 
            one_hot = True # Set one hot coding as default for preparing final revision
            num_samples = 20000
            grid_seed = 1
            if 'grid' in run_cfg:
                if 'order' in run_cfg['grid']:
                    grid_order = run_cfg['grid']['order']
                if 'one_hot' in run_cfg['grid']:
                    one_hot = run_cfg['grid']['one_hot']
                if 'num_samples' in run_cfg['grid']:
                    num_samples = run_cfg['grid']['num_samples']
                if 'grid_seed' in run_cfg['grid']:
                    grid_seed = run_cfg['grid']['grid_seed']                       
            
            
            if 'surrogate' in args and args['surrogate'] != None:
                debug("Create surrogate space: order-{}, one hot-{}, seed-{}".format(grid_order, one_hot, grid_seed))
                use_surrogate = args['surrogate']
                path = "{}{}.json".format(args['hconf_dir'], use_surrogate)
                
                hp_cfg = hconf.read_config(path)
                samples = space.create_surrogate_space(use_surrogate, 
                            grid_order=grid_order, 
                            one_hot=one_hot)
                
            elif 'hp_cfg' in args and args['hp_cfg'] != None:
                hp_cfg = args['hp_cfg']
                debug("Create grid space: seed-{}, one hot-{}, # of samples - {}".format(grid_seed, one_hot, num_samples))

                samples = space.create_grid_space(hp_cfg.get_dict(),
                            num_samples=num_samples,
                            grid_seed=grid_seed,
                            one_hot=one_hot)
            else:
                raise ValueError("Invalid arguments: {}".format(args))

            if args["early_term_rule"] != "None":
                run_cfg["early_term_rule"] = args["early_term_rule"]

            if valid.url(args['worker_url']):
                trainer = args['worker_url']
                    
                m = bandit.create_runner(trainer, samples,
                            args['exp_crt'], args['exp_goal'], args['exp_time'],
                            num_resume=num_resume,
                            save_internal=save_internal,
                            run_config=run_cfg,
                            hp_config=hp_cfg,
                            use_surrogate=use_surrogate)
            else:
                m = bandit.create_emulator(samples, 
                            args['exp_crt'], args['exp_goal'], args['exp_time'],
                            num_resume=num_resume,
                            save_internal=save_internal,
                            run_config=run_cfg)

            if args['mode'] == 'DIV' or args['mode'] == 'ADA':
                result = m.mix(args['spec'], args['num_trials'], save_results=save_results)
            elif args['mode'] in ALL_OPT_MODELS:
                result = m.all_in(args['mode'], args['spec'], args['num_trials'], save_results=save_results)
            else:
                raise ValueError('unsupported mode: {}'.format(args['mode']))
        else:
            raise ValueError("BATCH mode is not upsupported here.")
    except:
        warn('Exception occurs on executing SMBO:\n{}'.format(sys.exc_info()))     
    
    return result


def main():

    available_models = ALL_OPT_MODELS + ['DIV', 'ADA', 'BATCH']
    default_model = 'DIV'
    
    all_specs = ACQ_FUNCS + DIV_SPECS + ALL_MIXING_SPECS + BATCH_SPECS
    default_spec = 'SEQ'

    default_target_goal = 0.9999
    default_expired_time = '10d'
    default_early_term_rule = "None"
    
    exp_criteria = ['TIME', 'GOAL']
    default_exp_criteria = 'TIME'
    

    default_run_config = 'arms.json'
    default_log_level = 'warn'

    parser = argparse.ArgumentParser()

    # Hyperparameter optimization methods
    parser.add_argument('-m', '--mode', default=default_model, type=str,
                        help='The optimization mode. Set a model name to use a specific model only.' +\
                        'Set DIV to sequential diverification mode. Set BATCH to parallel mode. {} are available. default is {}.'.format(available_models, default_model))
    parser.add_argument('-s', '--spec', default=default_spec, type=str,
                        help='The detailed specification of the given mode. (e.g. acquisition function)' +\
                        ' {} are available. default is {}.'.format(all_specs, default_spec))
    
    # Termination criteria
    parser.add_argument('-e', '--exp_crt', default=default_exp_criteria, type=str,
                        help='Expiry criteria of the trial.\n Set "TIME" to run each trial until given exp_time expired.'+\
                        'Or Set "GOAL" to run until each trial achieves exp_goal.' +\
                        'Default setting is {}.'.format(default_exp_criteria))                        
    parser.add_argument('-eg', '--exp_goal', default=default_target_goal, type=float,
                        help='The expected target goal accuracy. ' +\
                        'When it is achieved, the trial will be terminated automatically. '+\
                        'Default setting is {}.'.format(default_target_goal))
    parser.add_argument('-et', '--exp_time', default=default_expired_time, type=str,
                        help='The time each trial expires. When the time is up, '+\
                        'it is automatically terminated. Default setting is {}.'.format(default_expired_time))
    parser.add_argument('-etr', '--early_term_rule', default=default_early_term_rule, type=str,
                        help='Early termination rule. Default setting is {}.'.format(default_early_term_rule))
    
    # Configurations
    parser.add_argument('-rd', '--rconf_dir', default=RUN_CONF_PATH, type=str,
                        help='Run configuration directory.\n'+\
                        'Default setting is {}'.format(RUN_CONF_PATH))                        
    parser.add_argument('-hd', '--hconf_dir', default=HP_CONF_PATH, type=str,
                        help='Hyperparameter configuration directory.\n'+\
                        'Default setting is {}'.format(HP_CONF_PATH))
    parser.add_argument('-rc', '--rconf', default=default_run_config, type=str,
                        help='Run configuration file in {}.\n'.format(RUN_CONF_PATH)+\
                        'Default setting is {}'.format(default_run_config))
    parser.add_argument('-w', '--worker_url', default='none', type=str,
                        help='Remote training worker node URL.\n'+\
                        'Set the valid URL if remote training required.') 

    # Debugging option
    parser.add_argument('-l', '--log_level', default=default_log_level, type=str,
                        help='Print out log level.\n'+\
                        '{} are available. default is {}'.format(ALL_LOG_LEVELS, default_log_level))
    
    # XXX:below options are for experimental use.  
    parser.add_argument('-r', '--rerun', default=0, type=int,
                        help='[Experimental] Use to expand the number of trials for the experiment. zero means no rerun. default is {}.'.format(0))
    parser.add_argument('-p', '--save_internal', default=False, type=bool,
                        help='[Experimental] Whether to save internal values into a pickle file.\n' + 
                        'CAUTION:this operation requires very large storage space! Default value is {}.'.format(False))                        
#    parser.add_argument('-tp', '--time_penalty', default=default_time_penalty, type=str,
#                        help='[Experimental] Time penalty strategies for acquistion function.\n'+\
#                        '{} are available. Default setting is {}'.format(time_penalties_modes, default_time_penalty))
 
    parser.add_argument('hp_config', type=str, help='hyperparameter configuration name.')
    parser.add_argument('num_trials', type=int, help='The total repeats of the experiment.')

    args = parser.parse_args()

    set_log_level(args.log_level)

    run_cfg, args = validate_args(args)

    if args['mode'] == 'BATCH':
        if args['worker_url'] is 'none':
            c = batch.get_simulator(args['spec'].upper(), args['surrogate'],
                                args['exp_crt'], args['exp_goal'], 
                                args['exp_time'], run_cfg)
            result = c.run(args['num_trials'], save_results=True)
        else:
            raise NotImplementedError("This version only supports simulation of parallelization via surrogates.")
    else:
        results = execute(run_cfg, args, save_results=True)        
   

if __name__ == "__main__":
    main()
