from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import traceback
import validators as valid

from ws.shared.logger import *
from ws.shared.read_cfg import *

import ws.hpo.bandit_config as bconf
import ws.hpo.bandit as bandit
import ws.hpo.batch_sim as batch
import ws.hpo.space_mgr as space

ALL_OPT_MODELS = ['SOBOL', 'GP', 'RF', 'TPE', 'GP-NM', 'GP-HLE', 'RF-HLE', 'TPE-HLE']
ACQ_FUNCS = ['RANDOM', 'EI', 'PI', 'UCB']
DIV_SPECS = ['SEQ', 'RANDOM']
ALL_MIXING_SPECS = ['HEDGE', 'BO-HEDGE', 'BO-HEDGE-T', 'BO-HEDGE-LE', 'BO-HEDGE-LET', 
                    'EG', 'EG-LE', 'GT', 'GT-LE', 'SKO']
BATCH_SPECS = ['SYNC', 'ASYNC']
    
LOOKUP_PATH = './lookup/'
RUN_CONF_PATH = './run_conf/'


def validate_args(args):

    surrogate = None
    valid = {}

    run_cfg = read_run_config(args.run_config, path=args.rconf_dir)
    if not bconf.validate(run_cfg):
        error("Invalid run configuration. see {}".format(args.rconf))
        raise ValueError('invaild run configuration.')    
    
    if run_cfg['debug_mode']:          
        set_log_level('debug')

    hp_cfg_path = run_cfg['hp_config_path']
    
    if run_cfg['hp_config'].endswith('.json'):
        # hp_config is a json file
        hp_cfg_path += run_cfg['hp_config']
    else:
        hp_cfg_path += (run_cfg['hp_config'] + ".json")
   
    hp_cfg = read_hyperparam_config(hp_cfg_path)
    if hp_cfg is None:
        raise ValueError('Invaild hyperparam config : {}'.format(hp_cfg_path))

    #debug("run configuration: {}".format(run_cfg))


    for attr, value in vars(args).items():
        valid[str(attr)] = value  

    valid['hp_config'] = hp_cfg
    valid['run_config'] = run_cfg
    valid['preevaluated'] = False
    if 'preevaluated' in run_cfg:
        valid['preevaluated'] = run_cfg['preevaluated']
           
    return valid


def check_lookup_existed(name):
    for csv in os.listdir(LOOKUP_PATH):
        if str(csv) == '{}.csv'.format(name):
            return True
    return False


def execute(args, save_results=False):
    try:
        num_resume = 0
        save_internal = False
        if 'rerun' in args:
            num_resume = args['rerun']
        if 'save_internal' in args:
            save_internal = args['save_internal']
        
        result = []
        run_cfg = args['run_config']

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
            
            
            if args['preevaluated'] == True:
                if not check_lookup_existed(run_cfg['hp_config']):
                    raise ValueError('Pre-evaluated configuration not found: {}'.format(run_cfg['hp_config']))
                debug("Create surrogate space: order-{}, one hot-{}, seed-{}".format(grid_order, one_hot, grid_seed))
                
                samples = space.create_surrogate_space(args['hp_config'], 
                            grid_order=grid_order, 
                            one_hot=one_hot)
            else:
                hp_cfg = args['hp_config']
                debug("Creating parameter space with seed:{}, one-hot:{}, samples:{}".format(grid_seed, one_hot, num_samples))

                samples = space.create_grid_space(hp_cfg.get_dict(),
                                                  num_samples=num_samples,
                                                  grid_seed=grid_seed,
                                                  one_hot=one_hot)

            if args["early_term_rule"] != "None":
                run_cfg["early_term_rule"] = args["early_term_rule"]

            
            if valid.url(run_cfg['train_node']):
                trainer = run_cfg['train_node']
                m = bandit.create_runner(trainer, samples,
                            args['exp_crt'], args['exp_goal'], args['exp_time'],
                            goal_metric= args['goal_metric'],
                            num_resume=num_resume,
                            save_internal=save_internal,
                            run_config=run_cfg,
                            hp_config=hp_cfg,
                            use_surrogate=use_surrogate)
            else:
                m = bandit.create_emulator(samples, 
                            args['exp_crt'], args['exp_goal'], args['exp_time'],
                            goal_metric= args['goal_metric'],
                            num_resume=num_resume,
                            save_internal=save_internal,
                            run_config=run_cfg)

            if args['mode'] == 'DIV' or args['mode'] == 'ADA':
                result = m.mix(args['spec'], args['num_trials'], 
                               save_results=save_results)
            elif args['mode'] in ALL_OPT_MODELS:
                result = m.all_in(args['mode'], args['spec'], args['num_trials'], 
                                  save_results=save_results)
            else:
                raise ValueError('unsupported mode: {}'.format(args['mode']))
        elif args['mode'] == 'BATCH':            
            c = batch.get_simulator(args['spec'].upper(), 
                                    args['surrogate'],
                                    args['exp_crt'], 
                                    args['exp_goal'], 
                                    args['exp_time'], 
                                    args['run_cfg'])
            result = c.run(args['num_trials'], save_results=True)
        else:
            raise ValueError("upsupported run mode: {}".format(args['mode']))
    except Exception as ex:
        warn('{} occurs on executing SMBO'.format(ex))     
    
    return result


def main(args):
    try:
        valid_args = validate_args(args)
        results = execute(valid_args, save_results=True)    
    except Exception as ex:
        error(ex)
        traceback.print_exc()        
   

if __name__ == "__main__":
    available_models = ALL_OPT_MODELS + ['DIV', 'ADA', 'BATCH']
    all_specs = ACQ_FUNCS + DIV_SPECS + ALL_MIXING_SPECS + BATCH_SPECS
    default_model = 'DIV'
    default_spec = 'SEQ'
    default_target_goal = 0.0
    default_goal_metric = 'error'
    default_expired_time = '1d'
    default_early_term_rule = "DecaTercet"
        
    default_exp_criteria = 'TIME'

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
                        help='The expected target goal. ' +\
                        'When it is achieved, this trial will be terminated automatically. '+\
                        'Default setting is {}.'.format(default_target_goal))
    parser.add_argument('-gm', '--goal_metric', default=default_goal_metric, type=str,
                        help='The target goal metric. "error" or "accuracy" can be applied. ' +\
                        'Default setting is {}.'.format(default_goal_metric))

    parser.add_argument('-et', '--exp_time', default=default_expired_time, type=str,
                        help='The time each trial expires. When the time is up, '+\
                        'it is automatically terminated. Default setting is {}.'.format(default_expired_time))
    parser.add_argument('-etr', '--early_term_rule', default=default_early_term_rule, type=str,
                        help='Early termination rule. Default setting is {}.'.format(default_early_term_rule))
    
    # Configurations
    parser.add_argument('-rd', '--rconf_dir', default=RUN_CONF_PATH, type=str,
                        help='Run configuration directory.\n'+\
                        'Default setting is {}'.format(RUN_CONF_PATH))                        
    parser.add_argument('-nt', '--num_trials', type=int, default=1,
                        help='The total number of runs for this experiment.')

    # XXX:below options are for experimental use.  
    parser.add_argument('-rr', '--rerun', default=0, type=int,
                        help='[Experimental] Use to expand the number of trials for the experiment. zero means no rerun. default is {}.'.format(0))
    parser.add_argument('-pkl', '--save_internal', default=False, type=bool,
                        help='[Experimental] Whether to save internal values into a pickle file.\n' + 
                        'CAUTION:this operation requires very large storage space! Default value is {}.'.format(False))                        
#    parser.add_argument('-tp', '--time_penalty', default=default_time_penalty, type=str,
#                        help='[Experimental] Time penalty strategies for acquistion function.\n'+\
#                        '{} are available. Default setting is {}'.format(time_penalties_modes, default_time_penalty))
 
    parser.add_argument('run_config', type=str, help='Run configuration name.')

    args = parser.parse_args()    
    main(args)
