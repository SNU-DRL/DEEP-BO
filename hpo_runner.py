from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import traceback
import validators as valid

from ws.shared.lookup import check_lookup_existed
from ws.shared.logger import *
from ws.shared.read_cfg import *

import hpo.bandit_config as bconf
import hpo.bandit as bandit

from hpo.space_mgr import *

ALL_OPT_MODELS = ['SOBOL', 'GP', 'RF', 'TPE', 'GP-NM', 'GP-HLE', 'RF-HLE', 'TPE-HLE']
ACQ_FUNCS = ['RANDOM', 'EI', 'PI', 'UCB']
DIV_SPECS = ['SEQ', 'RANDOM']
ALL_MIXING_SPECS = ['HEDGE', 'BO-HEDGE', 'BO-HEDGE-T', 'BO-HEDGE-LE', 'BO-HEDGE-LET', 
                    'EG', 'EG-LE', 'GT', 'GT-LE', 'SKO']

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

    hp_cfg_path = run_cfg['hp_config_dir']
    
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
        if attr in run_cfg:
            valid[str(attr)] = run_cfg[str(attr)]
        else:
            valid[str(attr)] = value  

    if not "early_term_rule" in run_cfg:
        run_cfg["early_term_rule"] = args.early_term_rule
		
    space_setting = {}
    if not 'search_space' in run_cfg:
        run_cfg['search_space'] = space_setting
    else:
        space_setting = run_cfg['search_space']
		
    if not 'order' in space_setting:
        space_setting['order'] = None
    if not 'num_samples' in space_setting:
        space_setting['num_samples'] = 20000
    if not 'seed' in space_setting:
        space_setting['seed'] = 1
    if not 'preevaluated' in space_setting:
        space_setting['preevaluated'] = False
    if not 'prior_history' in space_setting:
        space_setting['prior_history'] = None
    valid['hp_config'] = hp_cfg
    valid['run_config'] = run_cfg

           
    return valid




def run(args, save=True):
    try:
        
        run_cfg = args['run_config']
        space_set = run_cfg['search_space']
            
        space = None

        m = None        

        result = []
            
            
        if space_set['preevaluated']:
            if not check_lookup_existed(run_cfg['hp_config']):
                raise ValueError('Pre-evaluated configuration not found: {}'.format(run_cfg['hp_config']))
            debug("Creating surrogate space of {}...".format(space_set))
                
            space = create_space_from_table(run_cfg['hp_config'], 
                                                grid_order=space_set['order'])
            m = bandit.create_emulator(space, 
                                        args['exp_crt'], 
                                        args['exp_goal'], 
                                        args['exp_time'],
                                        goal_metric=args['goal_metric'],
                                        num_resume=args['rerun'],
                                        save_internal=args['save_internal'],
                                        run_config=run_cfg)
        else:
            hp_cfg = args['hp_config']
            debug("Search space will be created as {}".format(space_set))

            space = create_surrogate_space(hp_cfg.get_dict(), space_set)


            
            if valid.url(run_cfg['train_node']):
                trainer_url = run_cfg['train_node']
                m = bandit.create_runner(trainer_url, space,
                            args['exp_crt'], 
                            args['exp_goal'], 
                            args['exp_time'],
                            goal_metric= args['goal_metric'],
                            num_resume=args['rerun'],
                            save_internal=args['save_internal'],
                            run_config=run_cfg,
                            hp_config=hp_cfg

                            )
            else:
                raise ValueError("Invalid train node: {}".format(run_cfg["train_node"]))

        if not args['mode'] in ALL_OPT_MODELS + ['DIV']:
            raise ValueError('unsupported mode: {}'.format(args['mode']))
        if not args['spec'] in ACQ_FUNCS + DIV_SPECS + ALL_MIXING_SPECS:
            raise ValueError('unsupported spec: {}'.format(args['spec']))
        result = m.play(args['mode'], args['spec'], args['num_trials'], 
                        save=save)
        m.print_best(result)
    except Exception as ex:
        warn('Exception raised during optimization: {}'.format(ex))     
    
    if space != None:
        space.save() 
    return result



def main(args):
    try:
        valid_args = validate_args(args)
        run(valid_args)    
    except Exception as ex:
        error(ex)
        traceback.print_exc()        
   

if __name__ == "__main__":
    available_models = ALL_OPT_MODELS + ['DIV', 'ADA', 'BATCH']
    all_specs = ACQ_FUNCS + DIV_SPECS + ALL_MIXING_SPECS
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
 
    parser.add_argument('run_config', type=str, default='debug', help='Run configuration name.')

    args = parser.parse_args()    
    main(args)
