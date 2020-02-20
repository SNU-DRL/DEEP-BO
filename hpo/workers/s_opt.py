import sys
import time
import copy
import validators as valid

from ws.shared.logger import *
from ws.shared.read_cfg import read_hyperparam_config
from ws.shared.worker import Worker 
import ws.shared.lookup as lookup

import hpo.space_mgr as space
import hpo.bandit as bandit
from hpo_runner import ALL_OPT_MODELS

class SequentialOptimizer(Worker):
    
    def __init__(self, run_config, hp_config, id, hp_dir="hp_conf/"):
        
        super(SequentialOptimizer, self).__init__(id)
        self.rconf = run_config
        self.hconf = hp_config
        self.hp_dir = hp_dir

        self.type = 'tuner'
        if 'title' in run_config:
            self.id = run_config['title'].replace(" ", "_")

        self.device_id = 'cpu0'
        self.machine = None
        self.search_space = None

        self.reset()

    def get_device_id(self):
        return self.device_id

    def get_config(self):
        return self.rconf

    def set_params(self, params, index=None):
        if params:
            self.params = params
            
            return True
        else:
            debug("invalid params: {}".format(params))
            return False

    def get_search_space(self):
        if self.search_space == None:
            warn("Search space space is not initialized.")
        return self.search_space
        
    def start(self):
        if self.params is None:
            error('Set configuration properly before starting.')
            return
        else:
            super(SequentialOptimizer, self).start()

    def reset(self):
        self.results = []

    def get_cur_result(self):
        if len(self.results) == 0:
            if self.machine != None:
                latest = self.machine.get_results()
                #debug("current result: {}".format(latest))
            else:
                latest = {}
        else:
            latest = self.results
        result = {"result": latest}
        return result

    def execute(self):
        try:
            self.results = self.run(self.rconf, self.hconf, self.params)

        except Exception as ex:
            warn("{} occurs".format(sys.exc_info()[0]))            
            self.stop_flag = True

        finally:
            with self.thread_cond:
                self.busy = False
                self.params = None
                self.thread_cond.notify()

    def stop(self):
        if self.machine != None:
            self.machine.stop()

        super(SequentialOptimizer, self).stop()
 
    def run(self, run_cfg, hp_cfg, args, save_results=False):
        debug("Run sequential optimization with {}".format(run_cfg))    
        num_resume = 0
        save_internal = False
        if 'rerun' in args:
            num_resume = args['rerun']
        if 'save_internal' in args:
            save_internal = args['save_internal']

        results = []
        s_name = None
        self.search_space = None
        
        if 'surrogate' in args and args['surrogate'] != 'None':
            s_name = args['surrogate']
            hp_path = "{}{}.json".format(self.hp_dir, s_name)            
            hp_cfg = read_hyperparam_config(hp_path) # FIXME:rewrite here
            if hp_cfg == None:
                ValueError("Surrogate {} configuration not found.".format(s_name))
        
        if 'space_id' in args:            
            space_id = args['space_id']
            history_url = "{}/spaces/{}/".format(run_cfg["master_node"], space_id)
            debug("Global history: {}".format(history_url))
            if valid.url(history_url):
                self.search_space = space.connect_remote_space(history_url, 
                                                          run_cfg["credential"])
        else:
            warn("No valid space ID: {}".format(space_id))
            self.search_space = space.create_space_from_table(args['surrogate'])

        if self.search_space == None:
            raise ValueError("Invalid parameter space. Space is not initialized properly")

        goal_metric = 'error'
        if 'goal_metric' in args:
            goal_metric = args['goal_metric'] 

        if 'train_node' in args:
            if valid.url(args['train_node']):
                self.machine = bandit.create_runner(args['train_node'], 
                                                    self.search_space,
                                                    args['exp_crt'], 
                                                    args['exp_goal'], args['exp_time'],
                                                    run_cfg, hp_cfg,
                                                    goal_metric=goal_metric,                            
                                                    num_resume=num_resume,
                                                    save_internal=save_internal,
                                                    use_surrogate=s_name,
                                                    id=self.id)
                
            else:
                raise ValueError("Invalid node URL: {}".format(args["train_node"]))
        else:

            self.machine = bandit.create_emulator(self.search_space,
                                                  args['exp_crt'], 
                                                  args['exp_goal'], 
                                                  args['exp_time'],
                                                  goal_metric=goal_metric,
                                                  num_resume=num_resume,
                                                  save_internal=save_internal, 
                                                  run_config=run_cfg,
                                                  id=self.id + "_emul")

        results = self.machine.play(args['mode'], args['spec'], args['num_trials'], 
                save_results=save_results)
          
        return results