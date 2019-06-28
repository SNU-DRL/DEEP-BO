import sys
import time
import copy
import validators as valid

from ws.shared.logger import *
import ws.shared.hp_cfg as hconf
from ws.shared.worker import Worker 
import ws.shared.lookup as lookup

from ws.hpo.utils.grid_gen import *

import ws.hpo.connectors.remote_space as remote
import ws.hpo.space_mgr as space

import ws.hpo.bandit as bandit
from hpo_runner import ALL_OPT_MODELS

class SequentialOptimizer(Worker):
    
    def __init__(self, run_config, hp_config, id, hp_dir="hp_conf/"):
        
        super(SequentialOptimizer, self).__init__(id)
        self.rconf = run_config
        self.hconf = hp_config
        self.hp_dir = hp_dir

        self.type = 'smbo'
        if 'title' in run_config:
            self.id = run_config['title']

        self.device_id = 'hpo_cpu0'
        self.machine = None
        self.samples = None

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

    def get_sampling_space(self):
        if self.samples == None:
            warn("Sampling space is not initialized.")
        return self.samples
        
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
            if self.machine != None and self.machine.get_working_result() != None:
                latest = self.machine.get_current_results()
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
            warn("Exception occurs: {}".format(sys.exc_info()[0]))            
            self.stop_flag = True

        finally:
            with self.thread_cond:
                self.busy = False
                self.params = None
                self.thread_cond.notify()

    def stop(self):
        if self.machine != None:
            self.machine.force_stop()

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
        self.samples = None
        
        if 'surrogate' in args and args['surrogate'] != 'None':
            s_name = args['surrogate']
            hp_path = "{}{}.json".format(self.hp_dir, s_name)            
            hp_cfg = hconf.read_config(hp_path) # FIXME:rewrite here
            if hp_cfg == None:
                ValueError("Surrogate {} configuration not found.".format(s_name))
        
        if 'history_url' in args and valid.url(args['history_url']):            
            space_url = args['history_url']
            if not space_url.endswith('/'):
                space_url = space_url + "/"

            self.samples = remote.connect_remote_space(args['history_url'], 
                                                        run_cfg["credential"])
        else:
            warn("No valid history: {}".format(run_cfg['history_url']))
            self.samples = space.create_surrogate_space(args['surrogate'])

        if self.samples == None:
            raise ValueError("Invalid sampling space. Space is not initialized properly")

        if 'train_node' in args:
            if valid.url(args['train_node']):

                self.machine = bandit.create_runner(args['train_node'], self.samples,
                                                    args['exp_crt'], 
                                                    args['exp_goal'], args['exp_time'],
                                                    run_cfg, hp_cfg,                            
                                                    num_resume=num_resume,
                                                    save_internal=save_internal,
                                                    use_surrogate=s_name,
                                                    id=self.id)
                
            else:
                raise ValueError("Invalid node URL: {}".format(args["train_node"]))
        else:

            self.machine = bandit.create_emulator(self.samples,
                args['exp_crt'], args['exp_goal'], args['exp_time'],
                num_resume=num_resume,
                save_internal=save_internal, 
                run_config=run_cfg,
                id=self.id + "_emul")

        if args['mode'] == 'DIV' or args['mode'] == 'ADA':
            results = self.machine.mix(args['spec'], args['num_trials'], 
                save_results=save_results)
        elif args['mode'] in ALL_OPT_MODELS:
            results = self.machine.all_in(args['mode'], args['spec'], args['num_trials'], 
                save_results=save_results)
        else:
            raise ValueError('unsupported mode: {}'.format(args['mode']))
        
        return results