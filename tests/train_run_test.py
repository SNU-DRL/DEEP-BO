import os
import sys

# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import ws.hpo.bandit as bandit
from ws.shared.read_cfg import *
import ws.hpo.space_mgr as space

from ws.shared.logger import *


def test_run_main(surrogate):

    set_log_level('debug')
    print_trace()

    # XXX: prerequisite: training worker service should be executed before running.
    trainer_url = 'http://127.0.0.1:6001'

    hp_cfg_path = './hp_conf/{}.json'.format(surrogate)
    hp_cfg = read_hyperparam_config(hp_cfg_path)
    
    if hp_cfg is None:
        print("Invalid hyperparameter configuration file: {}".format(hp_cfg_path))
        return  

    run_cfg = read_run_config('p6div-etr.json') 
    #run_cfg = rconf.read('arms.json') 
        
    samples = space.create_grid_space(hp_cfg.get_dict())
    runner = bandit.create_runner(trainer_url, samples,
                                'TIME', 0.999, "1h",
                                #use_surrogate=surrogate, 
                                run_cfg, hp_cfg
                                )

    runner.mix('SEQ', 20, save_results=True) # BO-HEDGE
    #runner.all_in('TPE','EI', 1, save_results=True) 
    runner.temp_saver.remove()    


if __name__ == '__main__':
    test_run_main("kin8nm-MLP")
