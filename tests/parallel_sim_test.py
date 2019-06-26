import os
import sys
import numpy as np
from collections import Counter
# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ws.apis import *
from ws.shared.logger import *
from ws.shared.read_cfg import *

import ws.hpo.batch_sim as batch
import ws.hpo.space_mgr as space

def parallel_etr_test(etr):
    hp_cfg = read_hyperparam_config("hp_conf/CIFAR10-ResNet.json")
    samples = space.create_surrogate_space('CIFAR10-ResNet')
    
    set_log_level('debug')
    
    run_cfg = read_run_config("p6div-no_log-etr-nc.json", path="run_conf/")
    #run_cfg["early_term_rule"] = etr
    c = batch.get_simulator("ASYNC", "CIFAR10-ResNet",
                        "GOAL", 0.9318, 
                        "30h", run_cfg)
    results = c.run(1, save_results=False)
    for i in results.keys():
        result = results[i]
        traces = result["select_trace"]        
        log("At trial {}, traces: {}".format(i, traces))
        

if __name__ == "__main__":
    parallel_etr_test("TetraTercet")
    

