import os
import sys
from collections import Counter
# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ws.apis import *
from ws.shared.logger import *
import ws.hpo.bandit_config as bconf
import ws.shared.hp_cfg as hconf

import ws.hpo.bandit as bandit
import ws.hpo.space_mgr as space


def mnist_lenet2_test(etr):
    hp_cfg = hconf.read_config("hp_conf/MNIST-LeNet2.json")
    samples = space.create_surrogate_space('MNIST-LeNet2')
    
    set_log_level('debug')
    
    run_cfg = bconf.read("arms.json", path="run_conf/")
    run_cfg["early_term_rule"] = etr
    m = bandit.create_emulator(samples, 
                'TIME', 0.9999, '1d',
                run_config=run_cfg)
    results = m.mix('SEQ', 1, save_results=False)
    for i in range(len(results)):
        result = results[i]
        log("At trial {}, {} iterations by {}".format(i, len(result["select_trace"]), Counter(result["select_trace"])))


if __name__ == "__main__":
    early_term_test = mnist_lenet2_test
    #early_term_test("None")
    #early_term_test("Donham15")
    early_term_test("Donham15Fantasy")

    

