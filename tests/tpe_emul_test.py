import os
import sys

# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import ws.hpo.bandit as bandit
import ws.hpo.space_mgr as space
from ws.shared.read_cfg import *

from ws.shared.logger import *

def test_emul_main():

    conf = read_run_config('7div-etr.json')
    samples = space.create_surrogate_space('MNIST-LeNet1')
    lookup = samples.lookup
    emul = bandit.create_emulator(samples, lookup, 
                'TIME', 0.999, '6h', 
                run_config=conf)
    #emul.with_pkl = True
    set_log_level('debug')
    print_trace()

    emul.all_in('TPE', 'RANDOM', 1, save_results=False)

#    emul.mix('SEQ', 1, save_results=False)

    emul.temp_saver.remove()


if __name__ == '__main__':
    test_emul_main()