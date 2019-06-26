import os
import sys

# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ws.apis import *
from ws.shared.read_cfg import *

def main():
    run_cfg = read_run_config('parallel-test.json')
    hp_cfg = read_hyperparam_config('./hp_conf/data1.json')
    wait_hpo_request(run_cfg, hp_cfg, True)

if __name__ == "__main__":
    main()
    