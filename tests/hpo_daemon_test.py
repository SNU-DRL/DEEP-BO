import os
import sys

# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ws.apis import *
import ws.hpo.bandit_config as bconf
import ws.shared.hp_cfg as hconf

def main():
    run_cfg = bconf.read('parallel-test.json')
    hp_cfg = hconf.read('./hp_conf/data1.json')
    wait_hpo_request(run_cfg, hp_cfg, True)

if __name__ == "__main__":
    main()
    