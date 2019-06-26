import os
import time

from ws.apis import *
from ws.shared.read_cfg import *
from ws.shared.logger import *

import argparse
import numpy as np

RUN_CONF_PATH = './run_conf/'
GPU_ID = 0

@eval_task
def optimize_mnist_lenet1(config, **kwargs):
    from samples.mnist_lenet1_keras import KerasWorker
    from samples.keras_cb import *

    max_epoch = 15
    if "max_epoch" in kwargs:
        max_epoch = kwargs["max_epoch"]    
    global GPU_ID

    history = TestAccuracyCallback()
    debug("Training configuration: {}".format(config))
    worker = KerasWorker(run_id='{}'.format(GPU_ID))
    res = worker.compute(config=config, budget=max_epoch, 
                        working_directory='./gpu{}/'.format(GPU_ID), history=history)
    return res


def main(run_config):    
    global GPU_ID
    try:
        hp_cfg = run_config["hp_config"]

        surrogate_func = eval(run_config["eval_func"])
        register_url = None
        if "register_url" in run_config:
            if run_config['register_url'] != "None":
                register_url = run_config['register_url']
        
        port = 6000
        if "port" in run_config:
            port = run_config["port"]
        
        resource_type = "cpu"
        if "resource_type" in run_config:
            resource_type = run_config["resource_type"]  
        if resource_type == "gpu":
            GPU_ID = run_config["resource_id"]
            # Set using single GPU only
            os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

        set_log_level('debug')
        print_trace()

        hp_cfg_path = './hp_conf/{}.json'.format(hp_cfg)
        hp_cfg = read_hyperparam_config(hp_cfg_path)
        wait_train_request(surrogate_func, hp_cfg, True,
                        device_type="gpu",
                        device_index=GPU_ID,
                        register_url=register_url, 
                        port=port)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--rconf_dir', default=RUN_CONF_PATH, type=str,
                        help='Run configuration directory.\n'+\
                        'Default setting is {}'.format(RUN_CONF_PATH))  
    parser.add_argument('run_config', type=str, help='run configuration name.') 
    args = parser.parse_args()
    run_cfg = read_run_config(args.run_config, args.rconf_dir)       
    
    main(run_cfg)
    