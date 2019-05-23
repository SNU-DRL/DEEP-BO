import os
import sys
import random as rand

# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import ws.shared.hp_cfg as hconf
from ws.apis import wait_train_request, eval_task
from samples.keras_cb import *


@eval_task
def optimize_kin8nm_mlp(config, **kwargs):
    from samples.kin8nm_mlp_keras import KerasWorker
    max_epoch = 27
    if "max_epoch" in kwargs:
        max_epoch = kwargs["max_epoch"]

    global GPU_ID

    history = RMSELossCallback()
    debug("Training configuration: {}".format(config))
    worker = KerasWorker(run_id='{}'.format(GPU_ID))
    res = worker.compute(config=convert_config(config), budget=max_epoch, 
                        working_directory='./gpu{}/'.format(GPU_ID), history=history)
    return res


def convert_config(config):
    config['shuffle'] = bool(config['shuffle'])
    for n in range(1, config['n_layers'] + 1):        
        reg = config['layer_{}_reg'.format(n)]
        extras = {'name' : reg }
        if reg == 'dropout':
            extras['rate'] = config['dropout_rate_{}'.format(n)]
        config['layer_{}_extras'.format(n)] = extras

    return config


def main(bm_cfg, surrogate_func, gpu_id):
    global GPU_ID
    GPU_ID = gpu_id
    # Set using single GPU only
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    set_log_level('debug')
    print_trace()

    hp_cfg_path = './hp_conf/{}.json'.format(bm_cfg)
    hp_cfg = hconf.read_config(hp_cfg_path)
    wait_train_request(surrogate_func, hp_cfg, True,
                    device_type="gpu",
                    device_index=gpu_id, 
                    port=6000 + gpu_id, processed=True)



if __name__ == "__main__":
    
    main('kin8nm_mlp', optimize_kin8nm_mlp, args.gpu_id)
    