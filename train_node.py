import os
import time

from ws.apis import *
import ws.shared.hp_cfg as hconf
from ws.shared.logger import *

import keras
from samples.keras_cb import *
import argparse
import numpy as np

GPU_ID = 0


@eval_task
def optimize_mnist_lenet1(config, **kwargs):
    from samples.mnist_lenet1_keras import KerasWorker
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
    main('MNIST-LeNet1', optimize_mnist_lenet1, args.gpu_id)
    