import os
import time

from ws.apis import *
from ws.shared.read_cfg import *
from ws.shared.logger import *

import argparse
import validators as valid

from samples.keras_cb import *

RESOURCE_ID = 'cpu0'

@objective_function
def optimize_mnist_lenet1(config, **kwargs):
    from samples.mnist_lenet1_keras import KerasWorker
    global RESOURCE_ID
    
    start_time = time.time()
    max_epoch = 15
    
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]    

    history = TestAccuracyCallback()
    log("Training configuration: {}".format(config))
    worker = KerasWorker(run_id='{}'.format(RESOURCE_ID))
    res = worker.compute(config=config, 
                         budget=max_epoch, 
                         working_directory='./{}/'.format(RESOURCE_ID), 
                         history=history)
    elapsed_time = time.time() - start_time
    report_result(res, elapsed_time)


@objective_function
def optimize_kin8nm_mlp(config, **kwargs):
    from samples.kin8nm_mlp_keras import KerasWorker
    global RESOURCE_ID

    start_time = time.time()
    max_epoch = 27
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]    

    history = RMSELossCallback()
    debug("Training configuration: {}".format(config))
    worker = KerasWorker(run_id='{}'.format(RESOURCE_ID))
    res = worker.compute(config=convert_config(config), 
                         budget=max_epoch, 
                         working_directory='./{}/'.format(RESOURCE_ID), 
                         history=history)
    elapsed_time = time.time() - start_time
    report_result(res, elapsed_time)


def report_result(res, elapsed_time):
    # update final result
    try:
        update_current_loss(res['cur_iter'], 
                            res['cur_loss'], 
                            elapsed_time, 
                            iter_unit=res['iter_unit'],
                            loss_type=res['loss_type'])
        if 'info' in res:
            log("Training finished :{}".format(res['info']))
    except Exception as ex:
        warn("Final result updated failed: {}".format(ex))


def convert_config(config):
    config['shuffle'] = bool(config['shuffle'])
    for n in range(1, config['n_layers'] + 1):        
        reg = config['layer_{}_reg'.format(n)]
        extras = {'name' : reg }
        if reg == 'dropout':
            extras['rate'] = config['dropout_rate_{}'.format(n)]
        config['layer_{}_extras'.format(n)] = extras

    return config


def main(run_config):    
    global RESOURCE_ID

    try:
        master_node = None
        if "master_node" in run_config:
            if valid.url(run_config['master_node']):
                master_node = run_config['master_node']
                if master_node.endswith('/'):
                    master_node += master_node[:-1]
            else:
                raise ValueError("Invalid master URL: {}".format(run_config['master_node']))
        debug_mode = False
        if "debug_mode" in run_config:
            if run_config["debug_mode"]:
                debug_mode = True
                set_log_level('debug')
                print_trace()

        hp_config_dir = "./hp_conf/"
        if "hp_config_dir" in run_config:
            hp_config_dir = run_config["hp_config_dir"]         
       
        hp_cfg_file = run_config["hp_config"]
        hp_cfg_path = '{}{}.json'.format(hp_config_dir, hp_cfg_file)
        hp_cfg = read_hyperparam_config(hp_cfg_path)

        port = 6000
        if "port" in run_config:
            port = run_config["port"]
        
        resource_type = "cpu"
        if "resource_type" in run_config:
            resource_type = run_config["resource_type"]  
        if resource_type == "gpu":            
            # Set using single GPU only
            os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = str(run_config["resource_id"])
            RESOURCE_ID = "{}{}".format(resource_type, run_config["resource_id"])

        credential = None
        if "credential" in run_config:
            credential = run_config['credential']
        else:
            raise ValueError("No credential info in run configuration")

        eval_func = eval(run_config["eval_func"])
        log("Training DNN via {}...".format(run_config["eval_func"]))

        wait_train_request(eval_func, 
                           hp_cfg, 
                           debug_mode=debug_mode,
                           device_type=resource_type,
                           device_index=run_config["resource_id"],
                           master_node=master_node,
                           credential=credential, 
                           port=port)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    run_conf_path = './run_conf/'
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd', '--rconf_dir', default=run_conf_path, type=str,
                        help='Run configuration directory.\n'+\
                        'Default setting is {}'.format(run_conf_path))  
    parser.add_argument('run_config', type=str, help='run configuration name.') 
    args = parser.parse_args()
    run_cfg = read_run_config(args.run_config, args.rconf_dir)       
    
    main(run_cfg)
    