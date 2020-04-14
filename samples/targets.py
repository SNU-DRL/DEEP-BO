import os
import time

from ws.apis import *
from ws.shared.read_cfg import *
from ws.shared.logger import *

from ws.shared.worker import WorkerResource
from .configs.utils import *

RESOURCE = WorkerResource() # allocated computing resource identifier
START_TIME = None

def get_resource():
    global RESOURCE
    return RESOURCE


''' Simple classification problem '''
@objective_function
def tune_mnist_lenet5(config, fail_err=0.9, **kwargs):
    from .keras.lenet5_clr import KerasClassificationWorker
    from .keras.callbacks import TestAccuracyCallback
    from .keras.datasets import load_data

    start_time = time.time()
    max_epoch = 15
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]    

    acc_cb = TestAccuracyCallback()
    log("Training with {}".format(config))
    dataset = load_data('MNIST')
    worker = KerasClassificationWorker(dataset, run_id='{}'.format(get_resource().get_id()))
    res = worker.compute(config=config, 
                         budget=max_epoch, 
                         working_directory='./{}/'.format(get_resource().get_id()), 
                         epoch_cb=acc_cb)
    elapsed_time = time.time() - start_time
    report_result(res, elapsed_time)


''' Simple regression problem '''
@objective_function
def tune_kin8nm_mlp(config, **kwargs):
    from samples.keras.mlp_regr import KerasRegressionWorker
    from .keras.callbacks import RMSELossCallback
    from .keras.datasets import load_data

    start_time = time.time()
    max_epoch = 27
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]    

    rmse_cb = RMSELossCallback()
    debug("Training with {}".format(config))
    dataset = load_data('kin8nm')
    worker = KerasRegressionWorker(dataset, run_id='{}'.format(get_resource().get_id()))
    res = worker.compute(config=convert_config(config), 
                         budget=max_epoch, 
                         working_directory='./{}/'.format(get_resource().get_id()), 
                         epoch_cb=rmse_cb)
    elapsed_time = time.time() - start_time
    report_result(res, elapsed_time)


@objective_function
def tune_protein_mlp(config, **kwargs):
    from samples.keras.mlp_regr import KerasRegressionWorker
    from .keras.callbacks import RMSELossCallback
    from .keras.datasets import load_data

    start_time = time.time()
    max_epoch = 50
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]    

    rmse_cb = RMSELossCallback()
    debug("Training with {}".format(config))
    dataset = load_data('protein')
    worker = KerasRegressionWorker(dataset, loss_type='RMSE', run_id='{}'.format(get_resource().get_id()))
    res = worker.compute(config=convert_config(config), 
                         budget=max_epoch, 
                         working_directory='./{}/'.format(get_resource().get_id()), 
                         epoch_cb=rmse_cb)
    elapsed_time = time.time() - start_time
    report_result(res, elapsed_time)


''' Surrogates regression problem - internal use only '''
@objective_function
def tune_surrogates_mlp(config, **kwargs):
    from samples.keras.mlp_regr import KerasRegressionWorker
    from .keras.callbacks import RMSELossCallback
    from .keras.datasets import load_data

    start_time = time.time()
    max_epoch = 100
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]    

    rmse_cb = RMSELossCallback()

    debug("Training with {}".format(config))
    dataset = load_data('MNIST-LeNet1')
    worker = KerasRegressionWorker(dataset, run_id='{}'.format(get_resource().get_id()))
    res = worker.compute(config=convert_config(config), 
                         budget=max_epoch, 
                         working_directory='./{}/'.format(get_resource().get_id()), 
                         epoch_cb=rmse_cb)
    elapsed_time = time.time() - start_time
    report_result(res, elapsed_time)


def report_result(res, elapsed_time):
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


def update_epoch_acc(num_epoch, valid_acc):
    global START_TIME
    if valid_acc > 1.0:
        valid_acc = float(valid_acc / 100.0)
    debug("validation accuracy at {}: {}".format(num_epoch, valid_acc))
    valid_error = 1.0 - float(valid_acc)
    elapsed_time = time.time() - START_TIME
    update_current_loss(num_epoch, valid_error, elapsed_time)


''' SOTA classification problems '''
@objective_function
def tune_efficientnet_cifar10(config, fail_err=0.9, **kwargs):
    from .pytorch.trainer import train
    
    global START_TIME
    START_TIME = time.time()
    run_id = get_resource().get_id()

    max_epoch = 90
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]   

    cfg_path = create_yaml_config(run_id, 'cifar10', config, max_epoch)

    train(cfg_path, epoch_cb=update_epoch_acc)


@objective_function
def tune_efficientnet_cifar100(config, fail_err=0.99, **kwargs):
    from .pytorch.trainer import train

    global START_TIME
    START_TIME = time.time()
    run_id = get_resource().get_id()

    max_epoch = 90
    if "max_iters" in kwargs:
        if "iter_unit" in kwargs and kwargs["iter_unit"] == "epoch":
            max_epoch = kwargs["max_iters"]   

    cfg_path = create_yaml_config(run_id, 'cifar100', config, max_epoch)

    train(cfg_path, epoch_cb=update_epoch_acc)
