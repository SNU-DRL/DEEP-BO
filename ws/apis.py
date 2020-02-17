import traceback
import atexit
import inspect

import validators as valid

from ws.shared.logger import *
from ws.ws_mgr import WebServiceManager
from ws.shared.register import MasterServerConnector


JOB_MANAGER = None
API_SERVER = None


####################################################################
# Job request awaiting APIs
def create_master_server(hp_cfg,
                         debug_mode=False,
                         credential=None,
                         port=5000,
                         threaded=False):

    '''Spawn a master daemon which serves parameter space and parallel Bayesian Optimization run.  

    Arguments:
        hp_cfg {HyperparameterConfiguration} -- hyperparameter configuration which will be used to create parameter space.

    Keyword arguments:
        debug_mode {bool} -- show debug message or not (default False)
        credential {str} -- credential key for authentication (default None)
        port {int} -- the port number that is opened for response (default 5000)
        threaded {bool} -- enable multi-threading (default False)

    This API blocks the remained procedure unless a terminal signal enters.
 
    '''
    from hpo.batch_mgr import ParallelHPOManager

    global JOB_MANAGER
    global API_SERVER
    JOB_MANAGER = ParallelHPOManager(hp_cfg)
    API_SERVER = WebServiceManager(JOB_MANAGER, hp_cfg, credential=credential)
    API_SERVER.run_service(port, debug_mode, threaded)


def wait_hpo_request(run_cfg, 
                     hp_cfg,
                     debug_mode=False,
                     port=6000,
                     enable_surrogate=False,
                     credential=None,
                     master_node=None,
                     threaded=False):
    '''Spawn a worker daemon which serves Bayesian Optimization.  

    Arguments:
        run_cfg {dictionary} -- run configuration.
        hp_cfg {HyperparameterConfiguration} -- hyperparameter configuration which will be used to select hyperparameter vector.

    Keyword arguments:
        debug_mode {bool} -- show debug message or not (default False)
        port {int} -- the port number that is opened for response (default 6000)
        enable_surrogate {bool} -- whether the use of pre-evaluated lookup table or not (default False)
        credential {str} -- credential key for authentication (default None)
        master_node {str} -- the URL to register myself to master node. if None, no register (default None)
        threaded {bool} -- enable multi-threading (default False)

    This API blocks the remained procedure unless a terminal signal enters.
 
    '''
    from hpo.job_mgr import HPOJobManager

    global JOB_MANAGER
    global API_SERVER

    if JOB_MANAGER == None:
        JOB_MANAGER = HPOJobManager(
            run_cfg, hp_cfg, port, use_surrogate=enable_surrogate)
        if master_node != None and valid.url(master_node):
            try:
                ns = MasterServerConnector(
                    master_node, credential)
                ns.register(port, "BO Node")
            except Exception as ex:
                warn("Registering to master server failed: {}".format(ex))

        API_SERVER = WebServiceManager(JOB_MANAGER, hp_cfg, credential=credential)
        API_SERVER.run_service(port, debug_mode, threaded)
    else:
        warn("Job manager already initialized.")
        return


def wait_train_request(train_task, 
                       hp_cfg,
                       debug_mode=False,
                       port=6100,
                       device_type="cpu",
                       device_index=0,
                       retrieve_func=None,
                       enable_surrogate=False,
                       master_node=None,
                       credential=None,
                       processed=True
                       ):
    '''Spawn a worker daemon which serves DNN training.  

    Arguments:
        train_task {function} -- a target function which is decorated with @eval_task.
        hp_cfg {HyperparameterConfiguration}-- hyperparameter configuration which will be used to validate hyperparameter vector.

    Keyword arguments:
        debug_mode {bool} -- show debug message or not (default False)
        port {int} -- the port number that is opened for response (default 6100)
        device_type {str} -- type of processing unit. 'cpu' or 'gpu' (default 'cpu')
        device_index {int} -- index of processing unit (default 0)
        enable_surrogate {bool} -- whether the use of pre-evaluated lookup table or not (default False)
        master_node {str} -- the URL to register myself to master node. if None, no register (default None)
        credential {str} -- credential key for authentication (default None)
        processed {bool} -- enable spawning a process for training (default True)

    This API blocks the remained procedure unless a terminal signal enters.
 
    '''
    from ws.wot.job_mgr import TrainingJobManager

    global JOB_MANAGER
    global API_SERVER

    if JOB_MANAGER == None:
        task = train_task()
        task.set_resource(device_type, device_index)
        JOB_MANAGER = TrainingJobManager(task,
                                         use_surrogate=enable_surrogate,
                                         retrieve_func=retrieve_func)
        if master_node != None and valid.url(master_node):
            try:
                ns = MasterServerConnector(
                    master_node, credential)
                ns.register(port, "Training Node")
            except Exception as ex:
                warn("Registering myself to name server failed: {}".format(ex))

        API_SERVER = WebServiceManager(JOB_MANAGER, hp_cfg, credential=credential)
        API_SERVER.run_service(port, debug_mode, with_process=processed)
    else:
        warn("Job manager already initialized.")
        return


def stop_job_working():
    ''' Send stop request for current job '''
    
    global JOB_MANAGER

    if JOB_MANAGER != None:
        JOB_MANAGER.stop_working_job()
    else:
        warn("Job manager is not ready to serve.")


def update_current_loss(cur_iters, 
                        cur_loss, 
                        run_time,
                        iter_unit='epoch',
                        loss_type='error rate'):
    ''' Report the current loss of training task
    
    Arguments:
        cur_iters {int} -- current iteration number 
        cur_loss {float} -- current loss value
        run_time  {float} -- elapsed time (seconds) of the given task

    Keyword arguments:
        iter_unit {str} -- the iteration unit. "step" or "epoch" is valid (default "epoch")
        loss_type {str} -- the type of loss. if 'error rate', accuracy can be introduced by simple transformation. (default 'error rate')
    
    '''    
    global JOB_MANAGER

    if JOB_MANAGER != None:
        JOB_MANAGER.update_result(cur_iters, cur_loss, run_time,
                                  iter_unit=iter_unit,
                                  loss_type=loss_type)
    else:
        warn("Job manager is not ready to serve.")


#########################################################################
# Decorator functions
# (Do NOT invoke it directly)
def objective_function(eval_func):
    def wrapper_function():
        from ws.wot.workers.evaluator import TargetFunctionEvaluator

        argspec = inspect.getargspec(eval_func)
        fe = TargetFunctionEvaluator(
            "{}".format(eval_func.__name__))
        defaults = None
        try:
            defaults = getattr(argspec, 'defaults')
            debug("Target function arguments: {}, defaults: {}".format(argspec.args, defaults))
        except Exception as ex:            
            pass
        fe.set_exec_func(eval_func, argspec.args, defaults)
        return fe

    return wrapper_function


def progressive_objective_function(eval_func):
    def wrapper_function():
        from ws.wot.workers.evaluator import TargetFunctionEvaluator

        argspec = inspect.getargspec(eval_func)
        fe = TargetFunctionEvaluator(
            "{}".format(eval_func.__name__), progressive=True)
        fe.set_exec_func(eval_func, argspec.args)
        return fe

    return wrapper_function


@atexit.register
def exit():
    global JOB_MANAGER
    global API_SERVER
    if JOB_MANAGER != None:
        JOB_MANAGER.__del__()
        JOB_MANAGER = None

    if API_SERVER != None:

        API_SERVER.stop_service()
        debug("API server terminated properly.")

        API_SERVER = None
