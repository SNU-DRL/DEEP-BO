from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import copy

import numpy as np
import math

from ws.shared.logger import *
from ws.shared.proto import TrainerPrototype

class RemoteTrainer(TrainerPrototype):
    def __init__(self, connector, space, **kwargs):
        self.space = space
        self.hp_config = connector.hp_config

        self.controller = connector

        self.jobs = {}
        self.history = []

        #debug("Run configuration: {}".format(kwargs))
        
        if "base_error" in kwargs:
            self.base_error = float(kwargs["base_error"])
        else:
            self.base_error = 0.9
        if "polling_interval" in kwargs:
            self.polling_interval = int(kwargs["polling_interval"])
        else:
            self.polling_interval = 5
        
        if "max_timeout" in kwargs:
            self.max_timeout = int(kwargs["max_timeout"])
        else:
            self.max_timeout = int((60 * 5) / self.polling_interval) # Set default timeout is 5 min.

        if "min_train_epoch" in kwargs:
            self.min_train_epoch = int(kwargs["min_train_epoch"])
        else:
            self.min_train_epoch = 1
        
        self.max_train_epoch = None
        
        if hasattr(self.hp_config, "config") and hasattr(self.hp_config.config, "max_epoch"):
            self.max_train_epoch = self.hp_config.config.max_epoch

        super(RemoteTrainer, self).__init__()
    
    def reset(self):
        self.jobs = {}
        self.history = []

    def check_termination_condition(self, acc_curve, estimates):
        # No termination check
        return False

    def wait_until_done(self, job_id, model_index, estimates, space):
        acc_curve = [] #XXX: we use accuracy curve instead of loss curve 
        prev_interim_err = None
        time_out_count = 0
        early_terminated = False
        while True: # XXX: infinite loop
            try:
                j = self.controller.get_job("active")
                if j != None:
                    if "lr" in j and len(j["lr"]) > 0:
                        if "cur_loss" in j and j['cur_loss'] != None:
                            acc_curve = []
                            for loss in j["lr"]:
                                if loss != None:
                                    acc = 1.0 - loss # FIXME: handle if loss is not error rate
                                else:
                                    acc = 0.0
                                acc_curve.append(acc) 
                        
                        # Interim error update
                        interim_err = j["lr"][-1]
                        if prev_interim_err == None or prev_interim_err != interim_err:
                            #debug("Interim error {} will be updated".format(interim_err))
                            if space != None:
                                space.update_error(model_index, interim_err, True)
                        
                        if prev_interim_err != interim_err:
                            # XXX:reset time out count
                            time_out_count = 0 
                        else:
                            time_out_count += 1
                            if time_out_count > self.max_timeout:
                                log("Force to stop {} due to no update for {} sec".format(job_id, self.polling_interval * self.max_timeout))
                                self.controller.stop(job_id)
                                break
                        prev_interim_err = interim_err
                        
                        # Early termination check
                        if self.min_train_epoch < len(acc_curve) and \
                            self.check_termination_condition(acc_curve, estimates):                        
                            job_id = j['job_id']
                            #debug("This job will be terminated")
                            self.controller.stop(job_id)
                            early_terminated = True
                            break

                    elif "lr" in j and len(j["lr"]) == 0:
                        pass
                    else:
                        warn("Invalid job result: {}".format(j))
                elif j == None:
                    # cross check 
                    r = self.controller.get_job(job_id)
                    if "lr" in r:
                        num_losses = len(r["lr"])
                        if num_losses > 0:
                            debug("Current job finished with loss curve: {}.".format(r["lr"]))
                            break
                        else:
                            debug("Result of finished job: {}".format(r)) 
                
                #debug("Waiting {} sec...".format(self.polling_interval)) 
                time.sleep(self.polling_interval)
            
            except Exception as ex:
                if str(ex) == 'timed out':
                    time_out_count += 1                    
                    if time_out_count < self.max_timeout:
                        debug("Timeout occurred. Retry {}/{}...".format(time_out_count, self.max_timeout))
                        continue

                warn("Something goes wrong in remote worker: {}".format(ex))
                early_terminated = True
                break
        
        return early_terminated

    def train(self, cand_index, estimates=None, space=None):
        hpv = {}
        cfg = {'cand_index' : cand_index}
        param_names = self.hp_config.get_hyperparams()
        param_values = self.space.get_hpv(cand_index)
        if type(param_values) == np.ndarray:
            param_values = param_values.tolist()
        elif type(param_values) == dict:
            values_only = []
            for i in range(len(param_names)):
                p = param_names[i]
                v = param_values[p]
                values_only.append(v)
            param_values = values_only
        
        early_terminated = False
        log("Training model using hyperparameters: {}".format(param_values))
        
        if type(param_values) == dict:
            for param in param_names:
                value = param_values[param]
                if self.hp_config.get_type(param) == 'bool':
                    value = bool(value)
                elif self.hp_config.get_type(param) != 'str':
                    value = float(value)
                    if self.hp_config.get_type(param) == 'int':
                        value = int(value)
                hpv[param] = value
        elif type(param_values) == list and len(param_names) == len(param_values):
            for i in range(len(param_names)):
                param = param_names[i]
                value = param_values[i]
                if self.hp_config.get_type(param) != 'str':
                    value = float(value)
                    if self.hp_config.get_type(param) == 'int':
                        value = int(value)
                hpv[param] = value
        else:
            raise TypeError("Invalid hyperparams: {}/{}".format(param_names, param_values))

        if self.controller.validate():
            job_id = self.controller.create_job(hpv, cfg)
            if job_id is not None:                
                if self.controller.start(job_id):
                    
                    self.jobs[job_id] = {"cand_index" : cand_index, "status" : "run"}
                    
                    early_terminated = self.wait_until_done(job_id, cand_index, estimates, space)

                    result = self.controller.get_job(job_id)
                   
                    self.jobs[job_id]["result"] = result
                    self.jobs[job_id]["status"] = "done"
                    
                    loss_curve = result["lr"]
                    test_err = result['cur_loss']
                    best_epoch = len(loss_curve) + 1
                    train_epoch = len(loss_curve)
                    
                    # XXX: exceptional case handling - when timeout occurs, acc_curve increased largely.
                    if self.max_train_epoch != None and train_epoch > self.max_train_epoch:
                        train_epoch = self.max_train_epoch
                    
                    if loss_curve != None and len(loss_curve) > 0:
                        best_i = self.get_min_loss_index(loss_curve)
                        test_err = loss_curve[best_i]
                        best_epoch = best_i + 1

                        self.add_train_history(loss_curve, 
                                               result['run_time'], 
                                               train_epoch,
                                               measure='loss')
                                            
                    return {
                            "test_error": test_err,
                            "train_epoch": train_epoch,
                            "best_epoch" : best_epoch, 
                            "train_time" : result['run_time'], 
                            'early_terminated' : early_terminated
                    }  
                else:
                    error("Starting training job failed.")
            else:
                error("Creating job failed")
            
        else:
            error("Connection error: handshaking with trainer failed.")
        
        raise ValueError("Remote training failed")       

    def get_min_loss_index(self, loss_curve):
        best_i = 0
        best_loss = None
        for i in range(len(loss_curve)):
            loss = loss_curve[i]
            if best_loss == None:
                best_loss = loss
            if loss != None and best_loss > loss:
                best_loss = loss
                best_i = i
        return best_i



    def find_job_id(self, cand_index):
        for j in self.jobs.keys():
            if cand_index == self.jobs[j]['cand_index']:
                return j
        else:
            return None

    def get_interim_error(self, model_index, cur_dur):
        job_id = self.find_job_id(model_index)
        if job_id is not None:
            if self.controller.get_job("active") != None:
                result = self.controller.get_job(job_id)
                self.jobs[job_id]["result"] = result

                return result['cur_loss']
            else:
                debug("This job {} may be already finished.".format(job_id))
                return self.jobs[job_id]["result"]['cur_loss']
        
        return self.base_error


class EarlyTerminateTrainer(RemoteTrainer):
    
    def __init__(self, controller, hpvs, **kwargs):
        self.early_terminated_history = []
        super(EarlyTerminateTrainer, self).__init__(controller, hpvs, **kwargs)

        self.history = []
        self.early_terminated_history = []
        self.etr_checked = False

    def reset(self):
        # reset history
        self.history = []
        self.early_terminated_history = []

    def train(self, cand_index, estimates=None, space=None):
        self.etr_checked = False
        early_terminated = False
        train_result = super(EarlyTerminateTrainer, self).train(cand_index, estimates, space)
        if 'early_terminated' in train_result:
            early_terminated = train_result['early_terminated']
        self.early_terminated_history.append(early_terminated)

        return train_result  