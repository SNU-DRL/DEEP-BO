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
        self.response_time = time.time()

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
            self.max_timeout = 100000 
        
        if "min_train_epoch" in kwargs:
            self.min_train_epoch = int(kwargs["min_train_epoch"])
        else:
            self.min_train_epoch = 1
        
        self.max_train_epoch = None
        
        if hasattr(self.hp_config, "config") and hasattr(self.hp_config.config, "max_epoch"):
            self.max_train_epoch = self.hp_config.config.max_epoch

        debug("Timeout: {}, Interval: {}".format(self.max_timeout, self.polling_interval))

        super(RemoteTrainer, self).__init__()
    
    def reset(self):
        self.jobs = {}
        self.history = []

    def stop_check(self, acc_curve, estimates):
        # No termination at all
        return False

    def get_acc_curve(self, loss_curve):
        acc_curve = []
        prev_acc = None
        for loss in loss_curve:
            if loss != None:
                acc = 1.0 - loss # FIXME: consider when loss is not error rate
                prev_acc = acc
            else:
                if prev_acc == None:
                    acc = 0.0
                else:
                    acc = prev_acc
            acc_curve.append(acc)
        return acc_curve 

    def wait_until_done(self, job_id, sample_index, estimates, space):

        prev_epoch = None
        time_out_count = 0
        early_terminated = False
        
        while True: # XXX: infinite loop
            try:
                j = self.controller.get_job(job_id) # XXX: when job finished, it becomes None
                if j != None and j['status'] == 'processing':                    
                    interim_err = j['cur_loss']
                    cur_epoch = j['cur_iter']
                        
                    if prev_epoch != cur_epoch:
                        space.update_error(sample_index, interim_err, cur_epoch)
                        debug("Interim error {} updated at {} epoch".format(interim_err, cur_epoch))                            
                        
                        time_out_count = 0 # XXX:reset time out count
                        self.response_time = time.time()
                        if self.polling_interval > 1:
                            self.polling_interval -= 1                                
                    else:
                        time_out_count += 1
                        
                        if self.polling_interval < 10:
                            self.polling_interval += 1                            
                        
                        no_response = time.time() - self.response_time
                        if time_out_count > self.max_timeout:
                            log("Force to stop {} due to no update for {} sec".format(job_id, no_response))
                            self.controller.stop(job_id)
                            break
                        elif time_out_count % 10 == 0:
                            debug("No responses during {:.0f}... Timeout: {}/{}".format(no_response, 
                                                                                time_out_count, 
                                                                                self.max_timeout))
                
                    prev_epoch = cur_epoch
                    
                    # Early termination check
                    if "lr" in j and len(j['lr']) > 0:
                        acc_curve = self.get_acc_curve(j["lr"])
                        if self.min_train_epoch < len(acc_curve):
                            if self.stop_check(acc_curve, estimates):                        
                                self.controller.stop(job_id)
                                early_terminated = True
                                break
                elif j != None:
                    break
                else:
                    warn("Job monitoring failed: {}".format(job_id))
                
                #debug("Waiting {} sec...".format(self.polling_interval)) 
                time.sleep(self.polling_interval)
            
            except Exception as ex:
                if str(ex) == 'timed out':
                    time_out_count += 1                    
                    if time_out_count < self.max_timeout:
                        debug("Timeout! Retry {}/{}...".format(time_out_count, self.max_timeout))
                        time.sleep(3)
                        continue

                warn("Something goes wrong in remote worker: {}".format(ex))
                return True

        # final status check
        while True: # XXX: infinite loop
            j = self.controller.get_job(job_id)
            if j['status'] == 'processing':
                debug("Waiting until job being terminated...")
                time.sleep(self.polling_interval)
            else:
                min_loss = j["cur_loss"]
                cur_iter = j['cur_iter']
                if "lr" in j:
                    min_index = self.get_min_loss_index(j["lr"])
                    if min_index != None:
                        min_loss = j["lr"][min_index]
                        cur_iter = min_index # XXX: lr has 0 epoch performance

                # final best result report                
                space.update_error(sample_index, min_loss, cur_iter)
                debug("Training job finished! (id: {}, epoch:{}, loss: {})".format(sample_index, cur_iter, min_loss))
            break
        
        return early_terminated

    def train(self, cand_index, estimates=None, space=None):
        hpv = {}
        cfg = {'cand_index' : cand_index}
        param_names = self.hp_config.get_param_list()
        hpv_d = self.space.get_hpv_dict(cand_index)
          
        early_terminated = False
        
        # Type matching
        for param in param_names:
            value = hpv_d[param]
            if self.hp_config.get_type(param) == 'bool':
                value = bool(value)
            elif self.hp_config.get_type(param) != 'str':
                value = float(value)
                if self.hp_config.get_type(param) == 'int':
                    value = int(value)
            hpv[param] = value

        if self.controller.validate():
            job_id = self.controller.create_job(hpv, cfg)
            if job_id is not None:                
                if self.controller.start(job_id):
                    debug("Training starts using {}".format(hpv))

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
        best_i = None
        best_loss = None
        for i in range(len(loss_curve)):
            loss = loss_curve[i]
            if best_loss == None:
                best_loss = loss
                best_i = i
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

                return result['cur_loss'], result['cur_iter']
            else:
                debug("The {} job already finished.".format(job_id))
                result = self.jobs[job_id]["result"]
                return result['cur_loss'], result['cur_iter']
        
        return self.base_error, 0


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