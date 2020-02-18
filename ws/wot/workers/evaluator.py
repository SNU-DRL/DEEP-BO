import os
import signal
import sys
import time
import traceback
import copy
import math

import multiprocessing as mp
import numpy as np

from ws.shared.logger import *
from ws.wot.workers.trainer import Trainer


class TargetFunctionEvaluator(Trainer):
    def __init__(self, name, progressive=False, forked=True):
        
        super(TargetFunctionEvaluator, self).__init__(name, fork=forked)

        self.type = 'eval_func'
        self.eval_func = None
        
        self.max_iters = 1
        self.iter_unit = "epoch"
        self.progressive = progressive
        self.eval_process = None

    def get_config(self):
        return self.config

    def set_max_iters(self, num_max_iters, iter_unit="epoch"):
        self.max_iters = num_max_iters

    def set_exec_func(self, eval_func, args, defaults=None):
        self.eval_func = eval_func
        self.config = {"target_func": eval_func.__name__,
                        "arguments" : args,
                        "defaults" : defaults}

    def start(self):
        if self.is_working() == True:
            debug("Trainer is busy!")
            time.sleep(3) # XXX:set proper waiting time
            return False

        if self.params is None:
            error('Set configuration properly before starting!')
            return False
        else:
            debug("Training is starting with {}".format(self.params))
            super(TargetFunctionEvaluator, self).start()
            return True

    def is_working(self):
        try:
            if self.is_forked() and self.eval_process != None:
                self.busy = self.eval_process.is_alive()
                return self.busy
            else:
                self.busy = False
        except Exception as ex:
            debug("Exception raised when checking work state: {}".format(ex))
        finally:
            return self.busy
    def stop(self):
        if self.eval_process != None:
            try:
                while self.eval_process.is_alive():
                    os.kill(self.eval_process._popen.pid, signal.SIGKILL)
                    time.sleep(1)
                debug("Evaluation process stopped.")                
                self.stop_flag = True
            except Exception as ex:
                warn("Exception occurs on stopping an evaluation job: {}".format(ex))
                self.stop() # XXX:retry to kill process
        else:            
            super(TargetFunctionEvaluator, self).stop()
            while self.stop_flag == False:
                time.sleep(1)
    
    def execute(self):
        try:
            self.reset() # XXX:self.results should be empty here
            debug("Max iterations: {}{}".format(self.max_iters, self.iter_unit))
            
            max_iters = self.max_iters
            num_iters = 1
            
            if self.progressive == True:
                num_iters = max_iters                

            for i in range(num_iters):
                base_time = time.time()
                with self.pause_cond:
                    while self.paused:
                        self.pause_cond.wait()
                debug("Assigned params: {}".format(self.params))
                
                # check stop request before long time evaluation
                if self.stop_flag == True:
                    break

                if self.progressive == True:
                    max_iters = i + 1

                result = None
                job_id = copy.copy(self.job_id) # XXX:store started job id   

                if self.is_forked() == True:
                    self.eval_process = mp.Process(target=self.eval_func, 
                                                   args=(self.params,), 
                                                   kwargs={
                                                            "max_iters": max_iters,
                                                            "iter_unit": self.iter_unit,
                                                            "job_id": job_id
                                                           })
                    
                    self.eval_process.start()
                    time.sleep(10)
                    if self.check_started() == True:                                                               
                        self.eval_process.join() # Here takes long time to complete 
                        end_time = time.time()

                        et = time.asctime(time.localtime(end_time))
                        ls = time.asctime(time.localtime(self.last_sync_time))
                        debug("Task ended at {} but the final result was synchronized at {}.".format(et, ls))

                    # waits until the final result is synchronized
                        while self.last_sync_time == None or end_time > self.last_sync_time:
                            now = time.asctime()
                        #debug("Waiting the result synchronization at {}.".format(now))
                            time.sleep(1)
                            if self.stop_flag == True:
                                break
                    else:
                        raise RuntimeError("Fail to start the evaluation with {}".format(self.params))
                else:
                    self.eval_func(self.params, 
                                    cur_iter=i, 
                                    max_iters=max_iters, 
                                    iter_unit=self.iter_unit,
                                    job_id=job_id)
                

                result = self.get_cur_result(self.get_device_id())
                self.update_result(i+1, result, base_time)

        except Exception as ex:
            warn("Exception raised on evaluation: {}\n{}".format(ex, sys.exc_info()[0]))
            self.stop_flag = True

        finally:
            with self.thread_cond:
                #self.params = None # FIXME:error occurs when previous job finished after job assigned
                self.thread_cond.notify()
                self.load_results(self.get_device_id())
                debug("Evaluating {} has finished.".format(job_id))
                self.busy = False

    def update_result(self, cur_iter, result, base_time):
        if type(result) == dict and "cur_loss" in result:
            cur_loss = result["cur_loss"]
            cur_acc = None
            cur_dur = None                    
            if "run_time" in result and result["run_time"] > 0:                        
                cur_dur = result["run_time"]
            else:
                cur_dur = time.time() - base_time
            if "cur_acc" in result:
                cur_acc = result['cur_acc']
				
            if "cur_iter" in result:
                cur_iter = result["cur_iter"]

            if "iter_unit" in result:
                iter_unit = result["iter_unit"]

            result = { 
                "run_time": cur_dur,
                "cur_loss": cur_loss,
                "cur_acc" : cur_acc, 
                "cur_iter": cur_iter,
                "iter_unit": self.iter_unit 
            }
            self.results.append(result)
        elif type(result) == list and len(result) > 0:
            self.results = result # update all results            
        elif type(result) == float:
            if math.isnan(result):
                result = sys.float_info.max # Set max number of float when NaN 
            result = { 
                "run_time": time.time() - base_time,
                "cur_loss": result,
                "cur_acc" : 1.0 - cur_loss, 
                "cur_iter": cur_iter,
                "iter_unit": self.iter_unit 
            }
            self.results.append(result)
        else:
            warn("Invalid result format: {}".format(result))
            #raise ValueError("Invalid result")        