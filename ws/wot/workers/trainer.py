import os
import threading
import time
import traceback
import copy
import pickle
import subprocess
import json

from ws.shared.logger import *
from ws.shared.worker import Worker

class Trainer(Worker):

    def __init__(self, id=None, fork=False):
        self.fork = fork
        self.device_type = 'cpu'
        self.device_index = 0
        self.last_sync_time = None

        if id == None:
            id = 'trainer_proto'

        super(Trainer, self).__init__(id)
        self.reset()

    def set_resource(self, device_type, device_index):
        self.device_type = device_type
        self.device_index = device_index
		
    def get_device_id(self):
        return "{}{}".format(self.device_type, self.device_index)
		
    def reset(self):
        self.results = []
        try:
            pkl = "{}.pkl".format(self.get_device_id())
            os.remove(pkl)
        except OSError:
            pass

    def sync_result(self, retrieve_func):
        result = retrieve_func()
        # TODO: validate result contents
        if type(result) == list:
            if len(result) > 0:            
                self.results = result
            else:
                debug("No result found yet.")
        elif type(result) == dict:
            self.results.append(result)
        else:
            warn("Invalid result: {}".format(result))

        self.dump_results()

    def set_sync_time(self, sync_time):
        #debug("Result had been synched at {}".format(time.asctime(time.localtime(sync_time))))
        self.last_sync_time = sync_time

    def set_job_description(self, params, index=None, job_id=None):
        if job_id != None:
            self.job_id = job_id

        if params:
            debug("Assigned parameters: {}".format(params))
            self.params = params
            return True
        else:
            debug("Invalid parameters: {}".format(params))
            return False

    def is_forked(self):
        return self.fork
		
    def is_working(self):
        return False
    def dump_results(self):
        if self.is_forked() == True:
            pkl = "{}.pkl".format(self.get_device_id())
            with open("{}".format(pkl), 'wb') as f:
                pickle.dump(self.results, f)
				
    def load_results(self, device_id):
        pkl = "{}.pkl".format(device_id)        
        try:            
            with open("{}".format(pkl), "rb") as f:                
                self.results = pickle.load(f)
        except Exception as ex:
            self.results = []

    def get_cur_result(self, device_id):
        if self.is_forked() == True:
            self.load_results(device_id)
        
        if len(self.results) > 0:
            latest = self.results[-1]
            result = copy.copy(latest)
            result['lr'] = [copy.copy(r['cur_loss']) for r in self.results]
            result['run_time'] = latest['run_time']
            return result
        else:
            try:
                if "arguments" in self.config:
                    if "fail_err" in self.config['arguments']:
                        default_err = self.config['defaults'][-1]
                        default_result = {"cur_iter": 0, 
                                "iter_unit": "epoch",
                                "cur_loss": default_err,
                                "loss_type": "error_rate", 
                                "run_time": 0.0
                        }
                        return default_result
                    else:
                        return None                        
            except Exception as ex:
                return None 

    def add_result(self, cur_iter, cur_loss, run_time, 
                   iter_unit="epoch",
                   loss_type="error_rate"):
        
        if loss_type == "error_rate":
            cur_acc = 1.0 - cur_loss
        else:
            cur_acc = None
        
        result = { "cur_iter": cur_iter, 
                  "iter_unit": iter_unit,
                  "cur_loss": cur_loss,
                  "loss_type" : loss_type, 
                  "cur_acc": cur_acc, 
                  "run_time": run_time
        }

        self.results.append(result)
        self.dump_results()

    def check_started(self):
        if self.device_type == 'gpu':
            try:
			    # Assume that gpustat installed properly
                result = subprocess.check_output('gpustat --json', shell=True)
                gpu_dict = json.loads(result)
                for g in gpu_dict['gpus']:
                    if g['index'] == int(self.device_index):
                        debug("Working processes on {}: {}".format(self.get_device_id(), g['processes']))
                        if len(g['processes']) > 0:
                            return True
                        else:
                            return False
                error("No {} device found: {}".format(self.get_device_id(), gpu_dict))
                return False
            except Exception as ex:
                debug("Checking GPU processes failed: {}".format(ex))
                return True
        else:
            return True 

    def execute(self):
        ''' Execute target function and append an intermidiate result per epoch to self.results.
        The result is a dictionary object which has following attributes: 
          - "run_time" : float, run time (elapsed time for the given epoch) 
          - "cur_loss": float, current loss value
          - "cur_acc": float, current accuracy value. if current loss is an error rate, it becomes 1.0 - cur_loss. (optional)
          - "cur_iter": integer, number of current iterations
          - "iter_unit" : string, epoch or steps will be accepted
        '''
        raise NotImplementedError('execute() should be overrided.')