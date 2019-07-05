import os
import threading
import time
import traceback
import copy
import pickle

from ws.shared.logger import *
from ws.shared.worker import Worker

class Trainer(Worker):

    def __init__(self, id=None, fork=False):
        self.fork = fork
        self.config = {}
        self.device_id = 'cpu0'
        self.last_sync_time = None

        if id == None:
            id = 'trainer_proto'

        super(Trainer, self).__init__(id)
        self.reset()

    def set_resource(self, device_type, device_index):
        self.device_id = "{}{}".format(device_type, device_index)
		
    def get_device_id(self):
        return self.device_id
		
    def reset(self):
        self.results = []
        try:
            pkl = "{}.pkl".format(self.device_id)
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
        debug("synched at {}".format(time.asctime(time.gmtime(sync_time))))
        self.last_sync_time = sync_time

    def set_job_description(self, params, index=None, job_id=None):
        if job_id != None:
            self.job_id = job_id

        if params:
            debug("Assigned parameters: {}".format(params))
            self.params = params
            return True
        else:
            debug("Invalid params: {}".format(params))
            return False

    def is_forked(self):
        return self.fork
		
    def dump_results(self):
        if self.is_forked() == True:
            pkl = "{}.pkl".format(self.device_id)
            with open("{}".format(pkl), 'wb') as f:
                pickle.dump(self.results, f)
				
    def load_results(self, device_id):
        pkl = "{}.pkl".format(device_id)        
        try:            
            with open("{}".format(pkl), "rb") as f:                
                self.results = pickle.load(f)
        except Exception as ex:
            debug("Read error: {}".format(ex))
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
            return None

    def add_result(self, cur_iter, cur_loss, run_time, iter_unit="epoch"):
        if cur_loss < 1.0 and cur_loss > 0.0:
            cur_acc = 1.0 - cur_loss
        else:
            cur_acc = None
        result = {"cur_iter": cur_iter, "iter_unit": iter_unit,
            "cur_loss": cur_loss, "cur_acc": cur_acc, "run_time": run_time}

        self.results.append(result)
        self.dump_results()

    def execute(self):
        ''' Execute target function and append an intermidiate result per epoch to self.results.
        The result is a dictionary object which has following attributes: 
          - "run_time" : float, run time (elapsed time for the given epoch) 
          - "cur_loss": float, current loss value
          - "cur_acc": float, current accuracy value. typically 1.0 - cur_loss. (optional)
          - "cur_iter": integer, number of current iterations
          - "iter_unit" : string, epoch or steps will be accepted
        '''
        raise NotImplementedError('execute() should be overrided.')