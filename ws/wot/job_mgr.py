import os
import time
import json

from future.utils import iteritems

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 
import ws.shared.lookup as lookup

from ws.wot.workers.surrogates import SurrogateEvaluator


class TrainingJobFactory(object):
    def __init__(self, worker, jobs):
        self.jobs = jobs
        self.worker = worker

    def create(self, dataset, model, hpv, cfg):
        job_id = "{}-{}-{}{}".format(self.worker.get_id(), 
                                        self.worker.get_device_id(), 
                                        time.strftime('%Y%m%d',time.gmtime()),
                                        len(self.jobs))

        job = {
            "job_id" : job_id, 
            "created" : time.time(),
            "status" : "not assigned",
            "cur_loss" : None,
            "cur_acc" : None,
            "lr" : [],
            "run_time" : None,
            "times" : [],
            "cur_iter" : 0,
            "iter_unit" : "epoch",
            "dataset" : dataset,
            "model": model,
            "config": cfg,
            "hyperparams" : hpv
        }
        return job        

class TrainingJobManager(ManagerPrototype):
    def __init__(self, worker, use_surrogate=False, retrieve_func=None):

        super(TrainingJobManager, self).__init__(type(self).__name__)
        self.jobs =  self.database['jobs'] #[ dummy_item, ] # XXX:change to empty list in future
         
        self.worker = worker
        self.use_surrogate = use_surrogate
        self.retrieve_func = retrieve_func
        
        self.shelf = None
        
        self.timeout_count = 0
        self.max_timeout = 100 # XXX: For avoiding the shared file not found error

    def __del__(self):
        #debug("All of jobs will be terminated")
        # force to terminate all jobs
        for j in self.jobs:
            j["status"] = 'terminated'

        #self.save_db('jobs', self.jobs)

    def get_config(self):
        if self.use_surrogate:
            return {"target_func": "surrogate", "param_order": []}
        return self.worker.get_config()
        
    def get_spec(self):
        id = {
            "job_type": "ML_trainer",
            "id": self.worker.get_id(),
            "device_id": self.worker.get_device_id() }
        return id

    def add(self, args):
        job_id = None
        # TODO: validate parameters
        try:
            dataset = args['dataset'] # e.g. MNIST, CIFAR-10, 
            model = args['model']  # LeNet, VGG, LSTM, ... 
            hpv = args['hyperparams'] # refer to data*.json for keys
            cfg = args['config'] # max_iter, ...            
            f = TrainingJobFactory(self.worker, self.jobs)
            job = f.create(dataset, model, hpv, cfg)
            job_id = job['job_id']
            
            self.jobs.append(job)
            debug("Job appended properly.")
            
            worker = None
            if self.use_surrogate and "surrogate" in cfg:
                s = cfg['surrogate']
                l = lookup.load(s)
                ffr = None
                if "ffr" in cfg:
                    ffr = cfg['ffr']
                worker = SurrogateEvaluator(s, l, time_slip_rate=ffr)
            else:
                worker = self.worker

            max_epoch = None
            if "max_epoch" in cfg:
                max_epoch = cfg['max_epoch']
                worker.set_max_iters(max_epoch, "epoch")
            elif "max_iter" in cfg:
                max_iter = cfg['max_iter']
                iter_unit = "epoch"
                if "iter_unit" in cfg:
                    iter_unit = cfg['iter_unit']
                worker.set_max_iters(max_iter, iter_unit)    
            
            cand_index = None
            if 'cand_index' in cfg:
                cand_index = cfg['cand_index']

            if worker.set_job_description(hpv, cand_index, job_id):
                job['status'] = 'assigned'
                self.shelf = {"worker": worker, "job_id": job_id, "cand_index": cand_index, "hyperparams": hpv}
                debug("Work item created properly.")
            else:
                debug("Invalid hyperparam vector: {}".format(hpv))
                raise ValueError("Invalid hyperparameters")
            
        except Exception as ex:
            #debug("invalid arguments: {}, {}, {}, {}".format(dataset, model, hpv, cfg))
            warn("Adding job {} failed: {}".format(job, ex))
            raise ValueError("Invalid job description")
        finally:
            return job_id
    
    def get_active_job_id(self):        
        for j in self.jobs:
            if j['status'] == 'processing':
                return j['job_id']
        return None

    def get(self, job_id):
        for j in self.jobs:
            if j['job_id'] == job_id:
                return j
        debug("no such {} job is existed".format(job_id))
        return None        

    def get_all_jobs(self, n=10):
        
        if len(self.jobs) <= n: 
            return self.jobs
        else:
            selected_jobs = self.jobs[-n:]
            debug("number of jobs: {}".format(len(selected_jobs)))
            return selected_jobs

        

    def sync_result(self):
        t = self.shelf
        if t == None:
            return

        #debug("Work item: {}".format(t['job_id']))
        j = self.get(t['job_id'])
        cur_status = t['worker'].get_cur_status()
        if cur_status == 'processing':
            if self.retrieve_func != None:
                t['worker'].sync_result(self.retrieve_func)

            cur_result = t['worker'].get_cur_result(t['worker'].get_device_id())
            if cur_result != None:
                sync_time = time.time()
                debug("[{}] Intermidiate result synchronized at {}.".format(t['job_id'], time.asctime(time.gmtime(sync_time))))
                self.timeout_count = 0 # reset timeout count               
                self.update(t['job_id'], **cur_result)
                t['worker'].set_sync_time(sync_time)
            else:                                 
                debug("[{}] Result is not updated.".format(t['job_id']))
                self.timeout_count += 1
                if self.timeout_count > self.max_timeout:
                    self.remove(t['job_id'])

        elif cur_status == 'idle':
            self.update(t['job_id'], status='done')

    def update_result(self, cur_iter, iter_unit, cur_loss, run_time):
        t = self.shelf
        if t['worker'].get_cur_status() == 'processing':
            job_id = t['job_id']
            t['worker'].add_result(cur_iter, cur_loss, run_time, iter_unit)
            debug("The result is updated at {} {} ".format(cur_iter, iter_unit))
        else:
            warn("Invalid state - update request for inactive task.")

    def control(self, job_id, cmd):
        aj = self.get_active_job_id()
        if cmd == 'start':
            if aj is not None:
                debug("{} is processing now.".format(aj))
                return False
            w = self.shelf
            j = self.get(w['job_id'])
            if job_id == w['job_id'] and j != None:
                if j['status'] != 'processing':
                    while w['worker'].start() == False:
                        for t in self.shelf:
                            if t['job_id'] == job_id:                                
                                w['worker'].set_job_description(t['hyperparams'],
                                                                t['cand_index'],
                                                                job_id)
                        time.sleep(1)
                    self.update(job_id, status='processing')
                    return True
                else:
                    debug("{} job is already {}.".format(job_id, w['status']))
                    return False
            debug("No {} job is assigned yet.".format(job_id))
            return False
        elif cmd == 'pause':
                if aj == job_id:
                    w = self.shelf
                    w['worker'].pause()
                    self.update(job_id, status='pending')
                    
                    return True
                else:
                    debug("Unable to pause inactive job: {}".format(job_id))
                    return False
        elif cmd == 'resume':
            w = self.shelf            
            if w != None and w['status'] == 'pending':
                w['worker'].resume()            
                self.update(job_id, status='processing')
                return True
            else:
                debug('Unable to resume inactive job: {}'.format(job_id))
                return False
        else:
            debug("Unsupported command: {}".format(cmd))
            return False

    def update(self, job_id, **kwargs):
        for j in self.jobs:
            if j['job_id'] == job_id:
                for (k, v) in iteritems(kwargs):
                    if k in j:
                        j[k] = v
                        #debug("{} of {} is updated: {}".format(k, job_id, v))
                    else:
                        debug("{} is invalid in {}".format(k, job_id))

    def remove(self, job_id):
        for j in self.jobs:
            if j['job_id'] == job_id and j['status'] != 'terminated':
                w = self.shelf
                
                if w['job_id'] != job_id:
                    debug("Request to remove: {}, Working item: {}".format(job_id, w['job_id']))
                    job_id = w['job_id']

                debug("{} will be stopped".format(job_id))
                w['worker'].stop()
                self.update(job_id, status='terminated')                
                return True

        warn("No such jobs available: {}".format(job_id))
        return False

    def stop_working_job(self):
        t = self.shelf
        if t['worker'].get_cur_status() == 'processing':
            job_id = t['job_id']
            self.remove(job_id)                
            

           