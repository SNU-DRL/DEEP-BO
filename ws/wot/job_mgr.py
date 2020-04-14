import os
import time
import json

from future.utils import iteritems

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 


from ws.wot.workers.surrogates import SurrogateEvaluator


class TrainingJobFactory(object):
    def __init__(self, worker, jobs):
        self.jobs = jobs
        self.worker = worker

    def create(self, dataset, model, hpv, cfg):
        job_id = "{}-{}-{}-{}".format(self.worker.get_id(), 
                                     self.worker.get_device_id(), 
                                     time.strftime('%Y%m%d',time.localtime()),
                                     time.strftime('%H%M%S',time.localtime()))

        job = {
            "job_id" : job_id, 
            "created" : time.time(),
            "status" : "created",
            "cur_loss" : None,
            "cur_acc" : None,
            "lr" : [],
            "run_time" : None,
            "times" : [],
            "cur_iter" : 0,
            "iter_unit" : "epoch",
            "loss_type" : None,
            "dataset" : dataset,
            "model": model,
            "config": cfg,
            "hyperparams" : hpv
        }
        return job        

class TrainingJobManager(ManagerPrototype):
    def __init__(self, worker, use_surrogate=False, retrieve_func=None):

        super(TrainingJobManager, self).__init__(type(self).__name__)
        self.jobs = self.get_train_jobs()
         
        self.worker = worker
        self.use_surrogate = use_surrogate
        self.retrieve_func = retrieve_func
        
        self.work_item = None
        
        self.timeout_count = 0
        self.max_timeout = 100000 # XXX: For avoiding the result file not found error

    def __del__(self):
        #debug("All of jobs will be terminated")
        # force to terminate all jobs
        for j in self.jobs:
            j["status"] = 'terminated'

        #self.save_db('train_jobs', self.jobs) # XXX:uncomment if you want to keep job history 

    def get_config(self):
        if self.use_surrogate:
            return {"target_func": "surrogate", "param_order": []}
        return self.worker.get_config()
        
    def get_spec(self):
        id = {
            "node_type": "Training Node",
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
                import ws.shared.lookup as lookup
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
                self.work_item = {
                    "worker": worker, 
                    "job_id": job_id, 
                    "cand_index": cand_index, 
                    "hyperparams": hpv 
                }
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

    def get_job(self, job_id):
        for j in self.jobs:
            if j['job_id'] == job_id:
                return j
        debug("no such {} job is existed".format(job_id))
        return None        

    def get_all_jobs(self):
        
        job_ids = []
        for j in self.jobs:
            job_ids.append(j['job_id'])
        
        return job_ids

    def sync_result(self, job_id='active'):
        
        if self.work_item == None:
            return
        w = self.work_item
        if job_id == 'active':
            job_id = w['job_id']

        if job_id != w['job_id']:
            warn("Something wrong happens in result sync.- {}:{}".format(job_id, w['job_id']))
        
        #debug("Work item: {}".format(t['job_id']))
        j = self.get_job(job_id)
        cur_status = w['worker'].get_cur_status()
        if cur_status == 'processing':
            if self.retrieve_func != None:
                w['worker'].sync_result(self.retrieve_func)

            cur_result = w['worker'].get_cur_result(w['worker'].get_device_id())
            if cur_result != None:
                sync_time = time.time()
                #debug("The result of {} updated at {}.".format(job_id, 
                #                                          time.asctime(time.localtime(sync_time))))
                self.timeout_count = 0 # reset timeout count               
                self.update(w['job_id'], **cur_result)
                w['worker'].set_sync_time(sync_time)
            else:                                 
                self.timeout_count += 1
                if self.timeout_count > self.max_timeout:
                    self.remove(w['job_id'])
                elif w['worker'].is_working() == False:
                    warn("Job {} finished with no result".format(w['job_id']))
                    self.remove(w['job_id'])
                else:
                    debug("An interim result is not updated yet. Timeout: {}/{}".format(self.timeout_count, self.max_timeout))

        elif cur_status == 'idle':
            self.update(w['job_id'], status='done')

    def update_result(self, cur_iter, cur_loss, run_time, 
                      iter_unit='epoch',
                      loss_type='error_rate'):
        t = self.work_item
        if t['worker'].get_cur_status() == 'processing':
            job_id = t['job_id']
            t['worker'].add_result(cur_iter, cur_loss, run_time, 
                                    iter_unit=iter_unit, loss_type=loss_type)
            #debug("The result is updated at {} {} ".format(cur_iter, iter_unit))
        else:
            warn("Invalid state - update request for inactive task.")

    def control(self, job_id, cmd):
        aj = self.get_active_job_id()
        if cmd == 'start':
            if aj is not None:
                debug("{} is processing now.".format(aj))
                return False
            w = self.work_item
            j = self.get_job(w['job_id'])
            if job_id == w['job_id'] and j != None:
                if j['status'] != 'processing':
                    while w['worker'].start() == False:
                        warn("Killing the zombie process...")
                        w['worker'].stop()
                        time.sleep(3) # Waiting until the worker being stopped
                        w['worker'].set_job_description(w['hyperparams'],
                                                        w['cand_index'],
                                                        job_id)
                        
                    self.update(job_id, status='processing')
                    return True
                else:
                    debug("{} job is already {}.".format(job_id, w['status']))
                    return False
            else:
                debug("No {} job is assigned yet.".format(job_id))
                return False
        elif cmd == 'pause':
                if aj == job_id:
                    w = self.work_item
                    w['worker'].pause()
                    self.update(job_id, status='pending')
                    
                    return True
                else:
                    debug("Unable to pause inactive job: {}".format(job_id))
                    return False
        elif cmd == 'resume':
            w = self.work_item            
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
        debug("Job termination request accepted: {}".format(job_id))
        w = self.work_item                
                
        if w['job_id'] == job_id:
            debug("{} will be stopped".format(w['job_id']))
            w['worker'].stop()
            self.update(w['job_id'], status='terminated')
            return True
        else:
            warn("{} is not working! Current job is {}.".format(job_id, w['job_id']))
            return False          

    def stop_working_job(self):
        t = self.work_item
        if t['worker'].get_cur_status() == 'processing':
            job_id = t['job_id']
            self.remove(job_id)                
            

           