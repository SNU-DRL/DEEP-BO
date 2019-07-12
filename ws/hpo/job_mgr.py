import os
import time
import json
import copy

from future.utils import iteritems

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 
from ws.hpo.workers.s_opt import SequentialOptimizer

class HPOJobFactory(object):
    def __init__(self, worker, n_jobs):
        self.n_jobs = n_jobs
        self.worker = worker

    def create(self, jr):
        job = {}
        job['job_id'] = "{}-{}-{}{}".format(self.worker.get_id(), 
                                        self.worker.get_device_id(), 
                                        time.strftime('%Y%m%d',time.gmtime()),
                                        self.n_jobs)
        job['created'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        job['status'] = "created"
        job['result'] = None
        for key in jr.keys():
            job[key] = jr[key]
        
        return job  


class HPOJobManager(ManagerPrototype):
    def __init__(self, run_cfg, hp_cfg, port, use_surrogate=False):

        super(HPOJobManager, self).__init__(type(self).__name__)
        self.jobs = self.get_hpo_jobs()
         
        self.worker = SequentialOptimizer(run_cfg, hp_cfg, "s-opt_{}".format(port))
        self.prefix = self.worker.get_id()
        self.device_id = self.worker.get_device_id()
        
        self.use_surrogate = use_surrogate
        
        self.to_dos = []

    def __del__(self):
        debug("All jobs will be terminated")
        # force to terminate all jobs
        for j in self.jobs:
            j["status"] = 'terminated'

        #self.save_db('hpo_jobs', self.jobs) # TODO:uncomment if you save previous jobs

    def get_config(self):
        # This returns run config
        if self.use_surrogate:
            return {"target_func": "surrogate", "param_order": []}
        return self.worker.get_config() 

    def get_spec(self):
        my_spec = {
            "node_type": "BO Node",
            "id": self.worker.id,
            "device_id": self.worker.get_device_id() }
        return my_spec

    def add(self, args):
        job_id = None
        # TODO: validate parameters
        try:
            f = HPOJobFactory(self.worker, len(self.jobs))
            job = f.create(args)
            self.jobs.append(job)
            debug("Job {} added properly.".format(job['job_id']))

            if self.worker.set_params(args):
                args['status'] = 'assigned'
                self.to_dos.append({"worker": self.worker, "job_id": job['job_id']})
                debug("Job {} assigned properly.".format(job['job_id']))
            else:
                debug("invalid hyperparam vector: {}".format(args))
                raise ValueError("invalid hyperparameters")

        except:
            warn("invalid job description: {}".format(args))
            raise ValueError("invalid job description")
        
        return job['job_id']
    
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

    def get_to_do(self, job_id):
        for w in self.to_dos:
            if w['job_id'] == job_id:
                return w
        
        return None        

    def sync_result(self, job_id='active'):
        for w in self.to_dos:                         
            id = w['job_id']
            if job_id != 'active' and id != job_id:
                id = job_id
            j = self.get_job(id)
            if j['status'] == 'processing' or j['status'] == 'terminated':
                cur_result = w['worker'].get_cur_result()
                if cur_result is not None:
                    self.update(id, **cur_result)
                cur_status = w['worker'].get_cur_status()
                if cur_status == 'idle':
                    self.update(id, status='done')

                break

    def control(self, job_id, cmd):
        aj = self.get_active_job_id()
        if cmd == 'start':
            if aj is not None:
                debug("{} is processing now.".format(aj))
                return False
            w = self.get_to_do(job_id)
            j = self.get_job(job_id)
            if w is not None:
                if w['job_id'] == job_id and j['status'] != 'processing':
                    self.update(job_id, status='processing')
                    w['worker'].start()
                    return True
                else:
                    debug("{} job is already {}.".format(job_id, j['status']))
                    return False
            debug("No {} job is assigned yet.".format(job_id))
            return False
                
        elif cmd == 'pause':
                if aj == job_id:
                    w = self.get_to_do(job_id)
                    w['worker'].pause()
                    # XXX:waiting required until being paused
                    while w['worker'].paused == False:
                        time.sleep(1)

                    self.update(job_id, status='pending')
                    
                    return True
                else:
                    debug("Unable to pause inactive job: {}".format(job_id))
                    return False
        elif cmd == 'resume':
            w = self.get_to_do(job_id)
            j = self.get_job(job_id)
            if w is not None and j['status'] == 'pending':
                w['worker'].resume()
                while w['worker'].paused == True:
                    time.sleep(1)            
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
                w = self.get_to_do(job_id)
                
                if w is not None:
                    debug("{} will be stopped".format(job_id))
                    w['worker'].stop()
                    
                    self.update(job_id, status='terminated')                
                    return True
                else:
                    debug("No such {} in TO-DO list".format(job_id))
                    return False
        warn("No jobs available.")
        return False

    def stop_working_job(self):
        for w in self.to_dos:
            if w['worker'].get_cur_status() == 'processing':
                job_id = w['job_id']
                self.remove(job_id)                
                break

           