import os
import time
import json
import copy

from future.utils import iteritems

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 

from ws.hpo.space_mgr import SamplingSpaceManager
from ws.hpo.workers.p_opt import ParallelOptimizer

class HPOJobFactory(object):
    def __init__(self, workers, n_jobs):
        self.n_jobs = n_jobs
        self.workers = workers

    def create(self, jr):
        job = {}
        job['job_id'] = "batch-{}p-{}{}".format(len(self.workers), 
                                        time.strftime('%Y%m%d',time.gmtime()),
                                        self.n_jobs)
        job['created'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        job['status'] = "not assigned"
        job['result'] = None
        for key in jr.keys():
            job[key] = jr[key]
        
        return job  


class ParallelHPOManager(ManagerPrototype):

    def __init__(self, hp_config, **kwargs):
        self.hp_config = hp_config
        self.jobs = []
        self.nodes = {}
        self.pairs = []
        self.workers = []
        self.space_mgr = SamplingSpaceManager()
        return super(ParallelHPOManager, self).__init__(type(self).__name__)

    def __del__(self):
        self.stop_working_job()

    def get_config(self):
        cfg = { 
            "nodes" : self.nodes,
            "pairs" : self.pairs
        }
        return cfg

    def get_spec(self):
        my_spec = {
            "job_type": "HPO_space",
            "type": self.type }
        return my_spec

    def get_space_manager(self):
        return self.space_mgr

    def get_job(self, job_id):               
        for j in self.jobs:
            if j['job_id'] == job_id:
                return j
        debug("no such {} job is existed".format(job_id))
        return None 

    def get_active_job_id(self):        
        for j in self.jobs:
            if j['status'] == 'processing':
                return j['job_id']
        return None

    def register(self, node_spec):
        ip = None
        port = None
        job_type = None

        if "ip_address" in node_spec:
            ip = node_spec["ip_address"]
        else:
            raise ValueError("No IP address in specification")
        
        if "port_num" in node_spec:
            port = node_spec["port_num"]
        else:
            raise ValueError("No port number in specification")

        if "job_type" in node_spec:
            job_type = node_spec["job_type"]
        else:
            raise ValueError("No job type in specification")
        
        node_id = self.check_registered(ip, port)
        
        if node_id != None:
            debug("Node already registered: {}:{}".format(ip, port))
            return node_id, 200

        # Try handshaking with registered node to check it is healthy.
        url = "http://{}:{}".format(ip, port)
        # TODO:Check job type is compatible

        # Create node id and append to node repository
        node_id = "{}_node_{:03d}".format(job_type, len(self.nodes.keys()))
        node_spec = {
            "id" : node_id,
            "ip_address" : ip,
            "port_num" : port,
            "job_type" : job_type,
            "status" : "registered"
        }
        
        self.nodes[node_id] = node_spec
        return node_id, 201

    def check_registered(self, ip_addr, port):
        for nk in self.nodes.keys():
            n = self.nodes[nk]
            if n["ip_address"] == ip_addr and n["port_num"] == port:
                return n["id"]
        return None

    def get_pairs(self):
        return self.pairs

    def get_node(self, node_id):
        if node_id == "all":
            nodes = [ n for n in self.nodes]
            return nodes
        elif node_id in self.nodes:
            return self.nodes[node_id]
        else:
            return None

    def match_nodes(self):
        optimizers = []
        trainers = []
        for k in self.nodes.keys():
            n = self.nodes[k]
            if n["status"] != "paired":
                if n["job_type"] == "HPO_runner":
                    optimizers.append(n["id"])
                elif n["job_type"] == "ML_trainer":
                    trainers.append(n["id"])
                else:
                    warn("Invalid job type of node: {}".format(n["job_type"]))

        # pairing with optimizer and trainer one by one
        for i in range(len(optimizers)):
            if i < len(trainers):
                opt = self.nodes[optimizers[i]]
                train = self.nodes[trainers[i]]
                pair = {"optimizer" : opt, 
                        "trainer" : train
                }
                opt["status"] = "paired"
                train["status"] = "paired"
                self.pairs.append(pair)
            else:
                break

    def create_new_space(self, 
                        num_samples=20000,
                        grid_seed=1,
                        surrogate=None):
        
        space_spec = { 'hp_config' : self.hp_config.get_dict(),
                       "num_samples": num_samples,
                       "grid_seed": grid_seed}
        if surrogate != None:
            space_spec = { "surrogate" : surrogate }

        space_id = self.space_mgr.create(space_spec)
        
        return space_id

    def validate_space(self, space_id):
        if space_id in self.space_mgr.get_available_spaces():
            return True
        else:
            return False

    def prepare(self, args):
        # Create new result history
        num_samples = 20000
        if "num_samples" in args:
            num_samples = args['num_samples']                
        grid_seed = 1
        if 'grid_seed' in args:
            grid_seed = args['grid_seed']
        surrogate = None
        if 'surrogate' in args:
            surrogate = args['surrogate']
        space_id = self.create_new_space(num_samples, grid_seed, surrogate)
        args['space_id'] = space_id
        
        self.workers = []
        if len(self.pairs) == 0:
            self.match_nodes()
        
        for p in self.pairs:
            hpo = p["optimizer"]
            train = p["trainer"]
            w = ParallelOptimizer(hpo, train, self.hp_config, self.get_credential())
            jr = w.create_job_request(**args)
            #debug("worker job description: {}".format(jr))
            w.set_job_request(jr)
            self.workers.append(w)
        if len(self.workers) < 1:
            return False
        else:
            return True

    def add(self, args):
        # TODO: validate parameters
        try:            
            if not self.prepare(args):
                raise ValueError("invalid job description")

            f = HPOJobFactory(self.workers, len(self.jobs))
            job = f.create(args)            
            self.jobs.append(job)
            debug("Job added properly: {}".format(job))
        except:
            warn("invalid job description: {}".format(args))
            raise ValueError("invalid job description")
        
        return job['job_id']

    def control(self, job_id, cmd):

        if job_id == self.get_active_job_id():
            debug("{} is processing now.".format(aj))
            return False            

        j = self.get_job(job_id)

        if not 'space_id' in j:
            return False

        space_id = j['space_id']
        if cmd == 'start':
            if len(self.workers) == 0:
                debug("No worker is prepared.")
                return False
            
            if j['status'] == 'processing':
                debug("{} job is already {}.".format(job_id, j['status']))
                return False
            self.update(job_id, status='processing')
            self.space_mgr.set_space_status(space_id, "active")
            for w in self.workers:
                w.start()
            return True

        elif cmd == 'stop':
            if j['status'] != 'processing':
                debug("{} job is not working.".format(job_id))
                return False                
            self.space_mgr.set_space_status(space_id, "inactive")
            self.stop_working_job()
            return True

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

    def stop_working_job(self):
        for w in self.workers:
            w.stop()

    def sync_result(self):
        id = self.get_active_job_id()
        j = self.get_job(id)
        space_id = self.space_mgr.get_active_space_id()
        samples = self.space_mgr.get_samples(space_id)
        cur_errs = samples.get_errors("completes")
        # find index of current min error
        min_err_i = None
        cur_best_err = None
        cur_best_hpv = None
        i = 0
        for c in cur_errs:
            if cur_best_err == None or c < cur_best_err:
                cur_best_err = c
                min_err_i = i
            i += 1
        min_err_id = samples.get_completes()[min_err_i]
        cur_best_hpv = samples.get_hpv(min_err_id)

        summary = {             
            "best_err" : cur_best_err,
            "best_hpv" : cur_best_hpv
        }
        cur_result = {"result" : summary }
        self.update(id, **cur_result)
        
    def get_all_jobs(self, n=10):        
        if len(self.jobs) <= n: 
            return self.jobs
        else:
            selected_jobs = self.jobs[-n:]
            debug("number of jobs: {}".format(len(selected_jobs)))
            return selected_jobs
