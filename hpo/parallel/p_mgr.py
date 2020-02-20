import os
import time
import json
import copy

from future.utils import iteritems

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 

from hpo.space_mgr import SearchSpaceManager
from hpo.workers.p_opt import ParallelOptimizer

class HPOJobFactory(object):
    def __init__(self, workers, n_jobs):
        self.n_jobs = n_jobs
        self.workers = workers

    def create(self, jr):
        job = {}
        job['job_id'] = "batch-{}p-{}{}".format(len(self.workers), 
                                        time.strftime('%Y%m%d',time.localtime()),
                                        self.n_jobs)
        job['created'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        job['status'] = "created"
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
        self.space_mgr = SearchSpaceManager()
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
            "node_type": "Master Node",
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
        node_type = None

        if "ip_address" in node_spec:
            ip = node_spec["ip_address"]
        else:
            raise ValueError("No IP address in specification")
        
        if "port_num" in node_spec:
            port = node_spec["port_num"]
        else:
            raise ValueError("No port number in specification")

        if "node_type" in node_spec:
            node_type = node_spec["node_type"]
        else:
            raise ValueError("No node type in specification")
        
        node_id = self.check_registered(ip, port)
        
        if node_id != None:
            debug("Node already registered:{}".format(node_id))
            return node_id, 200

        # Try handshaking with registered node to check it is healthy.
        url = "http://{}:{}".format(ip, port)
        # TODO:Check job type is compatible

        # Create node id and append to node repository
        node_id = "{}-{:03d}".format(node_type, len(self.nodes.keys()))
        node_spec = {
            "id" : node_id,
            "ip_address" : ip,
            "port_num" : port,
            "node_type" : node_type,
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
                if n["node_type"] == "BO Node":
                    optimizers.append(n["id"])
                elif n["node_type"] == "Training Node":
                    trainers.append(n["id"])
                else:
                    warn("Invalid type of node: {}".format(n["node_type"]))

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
        # Default settings to create new result history
        num_samples = 20000
        grid_seed = 1
        surrogate = None

        if "num_samples" in args:
            num_samples = args['num_samples']                
        
        if 'seed' in args:
            grid_seed = args['seed']
        
        if 'surrogate' in args:
            surrogate = args['surrogate']
        
        if 'space_id' in args and self.validate_space(args['space_id']):
            space_id = args['space_id']
        else:
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
            warn("Workers are not prepared")
            return False
        else:
            return True

    def add(self, args):
        # TODO: validate parameters
        try:            
            if not self.prepare(args):
                raise ValueError("Preparation failed: {}".format(args))

            f = HPOJobFactory(self.workers, len(self.jobs))
            job = f.create(args)            
            self.jobs.append(job)
            debug("Job added properly: {}".format(job))
        except Exception as ex:
            warn("Job add failed: {}".format(ex))
            return None
        
        return job['job_id']

    def remove(self, job_id):
        self.control(job_id, "stop")

    def control(self, job_id, cmd):

        j = self.get_job(job_id)

        if not 'space_id' in j:
            return False

        space_id = j['space_id']
        if cmd == 'start':
            if job_id == self.get_active_job_id():
                debug("{} is processing now.".format(job_id))
                return False      

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
            if job_id != self.get_active_job_id():
                debug("{} is not processing now.".format(job_id))
                return False 

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

    def sync_result(self, job_id='active'):
        if job_id == 'active':
            id = self.get_active_job_id()
        else:
            id = job_id
        j = self.get_job(id)
        space_id = self.space_mgr.get_active_space_id()
        samples = self.space_mgr.get_samples(space_id)
        if samples == None:
            debug("No parameter space initialized")
            return
        cur_errs = samples.get_errors("completions")
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
        min_err_id = samples.get_completions()[min_err_i]
        if min_err_id != None:
            cur_best_hpv = samples.get_hpv_dict(min_err_id)

        summary = {             
            "best_err" : cur_best_err,
            "best_hpv" : cur_best_hpv
        }
        cur_result = {"result" : summary }
        self.update(id, **cur_result)

        # check whether all HPO nodes terminated and update status
        all_terminated = True
        for w in self.workers:
            if w.check_active() == True:
                all_terminated = False
                break
        if all_terminated:
            self.update(id, status="done")
            self.space_mgr.set_space_status(space_id, "inactive")

    def get_all_jobs(self):        
        job_ids = []
        for j in self.jobs:
            job_ids.append(j['job_id'])
        
        return job_ids
