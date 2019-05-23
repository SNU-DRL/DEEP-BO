import os
import time
import json
import copy

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 

from ws.hpo.space_mgr import SamplingSpaceManager
from ws.hpo.workers.p_opt import ParallelOptimizer

class ParallelHPOManager(ManagerPrototype):

    def __init__(self, hp_config, **kwargs):
        self.hp_config = hp_config
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

    def prepare(self, exp_time):
        if len(self.pairs) == 0:
            self.match_nodes()
        
        for p in self.pairs:
            hpo = p["optimizer"]
            train = p["trainer"]
            w = ParallelOptimizer(hpo, train, self.hp_config, self.get_credential())
            if exp_time != None:
                jr = w.create_job_request(exp_time=exp_time)
                w.set_job_request(jr)
            self.workers.append(w)

    def control(self, cmd, space_id, exp_time=None, node_id="all"):
        if node_id == "all":
            if len(self.nodes.keys()) < 2:
                debug("Not enough the registered nodes")
                return False
            else:
                self.prepare(exp_time)

        elif node_id in self.nodes:
            # TODO: control single node
            raise NotImplementedError("Controlling each node is not supported yet.")
            return False
        else:
            debug("No such node ID existed: {}".format(node_id))
            return False

        if cmd == 'start':
            if len(self.workers) == 0:
                debug("No worker is prepared.")
                return False
            self.space_mgr.set_space_status(space_id, "active")
            for w in self.workers:
                w.start()
            return True

        elif cmd == 'stop':
            self.space_mgr.set_space_status(space_id, "inactive")
            self.stop_working_job()
            return True

        else:
            debug("Unsupported command: {}".format(cmd))
            return False

    def stop_working_job(self):
        for w in self.workers:
            w.stop()        