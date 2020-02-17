import json
import time

from ws.shared.logger import * 
from ws.shared.worker import Worker 

from hpo.connectors.remote_hpo import RemoteOptimizerConnector

class ParallelOptimizer(Worker):
    def __init__(self, hpo_node, train_node, hp_config, credential,   
                 id=None, polling_interval=5):
        self.hpo_node = hpo_node
        self.train_node = train_node
        self.hp_config = hp_config
        self.credential = credential
        self.jobs = []
        self.polling_interval = polling_interval
        self.in_progress = False

        self.connector = RemoteOptimizerConnector(self.hpo_node["ip_address"], 
                                                  self.hpo_node["port_num"], 
                                                  self.credential)

        return super(ParallelOptimizer, self).__init__(id=id)

    def create_job_request(self, mode="DIV", spec="RANDOM", 
                           exp_crt="TIME", exp_time="24h", exp_goal=0.9999, 
                           num_trials=1,
                           space_id="None",
                           goal_metric="error", 
                           surrogate=None):
        job_desc = {}
        job_desc['exp_crt'] = exp_crt
        job_desc['exp_time'] = exp_time
        job_desc['exp_goal'] = exp_goal
        job_desc['num_trials'] = num_trials
        job_desc['goal_metric'] = goal_metric
        job_desc['mode'] = mode
        job_desc['spec'] = spec

        trainer_url = "http://{}:{}".format(self.train_node["ip_address"], self.train_node["port_num"])
        job_desc['train_node'] = trainer_url
        job_desc['space_id'] = space_id
        if surrogate != None:
            job_desc['surrogate'] = surrogate
        else:
            job_desc['hp_cfg'] = self.hp_config.get_dict()
        
        return job_desc

    def wait_until_done(self, method='polling'):
        cur_result = None
        if method == 'polling':
            try:
                while not self.stop_flag: # XXX: infinite loop
                    self.in_progress = True
                    j = self.connector.get_job("active")
                    if j != None:
                        if "result" in j:
                            pass
                        else:
                            raise ValueError("Invalid job description: {}".format(j))
                    else:
                        # working job is finished
                        break 
                    time.sleep(self.polling_interval)

            except Exception as ex:
                warn("Something goes wrong in remote worker: {}".format(ex))
            finally:
                self.in_progress = False
                
        else:
            raise NotImplementedError("No such waiting method implemented")

    def set_job_request(self, job_req):
        # TODO: validate job request
        self.job_req = job_req

    def optimize(self, job_req):
        if self.connector.validate():
            job_id = self.connector.create_job(job_req)
            if job_id is not None:                
                if self.connector.start(job_id):
                                        
                    self.jobs.append({"id": job_id, "options" : job_req, "status" : "run"}) 
                    
                    self.wait_until_done()

                    result = self.get_current_result(job_id)
                    for job in self.jobs:
                        if job['id'] == job_id:
                            job["result"] = result
                            job["status"] = "done"
                    
                    return result

                else:
                    error("Starting HPO job failed.")
            else:
                error("Creating job failed") 
        else:
            raise TypeError("Trainer validation failed")  

    def get_results(self):
        results = []
        try:
            debug("trying to getting all results from {}".format(self.jobs))
            for job in self.jobs:
                if job["status"] == "done":
                    results.append({ "job_id": job["id"], "result": job["result"]})

        except Exception as ex:
            error("getting the results failed")
        return results
    
    def get_current_result(self, job_id):
        try:
            j = self.connector.get_job(job_id)
            if "result" in j:
                return j["result"]
            else:
                return None
        except Exception as ex:
            warn("Something goes wrong in remote worker: {}".format(ex))
            return None

    def execute(self):
        if self.job_req == None:
            self.job_req = self.create_job_request()
        
        return self.optimize(self.job_req)

    def check_active(self):
        return self.connector.check_active()

