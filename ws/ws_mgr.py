import os
import time
import json
import copy

import multiprocessing as mp

from ws.shared.logger import * 
from ws.shared.proto import ManagerPrototype 

from ws.resources.billboard import Billboard
from ws.resources.config import Config
from ws.resources.jobs import Jobs
from ws.resources.job import Job

from flask import Flask
from flask_restful import Api


class WebServiceManager(ManagerPrototype):

    def __init__(self, job_mgr, hp_cfg,
                 credential=None):
        super(WebServiceManager, self).__init__(type(self).__name__)
        self.app = Flask(self.type)
        self.api = Api(self.app)
        self.job_mgr = job_mgr
        
        if credential != None:
            self.save_db('credential', credential)

        self.hp_cfg = hp_cfg        
        self.my_process = None
        
        self.initialize()

    def initialize(self):
        # For profile
        self.api.add_resource(Billboard, "/", # for profile and 
                        resource_class_kwargs={'resource_manager': self})
        
        self.api.add_resource(Config, "/config/", # for run spec
                        resource_class_kwargs={'job_manager': self.job_mgr, "hp_config": self.hp_cfg})
        
        # For job handling
        self.api.add_resource(Jobs, "/jobs/", 
                        resource_class_kwargs={'job_manager': self.job_mgr})
        self.api.add_resource(Job, "/jobs/<string:job_id>/", 
                        resource_class_kwargs={'job_manager': self.job_mgr})

        if self.job_mgr.type == "ParallelHPOManager":
            from ws.resources.candidates import Candidates
            from ws.resources.completions import Completions
            from ws.resources.grid import Grid
            from ws.resources.hparams import HyperparamVector
            from ws.resources.errors import ObservedErrors
            from ws.resources.error import ObservedError
            from ws.resources.spaces import Spaces
            from ws.resources.space import Space 
            from ws.resources.nodes import Nodes
            from ws.resources.node import Node            
            # For managing HPO nodes
            self.api.add_resource(Nodes, "/nodes/", 
                            resource_class_kwargs={'node_manager': self.job_mgr})
            self.api.add_resource(Node, "/nodes/<string:node_id>/", 
                            resource_class_kwargs={'node_manager': self.job_mgr})                                

            space_mgr = self.job_mgr.get_space_manager()
            # For managing parameter space and history sharing
            self.api.add_resource(Spaces, "/spaces/", 
                            resource_class_kwargs={'space_manager': space_mgr})    
            self.api.add_resource(Space, "/spaces/<string:space_id>/", 
                            resource_class_kwargs={'space_manager': space_mgr})
            self.api.add_resource(Grid, "/spaces/<string:space_id>/grid/<string:sample_id>/", 
                            resource_class_kwargs={'space_manager': space_mgr})
            self.api.add_resource(HyperparamVector, "/spaces/<string:space_id>/vectors/<string:sample_id>/", 
                            resource_class_kwargs={'space_manager': space_mgr})
            self.api.add_resource(Completions, "/spaces/<string:space_id>/completions/", 
                            resource_class_kwargs={'space_manager': space_mgr})
            self.api.add_resource(Candidates, "/spaces/<string:space_id>/candidates/", 
                            resource_class_kwargs={'space_manager': space_mgr})                                         
            self.api.add_resource(ObservedErrors, "/spaces/<string:space_id>/errors/", 
                            resource_class_kwargs={'space_manager': space_mgr})   
            self.api.add_resource(ObservedError, "/spaces/<string:space_id>/errors/<string:sample_id>/", 
                            resource_class_kwargs={'space_manager': space_mgr})                    

    def get_spec(self):
        return self.job_mgr.get_spec()

    def get_urls(self):
        urls = [
            {"/": {"method": ['GET']}},
            {"/config/": {"method": ['GET']}}
        ]
        
        job_urls = [
            {"/jobs/": {"method": ['GET', 'POST']}},
            {"/jobs/active/": {"method": ['GET']}},
            {"/jobs/[job_id]/": {"method": ['GET', 'PUT', 'DELETE']}}
        ] 

        space_urls = [ 
            {"/spaces/[space_id]/": {"method": ['GET']}},
            {"/spaces/[space_id]/completions/": {"method": ['GET']}},
            {"/spaces/[space_id]/candidates/": {"method": ['GET']}},
            {"/spaces/[space_id]/grid/[id]/": {"method": ['GET']}},
            {"/spaces/[space_id]/vectors/[id]/": {"method": ['GET']}},
            {"/spaces/[space_id]/errors/[id]/": {"method": ['GET', 'PUT']}}
        ]

        node_urls = [
            {"/nodes/": {"method": ['GET', 'POST', 'PUT', 'DELETE']}},
            {"/nodes/[node_id]/": {"method": ['GET', 'PUT', 'DELETE']}}
        ] 

        if self.job_mgr.type == "TrainingJobManager" or self.job_mgr.type == "HPOJobManager":
            return urls + job_urls
        elif self.job_mgr.type == "ParallelHPOManager":
            return urls + job_urls + space_urls + node_urls
        else:
            raise ValueError("Invalid type: {}".format(self.type))


    def run_service(self, port, debug_mode=False, threaded=False, with_process=False):
        if with_process == True:
            kwargs = { 
                        'host': '0.0.0.0', 
                        'port':port, 
                        'debug' :debug_mode
                    }
                    
            self.my_process = mp.Process(target=self.app.run, kwargs=kwargs)
            self.my_process.start()
            self.my_process.join()
        else:
            if debug_mode:
                set_log_level('debug')
            self.app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=threaded) 
    
    def stop_service(self):
        if self.my_process != None:
            self.my_process.terminate()
            self.my_process.join()
            debug("API server terminated properly.")            