import json
import time
import sys
import copy

import numpy as np
import math

from ws.shared.logger import *

from hpo.connectors.remote_job import RemoteJobConnector


class RemoteTrainConnector(RemoteJobConnector):
    
    def __init__(self, url, hp_config, cred, **kwargs):

        super(RemoteTrainConnector, self).__init__(url, cred, **kwargs)
        
        self.hp_config = hp_config
 
    def validate(self):
        try:
            profile = self.get_profile()
            if profile == None:
                warn("Getting profile failed") 
                return False
            elif "spec" in profile and "node_type" in profile["spec"]:
                #debug("Remote worker profile: {}".format(profile["spec"]))
                if profile["spec"]["node_type"] == "Training Node":
                    config = self.get_config()
                    if "run_config" in config and "target_func" in config["run_config"]:                
                        if config["run_config"]["target_func"] == 'surrogate':
                            return True  # skip parameter validation process
                    if "hp_config" in config:        
                        return self.compare_config(self.hp_config.get_dict(), 
                                                config["hp_config"]) 

        except Exception as ex:
            warn("Validation failed: {}".format(ex))
            
        return False

    def compare_config(self, origin, target):
        try:
            if "hyperparams" in origin and "hyperparams" in target:
                hps = origin["hyperparams"]
                ths = target["hyperparams"]
                # XXX:Check hyperparameter name only
                for k in hps.keys():
                    if not k in ths:
                        return False
                
                return True
        except Exception as ex:
            warn("Configuration comparision failed: {}\n{}\n{}".format(ex, origin, target))

        return False

    def create_job(self, hyperparams, config=None):
        try:
            #debug("RemoteTrainConnector tries to create a training job.")
            job_desc = copy.copy(self.hp_config.get_dict())
            # update body by hyperparams
            for hp in hyperparams.keys():
                value = hyperparams[hp]
                if hp in job_desc['hyperparams']: 
                    job_desc['hyperparams'][hp] = value
                else:
                    warn("{} is not the valid parameter of the given objective function".format(hp))
                    return None
            
            if config is not None:
                for key in config.keys():
                    job_desc['config'][key] = config[key]

            return super(RemoteTrainConnector, self).create_job(job_desc)

        except Exception as ex:
            warn("Create job failed: {}".format(ex))
            return None
