import json
import copy
import six
import base64
import time

from ws.shared.logger import *
from ws.shared.hp_cfg import HyperparameterConfiguration
from ws.shared.proto import RemoteConnectorPrototype

        


class RemoteParameterSpaceConnector(RemoteConnectorPrototype):
    
    def __init__(self, url, credential, **kwargs):
        
        super(RemoteParameterSpaceConnector, self).__init__(url, credential, **kwargs)

        self.num_samples = None
        self.hp_config = None
        self.space_id = None
        debug("Getting parameter space status...")
        space = self.get_status()
        while space == None:
            space = self.get_status()
            time.sleep(3)

    def get_status(self):
        try:
            resp = self.conn.request_get("/", args={}, headers=self.headers)
            status = resp['headers']['status']

            if status == '200':
                space = json.loads(resp['body'])
                
                self.num_samples = space['num_samples']
                self.hp_config = HyperparameterConfiguration(space["hp_config"])
                self.space_id = space['space_id']

                return space
            else:
                raise ValueError("Connection failed with code {}".format(status))

        except Exception as ex:
            debug("Getting remote space: {}".format(ex))
            return None

    def get_space_id(self):
        if self.space_id == None:
            while self.get_status() == None:
                time.sleep(3)            
        
        return self.space_id        

    def get_num_samples(self):
        if self.num_samples == None:
            while self.get_status() == None:
                time.sleep(3)            
        
        return self.num_samples

    def get_candidates(self): 
        resp = self.conn.request_get("/candidates/", args={}, headers=self.headers)
        status = resp['headers']['status']

        if status == '200':
            result = json.loads(resp['body'])
            
            return result["candidates"]
        else:
            raise ValueError("Connection failed: {}".format(status))

    def get_completions(self): 
        resp = self.conn.request_get("/completions/", args={}, headers=self.headers)
        status = resp['headers']['status']

        if status == '200':
            result = json.loads(resp['body'])
            
            return result["completions"]
        else:
            raise ValueError("Connection failed: {}".format(status))

    def validate(self, id):
        if id == 'all':
            return True
        elif id == 'candidates':
            return True
        elif id == 'completions':
            return True
        elif id in self.get_candidates():
            return True            
        elif id in self.get_completions():
            return True
        else:
            return False

    def get_param_vectors(self, id):

        if self.validate(id) == False:
            raise ValueError("Invalid id: {}".format(id))

        resp = self.conn.request_get("/grid/{}/".format(id), args={}, headers=self.headers)
        status = resp['headers']['status']

        if status == '200':
            grid = json.loads(resp['body'])
            
            returns = []
            if type(grid) == list:
                for g in grid:
                    returns.append(g['values'])
            else:
                returns.append(grid['values'])
            #debug("grid of {}: {}".format(id, returns))
            return returns
        else:
            raise ValueError("Connection failed: {}".format(status))

    def get_hpv_dict(self, id):
        if self.validate(id) == False:
            raise ValueError("Invalid id: {}".format(id))

        resp = self.conn.request_get("/vectors/{}/".format(id), args={}, headers=self.headers)
        status = resp['headers']['status']

        if status == '200':
            vec = json.loads(resp['body'])
            
            returns = []
            if type(vec) == list:
                for v in vec:
                    returns.append(v['hparams'])
            else:
                returns = vec['hparams']
            #debug("vector of {}: {}".format(id, returns))
            return returns
        else:
            raise ValueError("Connection failed: {}".format(status))

    def get_error(self, id):
        resource = "/errors/"
        if id != 'completions': 
            if not id in self.get_completions():
                raise ValueError("Invalid id: {}".format(id))
            else:
                resource = "/errors/{}/".format(id)

        resp = self.conn.request_get(resource, args={}, headers=self.headers)
        status = resp['headers']['status']

        if status == '200':
            err = json.loads(resp['body'])
            errors = []
            orders = []
            if type(err) == list:
                for e in err:
                    errors.append(e['error'])
                    if 'order' in e:
                        orders.append(e['order'])
            else:
                errors.append(err['error'])
                if 'order' in e:
                    orders.append(e['order'])                
            #debug("error of {}: {}".format(id, returns))
            return errors, orders
        else:
            raise ValueError("Connection failed: {}".format(status))

    def update_error(self, id, error, num_epochs=None):

        if self.validate(id) == False:
            raise ValueError("Invalid id: {}".format(id))
    
        args = {
            "value": error, 
            "num_epochs": num_epochs
        }
        resp = self.conn.request_put("/errors/{}/".format(id), args=args, headers=self.headers)
        status = resp['headers']['status']
        
        if status == '202':
            err = json.loads(resp['body'])
            
            return True
        else:
            raise ValueError("Invalid space status: {}".format(status))                

    def expand(self, hpv):
    
        if self.validate(id) == False:
            raise ValueError("Invalid id: {}".format(id))
    
        args = {"value": error}
        body = json.dumps(hpv)
        resp = self.conn.request_post("/", args={}, body=body, headers=self.headers)
        status = resp['headers']['status']
        
        if status == '202':
            err = json.loads(resp['body'])
            
            return True
        else:
            raise ValueError("Invalid space status: {}".format(status))   
