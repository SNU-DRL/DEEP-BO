import json
import socket
import time

try:
    from ws.rest_client.restful_lib import Connection
except ImportError:
    from ws.rest_client.request_lib import Connection
from ws.shared.logger import *
from ws.shared.proto import RemoteConnectorPrototype

class NameServerConnector(RemoteConnectorPrototype):

    def __init__(self, target_url, credential, **kwargs):
        self.outcome_domain = "gmail.com"
        return super(NameServerConnector, self).__init__(target_url, credential, **kwargs)

    def register(self, port, job_type):
        if job_type != "HPO_runner" and job_type != "ML_trainer":
            debug("Invalid job type: {}".format(job_type))
            return False
        
        ip_addr = self.get_my_ip_addr()
        register_doc = { "ip_address": ip_addr,
                        "port_num": port,
                        "job_type" : job_type
                        }
        body = json.dumps(register_doc)
        resp = self.conn.request_post("/nodes/", args={}, body=body, headers=self.headers)
        status = resp['headers']['status']
        
        if status == '201':
            js = json.loads(resp['body'])
            debug("Node {} registered properly.".format(js['node_id']))
            return js['node_id'] 
        elif status == '200':
            js = json.loads(resp['body'])
            debug("Node {} already registered.".format(js['node_id']))
            return js['node_id']             
        else:
            raise ValueError("Registeration error. status code: {}".format(status))
        
        return None               

    def get_my_ip_addr(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((self.outcome_domain, 80))
        r = s.getsockname()[0]
        s.close()
        return r
