import base64
try:
    from ws.rest_client.restful_lib import Connection
except ImportError:
    from ws.rest_client.request_lib import Connection
from ws.shared.db_mgr import get_database_manager 

from ws.shared.logger import * 
from ws.shared.resp_shape import *


class RemoteConnectorPrototype(object):
    def __init__(self, target_url, credential, **kwargs):
        self.url = target_url
        self.credential = credential
        
        if "timeout" in kwargs:
            self.timeout = kwargs['timeout']
        else:
            self.timeout = 10

        if "num_retry" in kwargs:
            self.num_retry = kwargs['num_retry']
        else:
            self.num_retry = 100

        self.conn = Connection(target_url, timeout=self.timeout)
        
        self.headers = {'Content-Type':'application/json', 'Accept':'application/json'}
        auth_key = base64.b64encode(self.credential.encode('utf-8'))
        auth_key = "Basic {}".format(auth_key.decode("utf-8"))
        #debug("Auth key to request: {}".format(auth_key))
        self.headers['Authorization'] = auth_key


class TrainerPrototype(object):

    def __init__(self, *args, **kwargs):
        self.shaping_func = apply_no_shaping
        self.history = []

    def set_response_shaping(self, shape_func_type):
        if shape_func_type == "hybrid_log":
            self.shaping_func = apply_hybrid_log
        elif shape_func_type == "log_err":
            self.shaping_func = apply_log_err
        else:
            self.shaping_func = apply_no_shaping

    def reset(self):
        self.history = []

    def add_train_history(self, curve, train_time, cur_epoch, measure='accuracy'):
        self.history.append({
            "curve": curve, 
            "measure" : measure, 
            "train_time": train_time, 
            "train_epoch": cur_epoch
        })

    def train(self, cand_index, estimates=None, min_train_epoch=None, space=None):
        raise NotImplementedError("This should return loss and duration.")

    def get_interim_error(self, model_index, cur_dur):
        raise NotImplementedError("This should return interim loss.")


class ManagerPrototype(object):

    def __init__(self, mgr_type):
        self.type = mgr_type
        self.dbm = get_database_manager()

    def get_credential(self):
        # XXX:access DB at request
        database = self.dbm.get_db()        
        return database['credential']

    def get_train_jobs(self):
        # XXX:access DB at request
        database = self.dbm.get_db()
        if 'train_jobs' in database:       
            return database['train_jobs']
        else:
            return []        

    def get_hpo_jobs(self):
        # XXX:access DB at request
        database = self.dbm.get_db()
        if 'hpo_jobs' in database:       
            return database['hpo_jobs']
        else:
            return []  

    def get_users(self):
        # XXX:access DB at request
        database = self.dbm.get_db()
        if 'users' in database:        
            return database['users']
        else:
            return []

    def save_db(self, key, data):
        # XXX:access DB at request
        database = self.dbm.get_db()            
        database[key] = data
        self.dbm.save(database)

    def authorize(self, auth_key):
        # XXX:Use of basic auth as default
        #debug(auth_key)
        key = auth_key.replace("Basic ", "")
        try:
            u_pw = base64.b64decode(key).decode('utf-8')
            #debug("User:Password = {}".format(u_pw))
            if ":" in u_pw:
                tokens = u_pw.split(":")
                #debug("Tokens: {}".format(tokens))
                for u in self.get_users():
                    if tokens[0] in u and u[tokens[0]] == tokens[1]:
                        return True
            elif u_pw == self.get_credential():
                #debug("Use of global auth key: {}".format(u_pw))
                # FIXME:global password for debug mode 
                return True
            else:
                return False

        except Exception as ex:
            debug("Auth key {} decoding error: {}".format(key, ex))
        return False