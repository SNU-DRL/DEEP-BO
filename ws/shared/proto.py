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
        self.headers['Authorization'] = "Basic {}".format(self.credential)


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

    def add_train_history(self, curve, train_time, cur_epoch):
        self.history.append({
            "curve": curve, 
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
        self.database = self.dbm.get_db()

    def get_credential(self):
        return self.database['credential']

    def save_db(self, key, data):
        if key in self.database: 
            self.database[key] = data
        
        if self.dbm:
            self.dbm.save(self.database)
        else:
            warn("database can not be updated because it does not loaded yet.")

    def authorize(self, auth_key):
        debug("Auth: {}".format(auth_key))
        # FIXME: remove dev auth key before it release
        if auth_key == "Basic {}".format(self.database['credential']):
            return True
        else:
            try:
                key = auth_key.replace("Basic ", "")
                u_pw = key.decode('base64')
                #debug("User:Password = {}".format(u_pw))
                if ":" in u_pw:
                    tokens = u_pw.split(":")
                    #debug("Tokens: {}".format(tokens))
                    for u in self.database['users']:
                        if tokens[0] in u and u[tokens[0]] == tokens[1]:
                            return True
                
                return False

            except Exception as ex:
                debug("Auth key decoding error: {}".format(ex))
            return False