import json
import requests as req
from requests.auth import HTTPBasicAuth

class Connection:
    def __init__(self, base_url, username=None, password=None, timeout=None):
        self.base_url = base_url
        self.auth = HTTPBasicAuth(username, password)

        if timeout != None:
            self.timeout = timeout
        else:
            self.timeout = 10.0

    def request_get(self, resource, args=None, headers={}):
        url = self.base_url + resource
        if args == None:
            args = {}
        res = req.request('GET', url, params=args, headers=headers, 
                            #auth=self.auth, 
                            timeout=self.timeout)
        
        res.headers['status'] = str(res.status_code)
        resp = {'headers' : res.headers, 'body': res.content }
        return resp

    def request_post(self, resource, args=None, body=None, headers={}):
        url = self.base_url + resource
        if args == None:
            args = {}

        if body != None:
            body = json.loads(body)
        else:
            body = {}

        res = req.request('POST', url, params=args, headers=headers, json=body,
                            #auth=self.auth, 
                            timeout=self.timeout)
        res.headers['status'] = str(res.status_code)
        resp = {'headers' : res.headers, 'body': res.content }
        return resp

    def request_put(self, resource, args=None, body=None, headers={}):
        url = self.base_url + resource
        if args == None:
            args = {}

        if body != None:
            body = json.loads(body)
        else:
            body = {}

        res = req.request('PUT', url, params=args, headers=headers, json=body,
                            #auth=self.auth, 
                            timeout=self.timeout)
        res.headers['status'] = str(res.status_code)
        resp = {'headers' : res.headers, 'body': res.content }
        return resp


    def request_delete(self, resource, args=None, body=None, headers={}):
        url = self.base_url + resource
        if args == None:
            args = {}

        if body != None:
            body = json.loads(body)
        else:
            body = {}

        res = req.request('DELETE', url, params=args, headers=headers, json=body,
                            #auth=self.auth, 
                            timeout=self.timeout)
        res.headers['status'] = str(res.status_code)
        resp = {'headers' : res.headers, 'body': res.content }
        return resp
		
		
