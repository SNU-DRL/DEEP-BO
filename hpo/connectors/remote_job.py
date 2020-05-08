import sys
import json
import time
import datetime as dt

if sys.version_info[0] < 3:
    from ws.rest_client.restful_lib import Connection
else:
    from ws.rest_client.request_lib import Connection
from ws.shared.logger import *
from ws.shared.proto import RemoteConnectorPrototype


class RemoteJobConnector(RemoteConnectorPrototype):

    def __init__(self, url, credential, **kwargs):
        self.wait_time = 3

        super(RemoteJobConnector, self).__init__(url, credential, **kwargs)

    def get_profile(self):
        try:
            resp = self.conn.request_get("/", args={}, headers=self.headers)
            status = resp['headers']['status']

            if status == '200':
                profile = json.loads(resp['body'])
                return profile
            else:
                raise ValueError("Connection failed with {}".format(resp['headers']))

        except Exception as ex:
            debug("Getting profile failed: {}".format(ex))
            return None

    def get_config(self):
        try:
            resp = self.conn.request_get("/config/", args={}, headers=self.headers)
            status = resp['headers']['status']

            if status == '200':
                config = json.loads(resp['body'])
                return config 
            else:
                raise ValueError("Connection failed: {}".format(status))

        except Exception as ex:
            warn("Getting configuration failed: {}".format(ex))
            return None

    def get_all_jobs(self):
        resp = self.conn.request_get("/jobs/", args={}, headers=self.headers)
        status = resp['headers']['status']

        if status == '200':
            jobs = json.loads(resp['body'])        
            return jobs
        else:
            raise ValueError("Connection error. worker status code: {}".format(status))   
    
    def get_job(self, job_id):
        retry_count = 0
        while True:
            resp = self.conn.request_get("/jobs/{}/".format(job_id), args={}, headers=self.headers)
            status = resp['headers']['status']

            if status == '200':
                job = json.loads(resp['body'])        
                return job
            elif status == '204':
                return None # if job_id is active, no active job is available now
            elif status == '500':
                retry_count += 1
                if retry_count > self.num_retry:
                    raise ValueError("Connection error to {} job. status code: {}".format(job_id, status))
                else:
                    debug("Connection failed due to server error. retry {}/{}".format(retry_count, self.num_retry))
                    continue
            else:
                raise ValueError("Connection error to {} job. status code: {}".format(job_id, status))

    def check_active(self):
        job = self.get_job("active") # job will be None when no job is working
        if job == None:
            return False
        else:
            return True 
    def create_job(self, job_desc):

        body = json.dumps(job_desc)
        resp = self.conn.request_post("/jobs/", args={}, body=body, headers=self.headers)
        status = resp['headers']['status']
        
        if status == '201':
            js = json.loads(resp['body'])
            #debug("Job {} is created remotely.".format(js['job_id']))
            return js['job_id'] 
        else:
            raise ValueError("Job creation error. code: {}, {}".format(status, resp['body']))
        
        return None

    def start(self, job_id):
        retry_count = 0        
        try:
            while True:
                active_job = self.get_job("active")
                if active_job != None:
                    debug("Worker is busy. to start: {}, working: {}".format(job_id, active_job['job_id']))
                    time.sleep(10)
                    stopped = self.stop(active_job['job_id'])
                    retry_count += 1
                    if stopped == True:
                        continue
                    if retry_count > self.num_retry:
                        warn("Starting {} job failed.".format(job_id))
                        return False
                    else:
                        time.sleep(self.wait_time)
                        debug("Retry {}/{} after waiting {} sec".format(retry_count, self.num_retry, self.wait_time))
                        continue
                else:
                    ctrl = {"control": "start"}
                    resp = self.conn.request_put("/jobs/{}/".format(job_id), args=ctrl, headers=self.headers)
                    status = resp['headers']['status']
                    
                    if status == '202':
                        js = json.loads(resp['body'])
                        #if 'hyperparams' in js:
                        #    debug("Current training item: {}".format(js['hyperparams']))
                        #else:
                        #    debug("Current HPO item: {}".format(js))
                        return True
                    elif status == '500':
                        retry_count += 1
                        if retry_count > self.num_retry:
                            warn("Starting {} job failed.".format(job_id))
                            return False
                        else:
                            time.sleep(self.wait_time)
                            debug("Retry {}/{}...".format(retry_count, self.num_retry, self.wait_time))
                            continue                        
                    else:
                        raise ValueError("Invalid worker status: {}".format(status))                
        except Exception as ex:
            warn("Starting job {} is failed".format(job_id))
            return False

    def pause(self, job_id):
        try:
            active_job = self.get_job("active")
            if active_job is None:
                warn("Job {} can not be paused.".format(active_job))
                return False 
            else:
                ctrl = {"control": "pause"}                
                resp = self.conn.request_put("/jobs/{}/".format(job_id), args=ctrl, headers=self.headers)
                status = resp['headers']['status']
                
                if status == '202':
                    js = json.loads(resp['body'])
                    debug("paused job: {}".format(js))
                    return True
                else:
                    raise ValueError("Invalid worker status: {}".format(status))                
        except Exception as ex:
            warn("Pausing job {} is failed".format(job_id))
            return False

    def resume(self, job_id):
        try:
            active_job = self.get_job("active")
            if active_job is not None and active_job['job_id'] != job_id:
                warn("Job {} can not be resumed.".format(job_id))
                return False 
            else:
                ctrl = {"control": "resume"}
                resp = self.conn.request_put("/jobs/{}/".format(job_id), args=ctrl, headers=self.headers)
                status = resp['headers']['status']
                
                if status == '202':
                    js = json.loads(resp['body'])
                    debug("resumed job: {}".format(js))
                    return True
                else:
                    raise ValueError("Invalid worker status: {}".format(status))                
        except Exception as ex:
            warn("Resuming job {} is failed".format(job_id))
            return False

    def stop(self, job_id):
        try:
            active_job = self.get_job("active")
            if active_job is not None and active_job['job_id'] != job_id:
                warn("Job {} can not be stopped.".format(job_id))
                return False 
            else:
                resp = self.conn.request_delete("/jobs/{}/".format(active_job['job_id'] ), args={}, headers=self.headers)
                status = resp['headers']['status']
                now = dt.datetime.now()
                if status == '200':
                    js = json.loads(resp['body'])
                    debug("Stop request to job {} is accepted at {}.".format(js['job_id'], now))
                    # XXX: waiting until the job finished!
                    for i in range(20):
                        time.sleep(3)
                        job = self.get_job(js['job_id'])
                        if job['status'] == 'processing':
                            debug("Waiting until {} is to be terminated...")
                        else:
                            break
                    return True
                elif status == '404':
                    debug("No job {} found to stop at {}.".format(job_id, now))
                    return True
                else:
                    raise ValueError("Invalid worker status: {} at {}".format(status, now))                
        except Exception as ex:
            # FIXME: When stopping fails, system will be down. 
            warn("Stopping job {} is failed".format(job_id))
            time.sleep(3)
            return self.stop(job_id)        

