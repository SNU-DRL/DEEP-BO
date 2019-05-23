import os
import time
import json

from flask import jsonify, request
from flask_restful import Resource, reqparse

from ws.shared.logger import * 

class Job(Resource):
    def __init__(self, **kwargs):
        self.jm = kwargs['job_manager']
        super(Job, self).__init__()

    def get(self, job_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        args = parser.parse_args()
        if not self.jm.authorize(args['Authorization']):
            return "Unauthorized", 401

        self.jm.sync_result() # XXX: A better way may be existed
        if job_id == 'active':
            aj = self.jm.get_active_job_id()
            if aj is not None:
                return self.jm.get(aj), 200
            else:
                return {}, 204

        else:
            job = self.jm.get(job_id)
            if job is None:
                return "Job {} not found".format(job_id), 404
            else:
                return job, 200
    
    def put(self, job_id):
        parser = reqparse.RequestParser()        
        parser.add_argument("Authorization", location="headers") # for security reason
        parser.add_argument("control", location='args')
        args = parser.parse_args()

        if not self.jm.authorize(args['Authorization']):
            return "Unauthorized", 401

        job = self.jm.get(job_id)
        if job is None:
            return "Job {} not found".format(job_id), 404
        else:           
            result = self.jm.control(job_id, args["control"])            
            if result is True:
                return job, 202
            else:
                return "Invalid request:{} of {}".format(args["control"], job_id), 400
    
    def delete(self, job_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason

        args = parser.parse_args()
        if not self.jm.authorize(args['Authorization']):
            return "Unauthorized", 401

        if self.jm.remove(job_id):
            deleted_job = { "job_id": job_id }
            return deleted_job, 200
        else:
            return "Job {} can not be terminated".format(job_id), 404