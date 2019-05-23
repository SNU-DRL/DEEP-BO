import os
import time
import json

from ws.shared.logger import * 

from flask import jsonify, request
from flask_restful import Resource, reqparse

class Space(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']
        
        super(Space, self).__init__()

    def get(self, space_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        args = parser.parse_args()
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        samples = self.sm.get_samples(space_id)
        if samples == None:
            return "{} space is not available".format(space_id), 500

        space = {}
        if hasattr(samples, 'name'):
            space["space_id"] = samples.get_name()            
        space["num_samples"] = samples.num_samples
        space["hp_config"] = samples.get_hp_config().get_dict()

        return space, 200

    def post(self, space_id):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument("Authorization", location="headers") # for security reason
            args = parser.parse_args()
            if not self.sm.authorize(args['Authorization']):
                return "Unauthorized", 401

            samples = self.sm.get_samples(space_id)
            if samples == None:
                return "{} space is not available".format(space_id), 500            
            
            expand_req = request.get_json(force=True)
            # TODO: validate space_req is valid
            
            samples.expand(expand_req)
            return {"space_id": space_id}, 201
        except Exception as ex:
            return "Sampling space expand failed: {}".format(ex), 400        



    def put(self, space_id):
        parser = reqparse.RequestParser()        
        parser.add_argument("Authorization", location="headers") # for security reason
        parser.add_argument("status", location='args')
        args = parser.parse_args()

        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        samples = self.sm.get_samples(space_id)
        if samples is None:
            return "Space {} not found".format(space_id), 404
        else:
            if "status" in args:           
                result = self.sm.set_space_status(space_id, args["status"])            
                if result is True:
                    return samples["status"], 202
                else:
                    return "Invalid request:{} of {}".format(args["status"], space_id), 400
            else:
                return "Invalid request:{} of {}".format(args, space_id), 400
    
    def delete(self, space_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason

        args = parser.parse_args()
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        if self.sm.set_space_status(space_id, "finished"):
            deleted_job = { "id": space_id }
            return deleted_job, 200
        else:
            return "{} space can not be eliminated".format(space_id), 404
 