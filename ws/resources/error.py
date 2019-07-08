import os
import time
import json

from flask import jsonify, request
from flask_restful import Resource, reqparse

from ws.shared.logger import * 


class ObservedError(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']

        super(ObservedError, self).__init__()

    def get(self, space_id, sample_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        
        args = parser.parse_args()
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401
        
        samples = self.sm.get_samples(space_id)
        if samples == None:
            return "Sampling space {} is not available".format(space_id), 404
        
        sample_id = int(sample_id)
        error = {"id": sample_id}
        error["error"] = samples.get_errors(sample_id)
        error["order"] = samples.get_search_order(sample_id)
        error['num_epochs'] = samples.get_train_epoch(sample_id)

        return error, 200 
    
    def put(self, space_id, sample_id):
        parser = reqparse.RequestParser()        
        parser.add_argument("Authorization", location="headers") # for security reason
        parser.add_argument("value", location='args', type=float)
        parser.add_argument("num_epochs", location='args', type=int, default=1)
        args = parser.parse_args()

        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        samples = self.sm.get_samples(space_id)
        if samples is None:
            return "Space {} not found".format(space_id), 404
        else:
            try:
                if space_id != "active":
                    self.sm.set_space_status(space_id, "active")
                sample_id = int(sample_id)
                samples.update_error(sample_id, args["value"], args["num_epochs"])
                error = {"id": sample_id}
                error["error"] = samples.get_errors(sample_id)
                error["num_epochs"] = args["num_epochs"]
                
                return error, 202

            except Exception as ex:
                warn("Error update exception: {}".format(ex))
                return "Invalid request:{}".format(ex), 400         

