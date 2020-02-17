import os
import time
import json

from ws.shared.logger import * 

from flask import jsonify, request
from flask_restful import Resource, reqparse

class Completions(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']
        
        super(Completions, self).__init__()

    def get(self, space_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        parser.add_argument("use_interim", type=bool, default=False)
        args = parser.parse_args()
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        samples = self.sm.get_samples(space_id)
        if samples == None:
            return "Search space {} is not available".format(space_id), 500

        result = {}
        
        result["completions"] = samples.get_completions(args['use_interim']).tolist()

        return result, 200 