import os
import time
import json

from flask import jsonify, request
from flask_restful import Resource, reqparse

from ws.shared.logger import * 

class Spaces(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']
        super(Spaces, self).__init__()

    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument("Authorization", location="headers") # for security reason
            args = parser.parse_args()
            if not self.sm.authorize(args['Authorization']):
                return "Unauthorized", 401
            
            space_req = request.get_json(force=True)
            # TODO:check whether 'surrogate', 'hp_cfg' existed
            debug("Search space creation request accepted.")  
            space_id = self.sm.create(space_req) 

            if space_id is None:
                return "Invalid parameter space creation request: {}".format(space_req), 400
            else:                
                return {"space_id": space_id}, 201

        except Exception as ex:
            return "Search space creation failed: {}".format(ex), 400

    def get(self):
        # TODO:add argument handling for windowing items
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        args = parser.parse_args()
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        return self.sm.get_available_spaces(), 200