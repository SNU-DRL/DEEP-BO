import os
import time
import json

from flask import jsonify, request
from flask_restful import Resource, reqparse

from ws.shared.logger import * 

class Nodes(Resource):
    def __init__(self, **kwargs):
        self.nm = kwargs['node_manager']
        super(Nodes, self).__init__()

    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument("Authorization", location="headers") # for security reason
            args = parser.parse_args()
            if not self.nm.authorize(args['Authorization']):
                return "Unauthorized", 401
            
            node_req = request.get_json(force=True)
            # TODO:validate node_req
            node_id, code = self.nm.register(node_req)
            if node_id is None:
                return "Invalid node creation request: {}".format(node_req), 400
            else:                
                return {"node_id": node_id}, code

        except Exception as ex:
            return "HPO node creation failed: {}".format(ex), 400

    def get(self):
        # TODO:add argument handling for windowing items
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        args = parser.parse_args()
        if not self.nm.authorize(args['Authorization']):
            return "Unauthorized", 401
    
        return self.nm.get_node("all"), 200

    # def put(self):
    #     parser = reqparse.RequestParser()        
    #     parser.add_argument("Authorization", location="headers") # for security reason
    #     parser.add_argument("control", location='args')
    #     parser.add_argument("exp_time", location='args')
    #     parser.add_argument("space_id", location='args')
    #     args = parser.parse_args()
    #     space_id = None
    #     if not self.nm.authorize(args['Authorization']):
    #         return "Unauthorized", 401

    #     if "control" in args:
    #         if args["control"] == "start":
    #             if "space_id" in args and self.nm.validate_space(args["space_id"]):
    #                 space_id = args["space_id"]
    #             else:
    #                 space_id = self.nm.create_new_space()

    #         result = self.nm.control(args["control"], space_id, args["exp_time"])            
    #         if result is True:
    #             return self.nm.get_pairs(), 202
    #         else:
    #             return "Failed to control:{}".format(args["control"]), 400   

    # def delete(self):
    #     parser = reqparse.RequestParser()        
    #     parser.add_argument("Authorization", location="headers") # for security reason
    #     args = parser.parse_args()

    #     if not self.nm.authorize(args['Authorization']):
    #         return "Unauthorized", 401
           
    #     result = self.nm.control("stop")            
    #     if result is True:
    #         return job, 202
    #     else:
    #             return "Fail to stop all nodes", 400   

