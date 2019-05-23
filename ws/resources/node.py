import os
import time
import json

from flask import jsonify, request
from flask_restful import Resource, reqparse

from ws.shared.logger import * 

class Node(Resource):
    def __init__(self, **kwargs):
        self.nm = kwargs['node_manager']
        super(Node, self).__init__()

    def get(self, node_id):
        # TODO:add argument handling for windowing items
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        args = parser.parse_args()
        if not self.nm.authorize(args['Authorization']):
            return "Unauthorized", 401

        node = self.nm.get_node(node_id)
        if node != None:
           return node, 200
        else:
            return "Node {} not found".format(node_id), 404

    def put(self, node_id):
        parser = reqparse.RequestParser()        
        parser.add_argument("Authorization", location="headers") # for security reason
        parser.add_argument("control", location='args')
        args = parser.parse_args()

        if not self.nm.authorize(args['Authorization']):
            return "Unauthorized", 401

        node = self.nm.get_node(node_id)
        if node == None:
            return "{} node not found".format(node_id), 404    

        if "control" in args:           
            result = self.nm.control(args["control"], node_id)            
            if result is True:
                return self.nm.get_node("all"), 202
            else:
                return "Failed to control:{}".format(args["control"]), 400   

    def delete(self, node_id):
        parser = reqparse.RequestParser()        
        parser.add_argument("Authorization", location="headers") # for security reason
        args = parser.parse_args()

        if not self.nm.authorize(args['Authorization']):
            return "Unauthorized", 401

        node = self.nm.get_node(node_id)
        if node == None:
            return "{} node not found".format(node_id), 404    

        result = self.nm.control("stop", node_id)            
        if result is True:
            return job, 202
        else:
                return "Fail to stop node {}".format(node_id), 400       
