import os
import time
import json

from ws.shared.logger import * 

from flask import jsonify, request
from flask_restful import Resource, reqparse

class Billboard(Resource):
    def __init__(self, **kwargs):
        self.rm = kwargs['resource_manager']
        super(Billboard, self).__init__()

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        args = parser.parse_args()
        if not self.rm.authorize(args['Authorization']):
            return "Unauthorized", 401

        return {"spec" : self.rm.get_spec(), "urls": self.rm.get_urls()}, 200 