import os
import time
import json

from ws.shared.logger import * 

from flask import jsonify, request
from flask_restful import Resource, reqparse

class Grid(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']
        super(Grid, self).__init__()

    def get(self, space_id, sample_id):
        
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        parser.add_argument("use_interim", type=bool, default=False)
        args = parser.parse_args()
        
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        samples = self.sm.get_samples(space_id)
        if samples == None:
            return "Sampling space {} is not available".format(space_id), 500

        if id == 'all':
            all_items =[]
            for c_id in range(samples.num_samples):
                grid = {"id": int(c_id)}
                grid["values"] = []
                for v in samples.get_grid(int(c_id)).tolist():
                    grid["values"].append(float(v))
                all_items.append(grid)
            
            return all_items, 200                
        
        elif sample_id == 'candidates':
            candidates = []
            for c_id in samples.get_candidates(args['use_interim']):
                grid = {"id": int(c_id)}
                grid["values"] = []
                for v in samples.get_grid(int(c_id)).tolist():
                    grid["values"].append(float(v))
                candidates.append(grid)
            
            return candidates, 200

        elif sample_id == 'completes':
            completes = []
            for c_id in samples.get_completes(args['use_interim']):
                grid = {"id": int(c_id)}
                grid["values"] = []
                for v in samples.get_grid(int(c_id)).tolist():
                    grid["values"].append(float(v))
                completes.append(grid)
            
            return completes, 200
        else:
            try:
                grid = {"id": int(sample_id)}
                grid["values"] = []
                for v in samples.get_grid(int(c_id)).tolist():
                    grid["values"].append(float(v))
                return grid, 200

            except Exception as ex:
                return "Getting grid failed: {}".format(ex), 404
