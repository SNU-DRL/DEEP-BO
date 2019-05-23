import os
import time
import json

from ws.shared.logger import * 

from flask import jsonify, request
from flask_restful import Resource, reqparse

class HyperparamVector(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']

        super(HyperparamVector, self).__init__()

    def get(self, space_id, sample_id):
        
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        args = parser.parse_args()
        
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        samples = self.sm.get_samples(space_id)
        if samples == None:
            return "Sampling space {} is not available".format(space_id), 500

        if sample_id == 'all':
            all_items =[]
            for c_id in range(samples.num_samples):
                grid = {"id": c_id}
                grid["hparams"] = samples.get_hpv(int(c_id))
                all_items.append(grid)
            
            return all_items, 200                
        
        elif sample_id == 'candidates':
            candidates = []
            for c_id in samples.get_candidates():
                hpv = {"id": c_id}
                hpv["hparams"] = samples.get_hpv(int(c_id))
                candidates.append(hpv)
            
            return candidates, 200

        elif sample_id == 'completes':
            completes = []
            for c_id in samples.get_completes():
                hpv = {"id": c_id}
                hpv["hparams"] = samples.get_hpv(int(c_id))
                completes.append(hpv)
            
            return completes, 200
        else:
            try:
                hpv = {"id": sample_id}
                hpv["hparams"] = samples.get_hpv(int(sample_id))
                return hpv, 200

            except Exception as ex:
                return "Getting vector failed: {}".format(ex), 404
