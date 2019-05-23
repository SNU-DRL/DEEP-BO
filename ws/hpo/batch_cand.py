from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy

import numpy as np

from ws.shared.logger import *

class CandidateSelector(object):
    def __init__(self, model, acq_func):
        self.model = model
        self.acq_func = acq_func

    def get_by_random(self, bandit, cur_shelves, cur_time):
        cur_opt_time = 0.0
        next_index = None
        cur_samples = bandit['samples']
        while True:            
            next_index, opt_time, model, acq_func = self.get_candidate(
                bandit['machine'], self.model, self.acq_func, cur_samples)
            cur_opt_time += opt_time

            if next_index in cur_shelves:
                debug('selected #{} is already in working shelves: {}.'.format(
                    next_index, cur_shelves))
                bandit['num_duplicates'] += 1
                # XXX:random sampling the next candidate for failover
                model = 'SOBOL'
                acq_func = 'RANDOM'

            else:
                return next_index, cur_opt_time, model, acq_func

    def get_by_rank(self, bandit, cur_shelves, cur_time):
        cur_opt_time = 0.0
        next_index = None
        cur_samples = bandit['samples']
        while True:
            rankers, opt_time, model, acq_func = self.get_k_candidates(
                bandit['machine'], self.model, self.acq_func, cur_samples, k=10)
            cur_opt_time += opt_time

            for next_index in rankers:
                if not next_index in cur_shelves:
                    return next_index, cur_opt_time, model, acq_func

    def get_by_premature(self, bandit, cur_shelves, cur_time):
        cur_opt_time = 0.0
        next_index = None
        cur_samples = bandit['samples']
        # create premature results 
        pre_samples = copy.deepcopy(cur_samples)
        for w in cur_shelves:
            # update working results
            working_model = w['model_idx']
            duration = cur_time - w['start_time']
            #debug("bandit: {} - training model: {}, current duration: {:.1f}".format(bandit["m_id"], working_model, duration))
            pre_error = bandit['machine'].trainer.get_interim_error(working_model, duration)
            pre_samples.update_error(working_model, pre_error)

        next_index, opt_time, model, acq_func = self.get_candidate(
            bandit['machine'], self.model, self.acq_func, pre_samples)
        cur_opt_time += opt_time

        return next_index, cur_opt_time, model, acq_func

    def get_any(self, bandit, cur_shelves, cur_time):
        cur_opt_time = 0.0
        optimizer = bandit['machine']
        cur_samples = bandit['samples']
        next_index = None
        try:
            next_index, opt_time, _ = optimizer.select_candidate(
                self.model, self.acq_func, cur_samples)
        except KeyboardInterrupt:
            sys.exit(-1)
        except:
            warn("Exception occurred in the estimation processing. " +
                 "To avoid stopping, it selects the candidate randomly.")

            next_index, opt_time, _ = optimizer.select_candidate(
                'SOBOL', 'RANDOM', cur_samples)
        cur_opt_time += opt_time

        if next_index in cur_shelves:
            debug('selected #{} is already in working shelves: {}.'.format(
                next_index, cur_shelves))
            b['num_duplicates'] += 1

        return next_index, cur_opt_time, self.model, self.acq_func

    def get_candidate(self, optimizer, model, acq_func, samples):
        try:
            next_index, opt_time, _ = optimizer.select_candidate(
                model, acq_func, samples)

        except KeyboardInterrupt:
            sys.exit(-1)
        except:
            warn("Exception occurred in the estimation processing. " +
                 "To avoid stopping, it selects the candidate randomly.")
            next_index, opt_time, _ = optimizer.select_candidate(
                'SOBOL', 'RANDOM', samples)
            model = 'SOBOL'
            acq_func = 'RANDOM'

        return next_index, opt_time, model, acq_func

    def get_k_candidates(self, optimizer, model, acq_func, samples, k=10):
        rankers = []
        try:
            
            next_index, opt_time, _ = optimizer.select_candidate(
                model, acq_func, samples)
            rankers = [next_index]

            estimates = optimizer.bandit.choosers[model].estimates
            if estimates is not None and 'candidates' in estimates and 'acq_funcs' in estimates:
                candidates = estimates['candidates']
                acq_values = estimates['acq_funcs']
                rankers = np.asarray(candidates)[np.asarray(
                    acq_values).argsort()[:][::-1]][0:k].tolist()
                debug("top {} rankers: {}".format(k, rankers))
              
        except KeyboardInterrupt:
            sys.exit(-1)
        except:
            warn("Exception occurred in the estimation processing. " +
                 "To avoid stopping, it selects the candidate randomly.")
            model = 'SOBOL'
            acq_func = 'RANDOM'
            next_index, opt_time, _ = optimizer.select_candidate(
                model, acq_func, samples)
            rankers = [next_index]

        return rankers, opt_time, model, acq_func

    def select(self, failover):

        if failover == None or failover == "None":
            return self.get_any
        elif failover == 'random':
            return self.get_by_random
        elif failover == 'next_candidate':
            return self.get_by_rank
        elif failover == 'premature':
            return self.get_by_premature            
        else:
            debug("invalid failover method: {}. a default selection method is used.".format(failover))
        return self.get_any   
