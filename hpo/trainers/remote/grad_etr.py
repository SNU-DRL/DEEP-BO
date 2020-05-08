import json
import time
import sys
import copy

import numpy as np
import math

from ws.shared.logger import *
from hpo.trainers.remote.trainer import EarlyTerminateTrainer


class GradientETRTrainer(EarlyTerminateTrainer):
    
    def __init__(self, controller, space):
        
        super(GradientETRTrainer, self).__init__(controller, space)
        self.estimates = None

    def get_gradient_average(self, acc_curve, num_step):

        acc_delta = []
        num_grads = len(acc_curve) - 1
        
        if num_grads < 2:
            return acc_curve[0]
        
        for j in range(num_grads):
            delta = acc_curve[j+1] - acc_curve[j]
            acc_delta.append(delta)
        avg_deltas =  sum(acc_delta) / num_grads
        #debug("delta average: {:.5f}, delta list: {}".format(avg_deltas, [round(d, 5) for d in acc_delta]))         
        return avg_deltas

    def stop_check(self, acc_curve):
        if self.estimates is None:
            self.early_terminated_history.append(False)
            return False
        else:
            candidates = self.estimates['candidates']
            acq_funcs = self.estimates['acq_funcs']
            means = self.estimates['means']
            vars = self.estimates['vars']        
        
            i_max = np.argmax(acq_funcs)
            v_max = acq_funcs[i_max]
            #print("max {} ({})".format(v_max, i_max))
            m = means[i_max]
            s = math.sqrt(vars[i_max])
        
            est_acc_mean = 1 - self.shaping_func(m)
            #debug("estimated mean: {:.4f}, stdev: {:.4f}".format(est_acc_mean, s))        
            acc_delta = 0
            cur_max_acc = 0
            
            max_epoch = 100
            try:
                max_epoch = self.hp_config.config.max_epoch
            except Exception as ex:
                pass

            for i in range(len(acc_curve)):
                acc = acc_curve[i]
                obs_n = i - 1
                if acc > cur_max_acc:
                    cur_max_acc = acc

                reduced_s = (s - s / max_epoch * i)
                lower_bound = est_acc_mean # - self.coefficient * reduced_s
                # dynamic stdev decrease
                
                min_delta = (est_acc_mean - acc) / (max_epoch - i)
                #debug("current accuracy: {:.4f}, lower bound: {:.4f}".format(acc, lower_bound))
                #debug("est. mean acc: {:.4f}, min delta: {:.4f}".format(est_acc_mean - acc, min_delta))
                if acc < lower_bound and min_delta > self.get_gradient_average(acc_curve, i):
                    debug("Early stopped curve: {}".format( [round(acc, 5) for acc in acc_curve]))
                    self.early_terminated_history.append(True)
                    return True
        
        self.early_terminated_history.append(False)
        return False                     

