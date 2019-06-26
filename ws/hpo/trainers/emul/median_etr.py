from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from ws.hpo.trainers.emul.trainer import EarlyTerminateTrainer
from ws.shared.logger import *


class VizMedianETRTrainer(EarlyTerminateTrainer): #

    def __init__(self, lookup):
        
        super(VizMedianETRTrainer, self).__init__(lookup)

        self.epoch_length = lookup.num_epochs        
        self.eval_epoch = int(self.epoch_length/3)
        self.threshold_percentile = 50 # median

    def train(self, cand_index, estimates, min_train_epoch=None, space=None):
        acc = 0 # stopping accuracy
        min_epoch = 0
        cur_max_acc = 0
        debug("cand_index:{}".format(cand_index))
        acc_curve = self.acc_curves.loc[cand_index].values

        history = []   

        for i in range(len(self.history)):
            curve = self.history[i]["curve"]
            history.append(np.mean(curve[:self.eval_epoch]))
        if len(history) > 0:
            threshold = np.percentile(history, self.threshold_percentile)
        else:
            threshold = 0.0

        debug("commencing iteration {}".format(len(self.history)))
        #debug("accuracy curve: {}".format(acc_curve))
        test_error = 1.0 - max(acc_curve)
        train_epoch = len(acc_curve)
        exec_time = self.total_times[cand_index]
        early_terminated = False
        for i in range(min_epoch, self.epoch_length-1):
            acc = acc_curve[i]
            if acc > cur_max_acc:
                cur_max_acc = acc
            
            #debug("current accuracy at epoch{}: {:.4f}".format(i+1, acc))

            if i+1 == self.eval_epoch:
                if acc < threshold:
                    debug("terminated at epoch{}".format(i+1))
                    train_epoch = self.eval_epoch
                    acc_curve = acc_curve[:train_epoch]
                    early_terminated = True
                    exec_time = self.get_train_time(cand_index, i+1)
                    break

        self.add_train_history(acc_curve, exec_time, 
                               train_epoch, early_terminated)
        return {
                "test_error":  test_error, 
                "train_epoch": train_epoch,
                "exec_time" : exec_time, 
                'early_terminated' : early_terminated
        }    


