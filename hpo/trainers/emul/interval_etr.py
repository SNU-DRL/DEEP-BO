from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from hpo.trainers.emul.trainer import EarlyTerminateTrainer
from ws.shared.logger import *


class IntervalETRTrainer(EarlyTerminateTrainer):

    def __init__(self, lookup):
        
        super(IntervalETRTrainer, self).__init__(lookup)

        self.epoch_length = lookup.num_epochs
        self.acc_min = 0.0
        self.acc_max = 0.2

    def train(self, cand_index, estimates, min_train_epoch=None, space=None):
        acc = 0 # stopping accuracy

        min_epoch = 0
        cur_max_acc = 0

        acc_curve = self.acc_curves.loc[cand_index].values
        self.history.append(acc_curve)

        debug("commencing iteration {}".format(len(self.history)))
        debug("accuracy curve: {}".format(acc_curve))
        train_epoch = len(acc_curve)
        test_error = 1.0 - max(acc_curve)
        exec_time = self.total_times[cand_index]
        etred = False
        for i in range(min_epoch, self.epoch_length-1):
            acc = acc_curve[i]
            if acc > cur_max_acc:
                cur_max_acc = acc

            #debug("current accuracy at epoch{}: {:.4f}".format(i+1, acc))                
            if self.acc_min < acc < self.acc_max:
                debug("stop at epoch{} if acc is ({},{})".format(i+1, self.acc_min, self.acc_max))
                self.early_terminated_history.append(True)
                test_error = 1.0 - cur_max_acc, 
                exec_time = self.get_train_time(cand_index, i+1)
                train_epoch = i + 1
                etred = True
                break

        return {
                "test_error": test_error , 
                "train_epoch": train_epoch,  
                "exec_time" : exec_time, 
                'early_terminated' : etred
        }    
