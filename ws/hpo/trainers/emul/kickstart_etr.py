from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import compress

import numpy as np

from ws.hpo.trainers.emul.trainer import EarlyTerminateTrainer
from ws.shared.logger import *


class KickStarterETRTrainer(EarlyTerminateTrainer): 

    def __init__(self, lookup, expired_time=None):

        super(KickStarterETRTrainer, self).__init__(lookup)
        
        self.epoch_length = lookup.num_epochs
        
        self.eval_epoch = int(self.epoch_length/5)
        self.satisfy_epochs = int(self.epoch_length/2*self.eval_epoch)
        self.percentile = 50
        self.acc_min = 0.0
        self.acc_max = 1.5
        if expired_time == None:
            expired_time = 1800.0   # XXX:default 30min.
        self.expired_time = expired_time

    def train(self, cand_index, estimates, min_train_epoch=None, space=None):
        acc = 0 # stopping accuracy
        knocked_in_count = 0
        min_epoch = 0
        cur_max_acc = 0

        acc_curve = self.acc_curves.loc[cand_index].values

        train_time = self.total_times[cand_index]
        lc = {"curve": acc_curve, "train_time":train_time}
        self.history.append(lc)
        history = []
        knock_in_barriers = [0] * self.eval_epoch
        early_terminated = False
        for i in range(self.eval_epoch):
            history.append([])
            for n in range(len(self.history)): # number of iterations
                history[i].append(self.history[n]["curve"][i]) # vertical congregation of curve values     
            knock_in_barriers[i] = np.percentile(history[i], self.threshold_percentile) 

        debug("commencing iteration {}".format(len(self.history)))
        debug("accuracy curve: {}".format(acc_curve))
        
        min_loss = 1.0 - max(acc_curve)
        train_epoch = len(acc_curve)
        for i in range(min_epoch, self.epoch_length-1):
            acc = acc_curve[i]
            if acc > cur_max_acc:
                cur_max_acc = acc
            
            #debug("current accuracy at epoch{}: {:.4f}".format(i+1, acc))
            cum_train_time = sum([lc["train_time"] for lc in self.history])

            if cum_train_time <= self.expired_time: #1800 seconds == 30 minutes
                if len(self.history) > int(round(1/(1-self.threshold_percentile/100))): # fully train a few trials for intial parameter setting
                    if 0 <= i:
                        if self.acc_min < acc < self.acc_max:
                            debug("stopped at epoch{} locked between ({},{})".format(i+1, self.acc_min, self.acc_max))
                            self.early_terminated_history.append(True)
                            min_loss = 1.0 - cur_max_acc
                            train_time = self.get_train_time(cand_index, i+1)
                            train_epoch = i + 1
                            early_terminated = True
                            break 

                    if i <= self.eval_epoch-1:
                        if acc > knock_in_barriers[i]:
                            debug("acc knocked into above {} at epoch{}".format(knock_in_barriers[i],i+1))
                            knocked_in_count += 1

                    if i == self.eval_epoch-1:
                        if knocked_in_count <= self.satisfy_epochs:
                            debug("terminated at epoch{} with {} less knock_ins".format(i+1, self.satisfy_epochs - knocked_in_count))
                            # stop early
                            self.early_terminated_history.append(True)
                            min_loss = 1.0 - cur_max_acc
                            train_time = self.get_train_time(cand_index, i+1)
                            early_terminated = True
                            train_epoch = i + 1
                            break
        
        return {
                "test_error": min_loss,
                "train_epoch": train_epoch,   
                "exec_time" : train_time, 
                'early_terminated' : early_terminated
        }    
