from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import compress

import numpy as np

from hpo.trainers.emul.trainer import EarlyTerminateTrainer
from ws.shared.logger import *


class IntervalKnockETRTrainer(EarlyTerminateTrainer): # 

    def __init__(self, lookup):

        super(IntervalKnockETRTrainer, self).__init__(lookup)
        
        self.epoch_length = lookup.num_epochs
        
        self.eval_epoch = int(self.epoch_length/4)
        self.satisfy_epochs = int(self.epoch_length/self.eval_epoch*3)
        self.percentile = 75 # percentile X 100
        self.acc_min = 0.0
        self.acc_max = 0.2

    def train(self, cand_index, estimates, min_train_epoch=None, space=None):
        acc = 0 # stopping accuracy
        knocked_in_count = 0
        min_epoch = 0
        cur_max_acc = 0
        knock_out_barrier = 0

        acc_curve = self.acc_curves.loc[cand_index].values
        self.history.append(acc_curve)

        history = []
        knock_temp_storage = []
        knock_in_barriers = [0] * self.eval_epoch #7
        unstopped_list = list(compress(self.history, self.early_terminated_history))
        knock_out_candidates = []       


        for i in range(self.eval_epoch): #7
            history.append([])
            for n in range(len(self.history)): # number of iterations
                history[i].append(self.history[n][i]) # vertical congregation of curve values     
            knock_in_barriers[i] = np.percentile(history[i], self.threshold_percentile) # 75% value of the vertical row
        
        for i in range(len(unstopped_list)): # number of iterations fully trained without ETR activated
            knock_temp_storage.append([1])
            for n in range(self.eval_epoch, self.epoch_length-1): # epoch 8 ~ 15
                knock_temp_storage[i].append(unstopped_list[i][n]) 
            knock_out_candidates.append(min(knock_temp_storage[i]))


        if len(knock_out_candidates) >= 1:
            if len(self.history) > int(round(1/(1-self.threshold_percentile/100))):

                knock_out_point = min(knock_out_candidates)
                knock_out_adjusted_margin = max(0,(0.05 - 0.001 * (len(self.history)-int(round(1/(1-self.threshold_percentile/100))))))
                knock_out_barrier = knock_out_point - knock_out_adjusted_margin
            else:
                knock_out_barrier = np.max(knock_out_candidates)
        else: 
            knock_out_barrier = knock_in_barriers[self.eval_epoch-1]

        debug("commencing iteration {}".format(len(self.history)))
        debug("accuracy curve: {}".format(acc_curve))

        for i in range(min_epoch, self.epoch_length-1):
            acc = acc_curve[i]
            if acc > cur_max_acc:
                cur_max_acc = acc
            
            #debug("current accuracy at epoch{}: {:.4f}".format(i+1, acc))

            if len(self.history) > int(round(1/(1-self.threshold_percentile/100))): # fully train a few trials for intial parameter setting
                if i >= 1:
                    if self.acc_min < acc < self.acc_max:
                        debug("stopped at epoch{} locked between ({},{})".format(i+1, self.acc_min, self.acc_max))
                        self.early_terminated_history.append(True)
                        return 1.0 - cur_max_acc, self.get_train_time(cand_index, i+1), True
                if i <= self.eval_epoch-1:
                    if acc > knock_in_barriers[i]:
                        debug("acc knocked into above {} at epoch{}".format(knock_in_barriers[i],i+1))
                        knocked_in_count += 1

                if i == self.eval_epoch-1:
                    if knocked_in_count <= self.satisfy_epochs:
                        debug("terminated at epoch{} with {} less knock_ins".format(i+1, self.satisfy_epochs - knocked_in_count))
                        # stop early
                        self.early_terminated_history.append(True)
                        return 1.0 - cur_max_acc, self.get_train_time(cand_index, i+1), True

                if self.epoch_length-1 > i > self.eval_epoch-1:
                    if knocked_in_count > self.satisfy_epochs:
                        if acc < knock_out_barrier:
                            #stop early
                            self.early_terminated_history.append(True)
                            debug("terminated at epoch{} by knocking out below {}".format(i+1, knock_out_barrier))
                            return 1.0 - cur_max_acc, self.get_train_time(cand_index, i+1), True
        return {
                "test_error":  1.0 - max(acc_curve), 
                "exec_time" : self.total_times[cand_index], 
                'early_terminated' : True
        }    

class HybridETRTrainer(EarlyTerminateTrainer):
    
    def __init__(self, lookup, 
                percentile=80, eval_start=0.5, eval_end=0.85, interval_bound_acc=0.2):
        
        super(HybridETRTrainer, self).__init__(lookup)

        self.epoch_length = lookup.num_epochs
        self.lcs = self.history
        self.eval_epoch_start = int(self.epoch_length * eval_start)
        self.eval_epoch_end = int(self.epoch_length * eval_end)
        self.percentile = percentile
        self.acc_min = 0.0
        self.acc_max = interval_bound_acc

    def train(self, cand_index, estimates, min_train_epoch=None, space=None):
        
        if min_train_epoch == None:
            min_epoch = 0
        cur_max_acc = 0

        acc_curve = self.acc_curves.loc[cand_index].values
        train_time = self.total_times[cand_index]
        min_loss = 1.0 - max(acc_curve)
        lc = {"curve": acc_curve, "train_time":train_time}
        self.history.append(lc)

        history = []   
        early_terminated = False
        for i in range(len(self.history)):
            history.append(np.mean(self.history[i]["curve"][self.eval_epoch_start:self.eval_epoch_end+1]))

        threshold = np.percentile(history, self.threshold_percentile)
        train_time = self.total_times[cand_index]

        debug("commencing iteration {}".format(len(self.history)))
        #debug("accuracy curve: {}".format(acc_curve))

        for i in range(min_epoch, self.epoch_length-1):
            acc = acc_curve[i]
            if acc > cur_max_acc:
                cur_max_acc = acc

            #debug("current accuracy at epoch{}: {:.4f}".format(i+1, acc))                
            if i >= self.eval_epoch_start and self.acc_min < acc < self.acc_max:
                debug("stop at epoch{} if acc is ({},{})".format(i+1, self.acc_min, self.acc_max))
                early_terminated = True
                min_loss = 1.0 - cur_max_acc
                train_time = self.get_train_time(cand_index, i+1)
                break
            else:
                cum_train_time = sum([lc["train_time"] for lc in self.history])

                if i+1 == self.eval_epoch_end:
                    if acc < threshold:
                        debug("terminated at epoch{}".format(i+1))
                        early_terminated = True
                        
                        min_loss = 1.0 - cur_max_acc
                        train_time = self.get_train_time(cand_index, i+1)
                        break                    
        self.early_terminated_history.append(early_terminated)
        return {
                "test_error":  min_loss,
                "train_epoch": i + 1, 
                "exec_time" : train_time, 
                'early_terminated' : early_terminated
        }    