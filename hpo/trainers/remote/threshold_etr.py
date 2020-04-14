import json
import time
import sys
import copy

import numpy as np
import math

from ws.shared.logger import *
from hpo.trainers.remote.trainer import EarlyTerminateTrainer


class MultiThresholdingETRTrainer(EarlyTerminateTrainer):
    
    def __init__(self, controller, space, survive_ratio, **kwargs):
        
        if survive_ratio < 0.0 or survive_ratio > 0.5:
            raise ValueError("Invalid survive_ratio: {}".format(survive_ratio))

        super(MultiThresholdingETRTrainer, self).__init__(controller, space, **kwargs)
        
        self.survive_ratio = survive_ratio
        self.early_drop_percentile = (survive_ratio * 100.0)
        self.late_drop_percentile = 100 - (survive_ratio * 100.0)
        self.num_epochs = self.max_train_epoch
                        
    def get_eval_indices(self, eval_start_ratio, eval_end_ratio):
        start_index = int(self.num_epochs * eval_start_ratio)
        if start_index > 0:
            start_index -= 1
        
        eval_start_index = start_index
        eval_end_index = int(self.num_epochs * eval_end_ratio) - 1
        return eval_start_index, eval_end_index

    def get_acc_threshold(self, cur_acc_curve, 
                        eval_start_index, eval_end_index, percentile):
        mean_accs = []
        cur_acc_curve = cur_acc_curve[eval_start_index:eval_end_index+1]   
        if len(cur_acc_curve) > 0:
            cur_mean_acc = np.mean(cur_acc_curve)
            if np.isnan(cur_mean_acc) == False:
                mean_accs.append(cur_mean_acc)

        for i in range(len(self.history)):
            acc_curve_span = []
            prev_curve = self.history[i]["curve"]
            
            if len(prev_curve) > eval_end_index:
                acc_curve_span = prev_curve[eval_start_index:eval_end_index+1]
            
            if len(acc_curve_span) > 0:
                mean_acc = np.mean(acc_curve_span)
                if np.isnan(mean_acc) == False:
                    mean_accs.append(mean_acc)
        
        if len(mean_accs) > 0:
            threshold = np.percentile(mean_accs, percentile)
        else:
            threshold = 0.0

        #debug("P:{}%, T:{:.4f}, mean accs:{}".format(percentile, threshold, ["{:.4f}".format(acc) for acc in mean_accs]))
        return threshold

    def stop_check(self, acc_curve, estimates):
        cur_epoch = len(acc_curve)
        
        early_drop_epoch = int(self.num_epochs * 0.5)
        survive_check_epoch = int(self.num_epochs * (1.0 - self.survive_ratio))

        #debug("Current epoch: {}, checkpoints: {}".format(cur_epoch, [early_drop_epoch, survive_check_epoch]))
        
        if self.etr_checked == False:
            if cur_epoch >= early_drop_epoch and cur_epoch < survive_check_epoch:
                # evaluate early termination criteria
                start_index, end_index = self.get_eval_indices(0.0, 0.5)
                cur_acc = acc_curve[end_index]
                
                acc_thres = self.get_acc_threshold(acc_curve, start_index, end_index, self.early_drop_percentile)
                debug("Termination check at {} epoch: {:.4f}".format(cur_epoch, cur_acc / acc_thres))
                if cur_acc < acc_thres:
                    cur_err = 1.0 - cur_acc
                    debug("Early dropped as {}".format(cur_err))  # XXX:change to error                  
                    return True
                else:
                    self.etr_checked = "early"
        elif self.etr_checked == "early":
            if cur_epoch >= survive_check_epoch:
                # evaluate late survival criteria
                eval_end_ratio = 1.0 - self.survive_ratio
                start_index, end_index = self.get_eval_indices(0.5, eval_end_ratio)
                cur_acc = acc_curve[end_index]
                acc_thres = self.get_acc_threshold(acc_curve, start_index, end_index, self.late_drop_percentile)
                debug("Termination check at {} epoch: {:.4f}".format(cur_epoch, cur_acc / acc_thres))
                if cur_acc < acc_thres:
                    cur_err = 1.0 - cur_acc
                    debug("Late dropped as {}".format(cur_err)) # XXX:change to error
                    return True
                else:
                    self.etr_checked = True
                    return False

        else:
            return False
