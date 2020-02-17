from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import copy

from hpo.trainers.emul.trainer import EarlyTerminateTrainer
from ws.shared.logger import *


class ThresholdingETRTrainer(EarlyTerminateTrainer):
    
    def __init__(self, lookup, survive_ratio, 
                eval_start_ratio=0.5, eval_end_ratio=0.67):

        if survive_ratio < 0.0 or survive_ratio > 0.5:
            raise ValueError("Invalid survive_ratio: {}".format(survive_ratio))
        
        super(ThresholdingETRTrainer, self).__init__(lookup)

        self.num_epochs = lookup.num_epochs

        start_index = int(self.num_epochs * eval_start_ratio)
        if start_index > 0:
            start_index -= 1
        
        self.eval_start_index = start_index
        self.eval_end_index = int(self.num_epochs * eval_end_ratio) - 1
        self.threshold_percentile = 100.0 - (survive_ratio * 100.0) 

    def get_eval_epoch(self):
        return self.eval_end_index + 1

    def get_acc_threshold(self, cur_acc_curve):
        mean_accs = []   
        if len(cur_acc_curve) > 0:
            cur_mean_acc = np.mean(cur_acc_curve)
            if np.isnan(cur_mean_acc) == False:
                mean_accs.append(cur_mean_acc)

        for i in range(len(self.history)):
            acc_curve_span = self.history[i]["curve"][self.eval_start_index:self.eval_end_index+1]
            if len(acc_curve_span) > 0:
                mean_acc = np.mean(acc_curve_span)
                if np.isnan(mean_acc) == False:
                    mean_accs.append(mean_acc)
        
        if len(mean_accs) > 0:
            threshold = np.percentile(mean_accs, self.threshold_percentile)
        else:
            threshold = 0.0
        #debug("mean accs:{}".format(["{:.4f}".format(acc) for acc in mean_accs]))
        return threshold

    def train(self, cand_index, estimates, space=None, min_train_epoch=None, max_train_epoch=None):

        if min_train_epoch == None:
            min_train_epoch = 1

        if max_train_epoch == None:
            max_train_epoch = self.num_epochs
        else:
            self.num_epochs = max_train_epoch

        #debug("cand_index:{}".format(cand_index))

        result = self.get_preevaluated_result(cand_index)
        acc_curve = result['acc_curve']
        train_time = result['train_time']
        min_loss = result['test_error']

        debug("{}: commencing iteration {}".format(type(self).__name__, len(self.history)))
        #debug("accuracy curve: {}".format(acc_curve))

        cur_acc_curve = acc_curve
        cur_max_acc = 0
        cur_epoch = min_train_epoch
        
        early_terminated = False
        
        while cur_epoch <= max_train_epoch:

            acc = acc_curve[cur_epoch - 1]
            
            if acc > cur_max_acc:
                cur_max_acc = acc
            
            if cur_epoch == self.get_eval_epoch():
                threshold = self.get_acc_threshold(acc_curve[min_train_epoch-1:cur_epoch])

                if acc < threshold:
                    debug("{}: terminates the config#{} at epoch {} ({:.4f} > {:.4f} asymptote: {:.4f})".format(
                        type(self).__name__, cand_index, cur_epoch, threshold, acc, max(acc_curve)))
                    
                    cur_acc_curve = copy.copy(acc_curve[min_train_epoch-1:cur_epoch])
                    min_loss = 1.0 - cur_max_acc
                    train_time = self.get_train_time(cand_index, cur_epoch)
                    early_terminated = True
                    break
            
            cur_epoch += 1

        self.add_train_history(cur_acc_curve, train_time, cur_epoch, early_terminated)

        return {
                "test_error": min_loss,
                "train_epoch": len(cur_acc_curve),
                "exec_time" : train_time, 
                'early_terminated' : early_terminated
        }    


class EarlyDropETRTrainer(ThresholdingETRTrainer):
    
    def __init__(self, lookup, drop_ratio): # 0 ~ 0.5

        survive_ratio = 1.0 - drop_ratio
        super(EarlyDropETRTrainer, self).__init__(lookup, survive_ratio,
                                                    eval_start_ratio=0,
                                                    eval_end_ratio=0.5)    


class LateSurviveETRTrainer(ThresholdingETRTrainer):
    
    def __init__(self, lookup, survive_ratio):

        eval_end_ratio = 1.0 - survive_ratio
        super(LateSurviveETRTrainer, self).__init__(lookup, survive_ratio,
                                                    eval_start_ratio=0.5,
                                                    eval_end_ratio=eval_end_ratio)


class MultiThresholdingETRTrainer(EarlyTerminateTrainer):
    def __init__(self, lookup, survive_ratio):

        self.num_epochs = lookup.num_epochs
        drop_ratio = 1.0 - survive_ratio
        self.lower_etr = EarlyDropETRTrainer(lookup, drop_ratio)
        self.higher_etr = LateSurviveETRTrainer(lookup, survive_ratio)

        super(MultiThresholdingETRTrainer, self).__init__(lookup)


    def train(self, cand_index, estimates, min_train_epoch=None, space=None):
        acc = 0 # stopping accuracy

        if min_train_epoch == None:
            min_train_epoch = 1
        
        result = self.get_preevaluated_result(cand_index)

        cur_acc_curve = result['acc_curve']
        best_epoch = np.argmax(result['acc_curve']) + 1
        cur_max_acc = 0
        cur_epoch = min_train_epoch
        early_terminated = False
        threshold_epoch = int(self.num_epochs * 0.5)
        
        while cur_epoch <= self.num_epochs:
           
            if cur_epoch < threshold_epoch:
                result = self.lower_etr.train(cand_index, estimates,
                                            max_train_epoch=threshold_epoch)
                
                if result['early_terminated'] == True:
                    cur_acc_curve = cur_acc_curve[:threshold_epoch]
                    cur_epoch = threshold_epoch
                else:
                    result = self.higher_etr.train(cand_index, estimates, 
                                                   min_train_epoch=threshold_epoch)
                    if result['early_terminated'] == True:
                        cur_epoch = self.higher_etr.get_eval_epoch()
                        cur_acc_curve = copy.copy(cur_acc_curve[:cur_epoch]) 
                    else:
                       cur_epoch = self.num_epochs 
                break

        self.add_train_history(cur_acc_curve, result['exec_time'], 
                                cur_epoch, result['early_terminated'])

        return {
                "test_error": result['test_error'],
                "train_epoch": len(cur_acc_curve), 
                "exec_time" : result['exec_time'], 
                'early_terminated' : result['early_terminated']
        }    

