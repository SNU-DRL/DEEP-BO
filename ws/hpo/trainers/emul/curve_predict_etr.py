from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import math

from ws.hpo.trainers.emul.trainer import EarlyTerminateTrainer
from ws.shared.logger import *


class SurrogateLearningCurveExtrapolation(object):
    
    def __init__(self, preset, data_folder='./lookup/', **kwargs):
        self.method = 'function_mcmc_extrapolation'
        csv_path = data_folder + 'LCE_{}.csv'.format(preset)
        try:
            data = pd.read_csv(csv_path)

            self.checkpoints = data.ix[:, 1].values
            self.best_accs = data.ix[:, 2].values
            self.fatasies = data.ix[:, 3].values
            self.est_times = data.ix[:, 4].values
        except Exception as ex:
            raise ValueError("Invalid LR surrogate: {}".format(ex))
        
        self.current_best = None

    def reset(self):
        self.current_best = None

    def get_preset_checkpoint(self, index=0):
        return int(self.checkpoints[index])

    def get_checkpoint_ratio(self):
        # XXX: hard coded
        return 0.5

    def get_fatasy(self, index):
        return float(self.fatasies[index])

    def get_prediction_time(self, index):
        if np.isnan(self.est_times[index]) != True:
            return float(self.est_times[index])
        else:
            return float(np.nanmean(self.est_times)) 

    def get_best_acc(self, index):
        return float(self.best_accs[index])
    
    def set_best_acc(self, best_acc):
        if self.current_best == None or self.current_best < best_acc: 
            self.current_best = best_acc

    def check_termination(self, index, current_epoch=None):
        if current_epoch != None:
            if self.get_preset_checkpoint(index) != current_epoch:
                debug("{} is not the checkpoint.".format(current_epoch))
                return False
        f = self.get_fatasy(index) # XXX: prediction failed when it returns 0.0 
        if self.current_best == None:
            return False
        elif f == 0.0:
            return False
        elif f < self.current_best:
            return True
        else:
            return False  

class CurvePredictETRTrainer(EarlyTerminateTrainer):
    
    def __init__(self, lookup, use_fantasy=True):
        
        super(CurvePredictETRTrainer, self).__init__(lookup)
        self.preset = lookup.data_type
        self.predictor = SurrogateLearningCurveExtrapolation(self.preset)
        self.use_fantasy = use_fantasy

    def reset(self):
        self.predictor.reset()
        super(CurvePredictETRTrainer, self).reset()
        
    def train(self, cand_index, estimates, min_train_epoch=None, space=None):
        acc = 0 # stopping accuracy

        min_epoch = 0
        cur_max_acc = 0

        acc_curve = self.acc_curves.loc[cand_index].values
        self.history.append(acc_curve)

        debug("commencing iteration {}".format(len(self.history)))
        #debug("accuracy curve: {}".format(acc_curve))
        train_epoch = len(acc_curve)
        test_acc = max(acc_curve)
        real_acc = test_acc
        exec_time = self.total_times[cand_index]
        etred = False

        if self.predictor.check_termination(cand_index) == True:
            exec_time = exec_time * self.predictor.get_checkpoint_ratio() + \
                        self.predictor.get_prediction_time(cand_index)
            train_epoch = self.predictor.get_preset_checkpoint(cand_index)
            etred = True
            fantasy = self.predictor.get_fatasy(cand_index)
            real_acc = max(acc_curve[:train_epoch])
            if self.use_fantasy == True:
                test_acc = fantasy
                debug("A fantasy value {} replaces the real terminal value {}.".format(fantasy, real_acc))
            else:
                test_acc = real_acc

        self.predictor.set_best_acc(real_acc)

        return {
                "test_error": 1.0 - test_acc, 
                "train_epoch": train_epoch,  
                "exec_time" : exec_time, 
                'early_terminated' : etred,
                "test_acc" : real_acc #XXX: real accuracy if the test error becomes a fantasy 
        }


         