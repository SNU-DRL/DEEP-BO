from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import random
import copy as cp

from ws.shared.logger import *

import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import rankdata


class SequentialStrategy(object):
    def __init__(self, num_arms, values, counts, title):
        self.name = "SEQ_" + title
        self.num_arms = num_arms
        self.values = values
        self.counts = counts                
        self.epsilon = 0.0

    def next(self, step):
        idx = step % self.num_arms
        return idx

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] += 1
        self.values[arm_index] = float(self.counts[arm_index]) / float(sum(self.counts))

class SequentialKnockOutStrategy(object):
    def __init__(self, num_arms, values, counts, iters_round, title):
        self.name = "SKO_" + title

        self.num_arms = num_arms
        self.values = values
        self.counts = counts
        self.iters_round = iters_round                
        self.epsilon = 0.0

        self.remain_arms = [ i for i in range(num_arms) ]
        self.min_remains = 2 # XXX: minimum remains
        
        init_prob = 1.0 / float(num_arms)
        for i in range(len(self.values)):
            self.values[i] = init_prob
 
        self.reset()

    def reset(self):
        self.prev_arm_values = [ 0.0 for i in range(self.num_arms) ]
        self.rank_sums = np.array([ 0.0 for i in range(self.num_arms)])
        self.cur_index = 0            

    def next(self, step):
        idx = 0
        if step != 0 and step % self.iters_round == 0:
            # remove the worst performed arm            
            try:
                i_worst = np.argmax(self.rank_sums)

                if len(self.remain_arms) > self.min_remains:
                    log('arm #{} will be eliminated in {}'.format(i_worst, self.remain_arms) )
                    self.remain_arms.remove(i_worst)
                    prob = 1.0 / float(len(self.remain_arms))
                    for i in range(len(self.values)):
                        if i in self.remain_arms:
                            self.values[i] = prob
                        else:
                            self.values[i] = 0.0                    
                    
                else:
                    debug('the number of remained arms is {}.'.format(len(self.remain_arms)))
            except:
                warn("no {} in {}".format(i_worst, self.remain_arms))
            finally:
                self.reset()
        else:
            if self.cur_index < len(self.remain_arms):
                idx = self.remain_arms[self.cur_index]                
                self.cur_index += 1
            else:
                self.cur_index = 0
                idx = self.remain_arms[self.cur_index]            

        return idx

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] = self.counts[arm_index] + 1
        self.prev_arm_values[arm_index] = curr_acc

        # calculate rank
        cur_ranks = rankdata(self.prev_arm_values)
        self.rank_sums += cur_ranks



class RandomStrategy(object):
    def __init__(self, num_arms, values, counts, title):
        self.name = "RANDOM_" + title
        self.num_arms = num_arms
        self.values = values
        self.counts = counts                
        self.epsilon = 1.0

    def next(self, step):
        idx = np.random.randint(0, self.num_arms)
        
        return idx

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] += 1
        self.values[arm_index] = float(self.counts[arm_index]) / float(sum(self.counts))
                    

class ClassicHedgeStrategy(object):
    '''
    algorithm code referred from:
    https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/hedge/hedge.py
    '''
    def __init__(self, arms, temperature, values, counts, title=""):
        self.name = "HEDGE"
        if title != "":
            self.name = self.name + "_" + title
        
        self.arms = arms
        self.temperature = temperature
        self.counts = counts
        self.values = values
        self.epsilon = 0.0

    def categorical_draw(self, probs):
        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            prob = probs[i]
            cum_prob += prob
            if cum_prob > z:
                return i
        raise ValueError("unrealistic status.")

    def next(self, step):
        z = sum([math.exp(v * self.temperature) for v in self.values])
        probs = [math.exp(v * self.temperature) / z for v in self.values]
        return self.categorical_draw(probs)

    def update(self, chosen_arm, curr_acc, opt):
        reward = curr_acc # TODO:reward design required              
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1                
        value = self.values[chosen_arm]

        self.values[chosen_arm] = value + reward                


class BayesianHedgeStrategy(object):
    ''' An extension of GP-Hedge algorthm'''
    def __init__(self, arms, temperature, values, counts, 
                    s_space, choosers, 
                    title="", 
                    unbiased_estimation=False,
                    reward_scaling=None):
        self.name = "BO-HEDGE"
        if title != "":
            self.name = self.name + "_" + title
        
        self.arms = arms
        self.temperature = temperature
        self.counts = counts
        self.values = values
        self.search_space = s_space
        self.choosers = choosers
        self.epsilon = 0.0
        self.nominees = None
        self.unbiased_estimation = unbiased_estimation
        self.reward_scaling = reward_scaling

    def categorical_draw(self, probs):
        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            prob = probs[i]
            cum_prob += prob
            if cum_prob > z:
                return i
        raise ValueError("unrealistic status.")

    def nominate(self):       
        all_nominees = []
        
        for arm in self.arms:
            search_space = cp.copy(self.search_space)
            optimizer = arm['model']
            aquisition_func = arm['acq_func']
            chooser = self.choosers[optimizer]
            mean_value = 0.0 # default mean value.
            
            next_index = chooser.next(search_space, aquisition_func) 
            if chooser.mean_value is not None:
                mean_value = chooser.mean_value
            test_error = search_space.get_errors(next_index)

            all_nominees.append({
                "optimizer": optimizer,
                "aquisition_func" : aquisition_func,
                "best_index" : next_index,
                "true_acc" : 1.0 - test_error,
                "est_acc_mean" : mean_value
            })
        return all_nominees

    def next(self, step):
        self.nominees = self.nominate()
        arm_index = 0
        try:       
            z = sum([math.exp(v * self.temperature) for v in self.values])
            probs = [round(math.exp(v * self.temperature) / z, 3) for v in self.values]
            debug('probability:{}'.format(probs))
            arm_index = self.categorical_draw(probs)
        except Exception as ex:
            warn("Exception on hedge: {}".format(ex))            
            arm_index = random.randrange(len(probs))
            log("Uniform random select: {}".format(arm_index))
        
        return arm_index

    def update(self, arm_index, curr_acc, opt):
        
        self.counts[arm_index] = self.counts[arm_index] + 1 

        if self.unbiased_estimation is False:
            for n in range(len(self.nominees)):
                selected_nominee = self.nominees[n]
                est_acc = selected_nominee['est_acc_mean']
                value = self.values[n]
                self.values[n] = value + self.scale_reward(est_acc)
        else:
            for n in range(len(self.nominees)):
                selected_nominee = self.nominees[n]
                acc = selected_nominee['true_acc']
                value = self.values[n]
                self.values[n] = value + self.scale_reward(acc)            
    
    def scale_reward(self, acc):
        if self.reward_scaling is None:
            return acc
        elif self.reward_scaling == "LOG_ERR":
            
            # XXX: truncate extrapolated estimation
            if acc < 0:
                acc = 0.00001
            if acc > 1.0:
                acc = 0.99999

            # scaling log with error
            err = 1.0 - acc
            abs_log_err = math.fabs(math.log10(err))
            return abs_log_err
        else:
            raise TypeError('unsupported reward scaling:{}'.format(self.reward_scaling))
            

class EpsilonGreedyStrategy(object):
    def __init__(self, num_arms, values, counts, 
        title="", 
        init_eps=1.0,
        decay_factor=5,
        reward_scaling=None, 
        log_scale_decay=False):

        self.name = "EG"
        if reward_scaling == 'LOG_ERR':
            self.name = self.name + '_LE'

        if title != "":
            self.name = self.name + "_" + title

        self.num_arms = num_arms
        self.values = values
        self.counts = counts

        self.initial_epsilon = init_eps
        self.epsilon = init_eps
        
        self.decay_factor = decay_factor
        self.log_scale_decay = log_scale_decay
        self.reward_scaling = reward_scaling

    def next(self, step):
        ran_num = np.random.random_sample()
        idx = None

        decay_steps = self.num_arms * self.decay_factor
        # linear scale decreasing
        self.epsilon = self.initial_epsilon - (step // decay_steps) * 0.1
            
        # force to keep epsilon greater than 0.1
        if self.epsilon < 0.1:
            self.epsilon = 0.1

        max_val = np.max(self.values)
        max_idxs = np.where(self.values == max_val)[0]

        if ran_num < 1 - self.epsilon:
            idx = np.random.choice(max_idxs, 1)[0]
        else:
            if len(max_idxs) == len(self.values):
                idx = np.random.choice(max_idxs, 1)[0]
            else:
                temp = np.arange(len(self.values))
                temp = np.setdiff1d(temp, max_idxs)
                idx = np.random.choice(temp, 1)[0]
        
        idx = np.asscalar(idx)
                            
        return idx

    def update(self, arm_index, curr_acc, opt):
        acc = self.scale_acc(curr_acc)
        reward = (acc - self.values[arm_index])
        if self.counts[arm_index] > 0:  
            reward = reward / self.counts[arm_index]                
        self.counts[arm_index] += 1
        self.values[arm_index] += reward
    
    def scale_acc(self, acc):
        if self.reward_scaling is None:
            return acc
        elif self.reward_scaling == "LOG_ERR":
            
            # XXX: truncate extrapolated estimation
            if acc < 0:
                acc = 0.00001
            if acc > 1.0:
                acc = 0.99999

            # scaling log with error
            err = 1.0 - acc
            abs_log_err = math.fabs(math.log10(err))
            return abs_log_err
        else:
            raise TypeError('unsupported reward scaling:{}'.format(self.reward_scaling))


class GreedyTimeStrategy(object):
    
    def __init__(self, num_arms, values, counts, 
        title="",
        time_unit='H',
        reward_scaling=None):
        
        self.name = "GT" + time_unit

        if reward_scaling == 'LOG_ERR':
            self.name = self.name + '_LE'

        if title != "":
            self.name = self.name + "_" + title

        self.num_arms = num_arms
        self.values = values
        self.counts = counts

        self.epsilon = 1.0

        self.reward_scaling = reward_scaling

        self.time_unit = time_unit       
        self.start_time = time.time()
        self.cum_exec_time = 0

    def get_elapsed_time(self):
        if self.cum_exec_time == 0:
            elapsed = time.time() - self.start_time
        else:
            elapsed = self.cum_exec_time

        if self.time_unit == 'H':
            elapsed = math.ceil(elapsed / (60 * 60))
        elif self.time_unit == 'M':
            elapsed = math.ceil(elapsed / 60)
        elif self.time_unit == 'S':
            elapsed = math.ceil(elapsed)        
        else:
            warn('unsupported time unit: {}'.format(self.time_unit))
        return elapsed
    
    def next(self, step):
        ran_num = np.random.random_sample()
        idx = None

        # time dependent epsilon
        t = self.get_elapsed_time()         
        self.epsilon = 1 / math.sqrt(t + 1)
            
        # force to keep epsilon greater than 0.1
        if self.epsilon < 0.1:
            self.epsilon = 0.1

        max_val = np.max(self.values)
        max_idxs = np.where(self.values == max_val)[0]

        if ran_num < 1 - self.epsilon:
            idx = np.random.choice(max_idxs, 1)[0]
        else:
            if len(max_idxs) == len(self.values):
                idx = np.random.choice(max_idxs, 1)[0]
            else:
                temp = np.arange(len(self.values))
                temp = np.setdiff1d(temp, max_idxs)
                idx = np.random.choice(temp, 1)[0]
        
        idx = np.asscalar(idx)
                            
        return idx

    def update(self, arm_index, curr_acc, opt):

        if 'exec_time' in opt:            
            self.cum_exec_time += opt['exec_time']        

        acc = self.scale_acc(curr_acc)
        reward = (acc - self.values[arm_index])
        if self.counts[arm_index] > 0:  
            reward = reward / self.counts[arm_index]                
        self.counts[arm_index] += 1
        self.values[arm_index] += reward
    
    def scale_acc(self, acc):
        if self.reward_scaling is None:
            return acc
        elif self.reward_scaling == "LOG_ERR":
            
            # XXX: truncate extrapolated estimation
            if acc < 0:
                acc = 0.00001
            if acc > 1.0:
                acc = 0.99999

            # scaling log with error
            err = 1.0 - acc
            abs_log_err = math.fabs(math.log10(err))
            return abs_log_err
        else:
            raise TypeError('unsupported reward scaling:{}'.format(self.reward_scaling))
   