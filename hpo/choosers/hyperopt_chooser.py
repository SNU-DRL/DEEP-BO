from __future__ import division
from __future__ import print_function

import copy
import math

import pandas as pd
import numpy as np


from hyperopt import hp, fmin, tpe, base, rand, Trials, STATUS_OK, JOB_STATE_DONE

import ws.shared.hp_cfg as hp_cfg
from ws.shared.logger import *
from ws.shared.resp_shape import *
from ws.shared.hp_cfg import HyperparameterConfiguration

from hpo.choosers.util import *

def init(samples, arg_string):
    args = unpack_args(arg_string)
    ssc = HyperOptSearchSpaceConfig(samples.get_hp_config())    
    return HyperOptChooser(ssc, **args)


class HyperOptChooser(object):

    def __init__(self, space_cfg,
                 response_shaping=False,
                 shaping_func="log_err"):
        self.space_cfg = space_cfg
        self.hyperparams = space_cfg.hyperparams
        self.hyperopt_space = space_cfg.get_hyperopt_space()
        self.acq_funcs = ['EI', 'RANDOM']
        self.mean_value = None
        self.estimates = None
        self.last_params = None
        self.response_shaping = bool(response_shaping)
        self.shaping_func = shaping_func

    def set_eval_time_penalty(self, est_eval_time):
        # TODO:We can not apply the evaluation time penalty now. 
        pass

    def next(self, samples, acq_func, use_interim=True):
        algo = tpe.suggest
        if acq_func == 'RANDOM':
            algo = rand.suggest
        elif acq_func != 'EI':
            debug("Unsupported acquisition function: {}".format(acq_func))
        candidates = samples.get_candidates(use_interim) 
        errs = samples.get_errors("completions")
        if len(errs) == 0:
            return candidates[0] # return the first candidate 

        comp = samples.get_completions()
        helper = HyperoptTrialMaker(samples.get_hp_vectors(), 
                                    self.space_cfg, 
                                    self.response_shaping,
                                    self.shaping_func)
        
        t = helper.create_trials(comp, errs)
        num_iters = len(t.trials) + 1
        fmin(self.fake_evaluate, self.hyperopt_space, algo, num_iters,
                        trials=t, return_argmin=False) 

        hpv = self.get_last_hpv()
        idx = samples.expand(hpv)
        return idx[0] # XXX:return the first single index

    def get_last_hpv(self):
        hpv = []
        if self.last_params != None:
            for p in self.hyperparams:
                hpv.append(self.last_params[p])
        
        return np.array(hpv)

    def fake_evaluate(self, params):
        # XXX: replace index value of categorical string type to original string 
        params = self.space_cfg.replace_string_values(params)
        #debug("Selected by HyperOpt: {}".format(params))
        est_loss = 1.0 # XXX:meaningless loss value
        self.last_params = params
        
        return create_ok_result(est_loss)                 


class HyperOptSearchSpaceConfig(object):
    
    def __init__(self, config):
        if type(config) == dict:
            config = HyperparameterConfiguration(config)
        self.config = config
        self.hyperparams = self.config.get_param_names()

    def get_hyperopt_space(self):
        space = {}
        
        for param, setting in self.config.hyperparams.__dict__.items():
            range_distribution = None
            if setting.value_type == 'discrete' and setting.type == 'int':
                if hasattr(setting, 'power_of'):
                    base = setting.power_of
                    log_range = []
                    # transform setting.range values to log
                    for value in setting.range:
                        log_range.append(int(math.log(base**value)))                    
                    range_distribution = hp.qloguniform(param, log_range[0], log_range[-1], 1)                     
                else:
                    range_distribution = hp.quniform(param, setting.range[0], setting.range[-1], 1)
            elif setting.value_type == 'continuous' and setting.type == 'float':
                if hasattr(setting, 'power_of'):
                    base = setting.power_of
                    log_range = []
                    # transform setting.range values to log
                    for value in setting.range:
                        log_range.append(math.log(base**value))                    
                    range_distribution = hp.loguniform(param, log_range[0], log_range[-1])                     
                else:
                    range_distribution = hp.uniform(param, setting.range[0], setting.range[-1])
            elif setting.value_type == 'categorical' or setting.value_type == 'preordered':
                options = []
                str_index = 0
                for c in setting.range:
                    if isinstance(c, str):
                        # XXX: To avoid binning error
                        #c = str_index #str(c)                        
                        str_index += 1
                    options.append(c)

                range_distribution = hp.choice(param, options)
            else:
                error('invalid configuration: %s, %s' % (setting.value_type, setting.value))
            space[param] = range_distribution

        return space

    def get_type(self, name):
        range = []

        if name in self.hyperparams:
            hp = getattr(self.config.hyperparams, name)
            if hp.type == 'unicode':
                return "str"
            else:
                return hp.type
        
        return range

    def get_value_type(self, name):
        if name in self.hyperparams:
            hp = getattr(self.config.hyperparams, name)
            return hp.value_type
        error("Invalid name: {}".format(name))
        return None

    def get_range(self, name):
        return self.config.get_range(name)

    def replace_string_values(self, param_values):
        for param, setting in self.config.hyperparams.__dict__.items():
            try:
                if setting.type == "str":
                    if setting.value_type == 'categorical':
                        str_value = param_values[param]                        
                        param_values[param] = str(str_value)
                    else:
                        warn("Unknown case: {} - {}".format(param, setting.value_type))
            except Exception as ex:
                warn("Error: {}".format(ex))

        return param_values

def create_ok_result(loss, index=None):
    ok_result = {
        'loss': loss,
        'status': STATUS_OK
    }
    if index != None:
        ok_result['idx'] = index

    return ok_result


class HyperoptTrialMaker(object):
    def __init__(self, hpvs, space_cfg, 
                 response_shaping=False,
                 shaping_func="log_err"):
        self.hpvs = hpvs
        #debug("hparam vector dimension: {}".format(np.asarray(hpvs).shape))
        self.space_cfg = space_cfg
        self.hparams = self.space_cfg.hyperparams
        self.response_shaping = bool(response_shaping)
        self.shaping_func = shaping_func        

    def convert_to_bool_list(self, bool_vals):
        bools = []
        for b in bool_vals:
            b = self.convert_to_bool(str(b))
            bools.append(b)
        return bools

    def convert_to_bool(self, bool_val):
        b = str(bool_val).lower() in ("yes", "true")
        #debug("{} -> {}".format(bool_val, b))
        return b

    def create_history(self, completed):
        history = []
        
        if len(completed) > 0:
            #debug("completed: {}".format(completed))
            for ci in completed:
                hpv = self.hpvs[ci]
                if ci != None and len(self.hparams) == len(hpv):
                    h = {}
                    for i in range(len(self.hparams)):
                        str_index = 0
                        param = self.hparams[i]
                        val = hpv[i]
                        htype = self.space_cfg.get_value_type(param) 
                        t = self.space_cfg.get_type(param)
                        
                        if t == 'float': 
                            val = float(val)                        
                        elif t == 'int':
                            val = int(float(val))

                        if htype == "categorical" or htype == "preordered":
                            #XXX: This type raises binning error. To avoid this error, we replace it an index
                            cat_vals = self.space_cfg.get_range(param)
                            if t == 'bool':
                                #debug("boolean value: {}".format(val))
                                val = self.convert_to_bool(val)
                                cat_vals = self.convert_to_bool_list(cat_vals)
                            
                            index_val = cat_vals.index(val)
                            #debug("Transformed {} to {} at {}".format(val, index_val, cat_vals))
                            val = index_val

                        h[str(self.hparams[i])] = val
                    #debug("History: {}".format(h))
                    history.append(h)
                else:
                    debug("Invalid index {}".format(ci))
                
        return history

    def create_trials(self, completed, losses):
        if len(completed) > 0:
            trials = Trials()
            hist = self.create_history(completed)
            #index = 0
            #for c in completed:
            for index in len(completed):
                c = completed[index]
                loss = losses[index]
                rval_specs = [None]
                new_id = index
                rval_results = [ ]
                rval_results.append(create_ok_result(loss, c))
                rval_miscs = [  ]
                rval_miscs.append(self.create_misc(index, hist))
                    
                hopt_trial = trials.new_trial_docs([new_id], rval_specs, rval_results, rval_miscs)[0]

                if self.response_shaping is True:
                        # transform log applied loss for enhancing optimization performance
                        #debug("before scaling: {}".format(loss))
                    if self.shaping_func == "log_err":                        
                        loss = apply_log_err(loss)
                    elif self.shaping_func == "hybrid_log":
                        loss = apply_hybrid_log(loss)
                    else:
                        debug("Invalid shaping function: {}".format(self.shaping_func))
                if loss != None:                    
                    hopt_trial['result'] = {'loss': float(loss), 'status': STATUS_OK}
                    hopt_trial['state'] = JOB_STATE_DONE
                    #debug("History appended: {}-{}".format(c, loss))
                    trials.insert_trial_doc(hopt_trial)
            trials.refresh()
            return trials
        else:        
            return Trials()

    def create_misc(self, index, history):
        misc = {'tid': index, 
                'cmd': ('domain_attachment', 'FMinIter_Domain'), 
                'idxs': {}, 'vals': {} }
        result = history[index]
        for key in result.keys():
            misc['idxs'][key] = [ index ] 
            misc['vals'][key] = [ result[key] ] 
        
        return misc

