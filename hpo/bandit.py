from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import time
import traceback
import pickle
import gzip
import threading

import numpy as np

from ws.shared.logger import *
from ws.shared.hp_cfg import HyperparameterConfiguration
from ws.shared.saver import *

from hpo.search_space import *
from hpo.space_mgr import *

from hpo.results import ResultsRepository

from hpo.utils.measures import RankIntersectionMeasure
from ws.shared.converter import TimestringConverter

from hpo.bandit_config import BanditConfigurator

from hpo.connectors.remote_train import RemoteTrainConnector
import hpo.trainers.trainer as trainer

NUM_MAX_ITERATIONS = 10000


def create_emulator(space,
                    run_mode, target_val, time_expired,
                    goal_metric="error", 
                    run_config=None,
                    save_internal=False,
                    num_resume=0,
                    id="Emulator"):

    t = trainer.get_simulator(space, run_config)

    if run_config != None and "early_term_rule" in run_config:
        id = "{}.ETR-{}".format(id, run_config["early_term_rule"]) 

    machine = HPOBanditMachine(space, t, 
                               run_mode, target_val, time_expired, run_config, 
                               goal_metric=goal_metric,
                               num_resume=num_resume, 
                               save_internal=save_internal,
                               min_train_epoch=t.get_min_train_epoch(),
                               id=id)
    
    return machine


def create_runner(trainer_url, space, 
                  run_mode, target_val, time_expired, 
                  run_config, hp_config,
                  goal_metric="error",
                  save_internal=False,
                  num_resume=0,
                  use_surrogate=None,
                  early_term_rule="DecaTercet",
                  id="Runner"
                  ):
    
    try:
        kwargs = {}
        if use_surrogate != None:            
            kwargs["surrogate"] = use_surrogate
            id += "-S_{}".format(use_surrogate)
        
        
        if isinstance(hp_config, dict):
            hp_config = HyperparameterConfiguration(hp_config)

        cred = ""
        if "credential" in run_config:
            cred = run_config["credential"]
        else:
            raise ValueError("No credential info in run configuration")            
        
        rtc = RemoteTrainConnector(trainer_url, hp_config, cred, **kwargs)        
        t = trainer.get_remote_trainer(rtc, space, run_config)

        if run_config and "early_term_rule" in run_config:
            early_term_rule = run_config["early_term_rule"]
            if early_term_rule != "None":
                id = "{}.ETR-{}".format(id, early_term_rule)

        machine = HPOBanditMachine(space, t, 
                                   run_mode, target_val, time_expired, run_config,
                                   goal_metric=goal_metric,
                                   num_resume=num_resume, 
                                   save_internal=save_internal,
                                   id=id)
        
        return machine

    except Exception as ex:
        warn("Runner creation failed: {}".format(ex))


class HPOBanditMachine(object):
    ''' k-armed bandit machine of hyper-parameter optimization.'''
    def __init__(self, s_space, trainer, 
                 run_mode, target_val, time_expired, run_config,
                 goal_metric,
                 num_resume=0, 
                 save_internal=False, 
                 calc_measure=False,
                 min_train_epoch=1,
                 id="HPOBanditMachine"):

        self.id = id

        self.search_space = s_space
        self.save_name = s_space.get_name()
        
        self.sample_thread = None
        self.trainer = trainer
        
        self.calc_measure = calc_measure
        
        self.target_goal = target_val
        self.goal_metric = goal_metric
        self.time_expired = TimestringConverter().convert(time_expired)

        self.save_internal = save_internal

        self.num_resume = num_resume
        self.min_candidates = 100 # XXX:any number to be tested 

        self.stop_flag = False

        self.repo = ResultsRepository(self.goal_metric)
        self.current_results = None
        self.warm_up_time = 0
        self.warm_up_select = {} # XXX: dictionary for the selected configurations in warm-up phase
        self.warm_up_revisit = 3 # XXX: number of best configurations from warm-up phase to be revisited
        self.cur_runtime = 0.0 
        self.run_config = run_config
        self.min_train_epoch = min_train_epoch
        self.max_train_epoch = None
        if self.run_config:
            if "min_train_epoch" in self.run_config:
                self.min_train_epoch = self.run_config["min_train_epoch"]

            if "max_train_epoch" in self.run_config:
                self.max_train_epoch = self.run_config["max_train_epoch"]
            if "warm_up_time" in self.run_config:
                self.warm_up_time = TimestringConverter().convert(self.run_config["warm_up_time"])
            if "warm_up_revisit" in self.run_config:
                self.warm_up_revisit = self.run_config['warm_up_revisit']

        self.run_mode = run_mode  # can be 'GOAL' or 'TIME'
        criterion = ""
        if run_mode == "TIME":
            num_trials = 1
            if 'num_trials' in self.run_config:
                num_trials = self.run_config['num_trials']
            term_time = time.localtime(time.time() + (self.time_expired * num_trials))
            criterion = "near {}".format(time.asctime(term_time))
        elif run_mode == "GOAL":
            criterion = "when achieving {} {}".format(self.target_goal, goal_metric)
        log("This will be ended {}".format(criterion))

        self.print_exception_trace = False


    def reset(self, config=None):
        if config is None:
            config = self.run_config
        self.bandit = BanditConfigurator(self.search_space, config)
        self.search_space.reset()
        self.trainer.reset()
        self.repo = ResultsRepository(self.goal_metric)
        self.cur_runtime = 0.0
        self.warm_up_select = {}
        
        self.eval_end_time = time.time()

    def stop(self):
        self.stop_flag = True
    
    def predict_time(self, cand_index, model, time_model):
        ''' Experimental feature. This is not be suggested to use. '''        
        from hpo.predict_time import get_estimator
        et = get_estimator(self.search_space, time_model)
        success, cand_est_times = et.estimate(self.search_space.get_candidates(), 
                                            self.search_space.get_completions())
        if success:
            chooser = self.bandit.choosers[model]
            chooser.set_eval_time_penalty(cand_est_times)

        if cand_est_times is not None:
            for i in range(len(self.search_space.get_candidates())):
                if self.search_space.get_candidates()[i] == cand_index:
                    return int(cand_est_times[i])
        return None

    def choose_from_low_fidelity(self, space, n_rank):
        try:
            for k in self.warm_up_select.keys():                
                err = space.get_errors(k)
                if type(err) == list:
                    err = err[0]                
                s = self.warm_up_select[k]
                if s['index'] != k: # validate key-value set
                    raise ValueError("Invalid key-value pair in dictionary.")
                s['error'] = err
            sorted_list = sorted(self.warm_up_select.items(), key=lambda item: item[1]['error'] )
            r = sorted_list[n_rank][1]
            return r
        except Exception as ex:
            warn("Exception at choose_from_low_fidelity(): {}".format(ex))
    def choose(self, model, acq_func, search_space=None):
        if search_space is None:
            search_space = self.search_space
        ss = 1  # XXX:default steps of sampling
        
        if "search_space" in self.run_config:
            if "resample_steps" in self.run_config['search_space']:
                ss = self.run_config['search_space']["resample_steps"]

        metrics = []
        chooser = self.bandit.choosers[model]
        exception_raised = False
        
        if self.sample_thread != None and self.sample_thread.is_alive():
            while len(search_space.get_candidates()) < self.min_candidates: # waiting until enough sample population
                time.sleep(1)
        num_done = len(search_space.get_completions())
        num_cand = len(search_space.get_candidates())
        debug("# of observations, candidates: {}, {}".format(num_done, num_cand))
        
        next_index = chooser.next(search_space, acq_func)
        est_values = chooser.estimates
        if self.cur_runtime > self.warm_up_time:
            self.search_space.set_min_train_epoch(self.min_train_epoch) 
            if len(self.warm_up_select.keys()) < self.warm_up_revisit:
                warn("Number of evaluations in warm-up phase ({}) is less than {}.".format(len(self.warm_up_select), self.warm_up_revisit))
                self.warm_up_revisit = len(self.warm_up_select.keys())
            n_compl = len(search_space.get_completions())            
            if n_compl < self.warm_up_revisit:
                debug("High-fidelity optimization: {}/{}".format(n_compl, self.warm_up_revisit))
                pre_select = self.choose_from_low_fidelity(search_space, n_compl)
                if pre_select != None:                    
                    next_index = pre_select['index']
                    model = pre_select['model']
                    acq_func = pre_select['acq_func']
                    est_values = None
                    debug("Revisit {}th best configuration in warm-up phase: {}".format(n_compl+1, next_index))
                else:
                    warn("Restoring best candidate from warm-up phase failed.") 
        else:
            self.warm_up_select[next_index] = { "model": model, "acq_func": acq_func, "index": next_index }

        if num_done > 0 and num_done % ss == 0:
            if 'increment' in self.run_config['search_space'] and \
                self.run_config['search_space']["increment"]:
                # samples will be transformed in parallel
                if self.sample_thread != None and self.sample_thread.is_alive():
                    debug("Now on candidate sampling... ")
                else:
                    debug("Incremental sampling performed asynchronously.")
                    self.sample_thread = threading.Thread(target=self.sample, 
                                                          args=(est_values,))
                    self.sample_thread.start()
            else:
                # samples will be generated sequentially                               
                self.sample(est_values)

        # for measure information sharing effect
        if self.calc_measure:
            mr = RankIntersectionMeasure(search_space.get_errors())
            if chooser.estimates:
                metrics = mr.compare_all(chooser.estimates['candidates'],
                                         chooser.estimates['acq_funcs'])

        
        return next_index, metrics

    def evaluate(self, chooser, cand_index, train_epoch):
        
        eval_start_time = time.time()
        exec_time = 0.0
        
        early_terminated = False
        interim_error, cur_iter = self.trainer.get_interim_error(cand_index, 0)
        self.search_space.update_error(cand_index, interim_error)
        if chooser.response_shaping == True:
            self.trainer.set_response_shaping(chooser.shaping_func)
        if chooser.estimates != None:
            self.trainer.set_estimation(chooser.estimates)
        
        train_result = self.trainer.train(cand_index, train_epoch)

        if train_result == None or not 'error' in train_result:
            train_result = {}
            # return interim error for avoiding stopping
            train_result['error'] = interim_error            
            train_result['early_terminated'] = True
            test_error = interim_error
        else:
            test_error = train_result['error']
        
        train_result['model_idx'] = cand_index

        self.eval_end_time = time.time()
        if not 'exec_time' in train_result:
            train_result['exec_time'] = self.eval_end_time - eval_start_time

        if 'train_epoch' in train_result:
            train_epoch = train_result['train_epoch']
        else:
            train_result['train_epoch'] = train_epoch
        return train_result

    def pull(self, model, acq_func, pre_time=0):
        start_time = time.time()
        exception_raised = False
        try:
            cand_index, metrics = self.choose(model, acq_func)
        
        except KeyboardInterrupt:
            self.stop_flag = True
            return 0.0, None
        except:
            warn("Exception occurred in the estimation processing. " +
                 "To avoid stopping, it selects the candidate randomly.")
            model = 'NONE'
            acq_func = 'RANDOM'
            cand_index, metrics = self.choose(model, acq_func)            
            exception_raised = True

        opt_time = time.time() - start_time
        total_opt_time = pre_time + opt_time
        # XXX: To solve time mismatch problem   
        if self.eval_end_time != None:
            total_opt_time = time.time() - self.eval_end_time 
        
        # evaluate the candidate

        if self.cur_runtime < self.warm_up_time:
            train_epoch = self.min_train_epoch
        else:
            train_epoch = self.max_train_epoch
                 
        debug("Evaluation will be performed with {} epochs".format(train_epoch))
        chooser = self.bandit.choosers[model]
        eval_result = self.evaluate(chooser, cand_index, train_epoch)
        eval_result['opt_time'] = total_opt_time
        exec_time = eval_result['exec_time']
        
        self.cur_runtime += (total_opt_time + exec_time)
        self.repo.append(eval_result)
        
        opt_info = {
            'exception_raised': exception_raised,
            'exec_time': exec_time
        }
        if self.save_internal == True:
            opt_info["estimated_values"] = self.bandit.choosers[model].estimates

        return eval_result[self.goal_metric], opt_info


    def play(self, mode, spec, num_runs, save=True):

        temp_saver = TemporaryHistorySaver(self.save_name,
                                            mode, spec, num_runs, 
                                            self.run_config)
        saver = None
        if save == True:
            saver = HistorySaver(self.save_name, self.run_mode, self.target_goal,
                                    self.time_expired, self.run_config, 
                                    postfix=".{}".format(self.id))            
        
        # For in-depth analysis
        internal_records = None
        if self.save_internal:
            internal_records = {}

        # restore prior history
        if self.num_resume > 0:
            self.current_results, start_idx = saver.load(mode, spec, self.num_resume)
        else:
            self.current_results, start_idx = temp_saver.restore()
        
        if start_idx > 0:
            log("Temporary result of the prior execution is restored")
        for i in range(start_idx, num_runs): # loop for multiple runs           
            start_time = time.time()
            self.reset()
            incumbent = None
            arms = None
            if mode == 'DIV':
                arms = self.bandit.get_arms(spec)  

            if internal_records:
                internal_records[str(i)] = []

            for j in range(NUM_MAX_ITERATIONS): # loop for each run
                iter_start_time = time.time()

                # Model selection
                if arms:
                    arms = self.bandit.get_arms(spec)
                    model, acq_func = arms.select(j) 
                    debug("Selecting next candidate with {}-{}".format(model, acq_func))
                else:
                    model = mode
                    acq_func = spec
                prepare_time = time.time() - iter_start_time
                y, opt = self.pull(model, acq_func, prepare_time)
                if arms:
                    arms.update(j, y, opt)
               
                if opt['exception_raised']:
                    model = 'NONE'
                    acq_func = 'RANDOM'
                self.repo.update_trace(model, acq_func)

                if self.stop_flag == True:
                    return self.current_results

                if internal_records:
                    internal_records[str(i)].append(opt)

                if num_runs == 1:
                    self.current_results[i] = self.repo.get_current_status()
                    temp_saver.save(self.current_results)

                self.update_history(num_runs)
                # incumbent update
                if y == None: # in case of error, skip belows
                    continue                
                
                if incumbent == None:
                    incumbent = y
                elif self.goal_metric == "accuracy" and incumbent < y:
                    incumbent = y
                elif self.goal_metric == "error" and incumbent > y:
                    incumbent = y

                # stopping criteria check
                if self.check_stop(incumbent, start_time):
                    if mode == 'DIV':
                        self.repo.feed_selection(arms)  
                    break

            wall_time = time.time() - start_time
            log("Best {} at run #{} is {:.4f}. (wall time: {:.1f} secs)".format(self.goal_metric, 
                                                             i, incumbent, wall_time))

            if self.sample_thread != None and self.sample_thread.is_alive():
                self.sample_thread.join()
                self.sample_thread = None

            self.current_results[i] = self.repo.get_current_status()
            temp_saver.save(self.current_results)
            self.search_space.archive(i)

        if saver:
            saver.save(mode, spec, num_runs, self.current_results, internal_records)
        if start_idx == num_runs:
            warn("No more extra runs.")
        temp_saver.remove()

        return self.current_results

    def check_stop(self, incumbent, start_time):
        if self.run_mode == 'GOAL':
            if self.goal_metric == "accuracy" and incumbent >= self.target_goal:
                return True
            elif self.goal_metric == "error" and incumbent <= self.target_goal:
                return True
        elif self.run_mode == 'TIME':
            duration = self.repo.get_elapsed_time()
            #debug("current time: {} - {}".format(duration, self.time_expired))
            if duration >= self.time_expired:
                return True
            elif time.time() - start_time >= self.time_expired:
                debug("Trial time mismatch: {}".format(self.time_expired - duration))
                return True         
        return False
        
    def update_history(self, num_run):
        self.search_space.update_history(num_run)
    def get_results(self):
        results = []
        if self.current_results:
            results += self.current_results

        results.append(self.get_repo().get_current_status())
        return results

    def get_repo(self):
        if self.repo == None:
            self.repo = ResultsRepository(self.goal_metric) 
        return self.repo

    def print_best(self, results):
        for k in results.keys():
            try:
                result = results[k]
                
                error_min_index = 0
                cur_best_err = None
                for i in range(len(result['error'])):
                    err = result['error'][i]
                    if cur_best_err == None:
                        cur_best_err = err    
                    # if an error is None, ignore them
                    if err != None:
                        if cur_best_err > err:
                            cur_best_err = err
                            error_min_index = i
                
                best_model_index = result['model_idx'][error_min_index]
                best_error = result['error'][error_min_index]
                best_hpv = self.search_space.get_hpv_dict(best_model_index, int(k))
                log("Best performance {} at run #{} is achieved by {}.".format(best_error, k, best_hpv))
            except Exception as ex:
                warn("Report error ar run #{}".format(k))

    def sample(self, estimates):
        s_t = time.time()
        if not "search_space" in self.run_config:
            return

        if estimates == None:
            debug("No estimation available to modify samples.")
        else:

            if 'remove' in self.run_config['search_space']:
                start_t = time.time()
                ds = self.run_config['search_space']["remove"]
                remove_samples(self.search_space, ds, estimates)
                debug("Candidate(s) removed by {} ({:.1f} sec)".format(ds, time.time() - start_t))

            if 'add' in self.run_config['search_space']:
                start_t = time.time()
                space_setting = self.search_space.spec
                if 'sample_method' in self.run_config['search_space']:
                    space_setting['sample_method'] = self.run_config['search_space']['sample_method']
                else:
                    space_setting['sample_method'] = 'Sobol'
                ns = self.run_config['search_space']["add"]                                
                append_samples(self.search_space, ns)
                debug("{} candidate(s) added ({:.1f} sec)".format(ns, time.time() - start_t))
            if 'intensify' in self.run_config['search_space']:
                start_t = time.time()
                # intensify # of promissing samples using estimated values
                cands = np.array(estimates['candidates']) # has index
                est_values = np.array(estimates['acq_funcs']) # estimated performance by acquistion function
                ns = self.run_config['search_space']['intensify']
                top_k = est_values.argsort()[-1*ns:][::-1]
                
                for k in cands[top_k]:
                    cand = self.search_space.get_hpv_dict(k)
                    num_gen = self.search_space.get_generation(k)
                    intensify_samples(self.search_space, 1, cand, num_gen)
                debug("{} candidate(s) intensified. ({:.1f} sec)".format(ns, time.time() - start_t))

            if 'evolve' in self.run_config['search_space']:
                start_t = time.time()
            # evolving # of promissing samples using estimated values
                cands = np.array(estimates['candidates']) # has index
                est_values = np.array(estimates['acq_funcs']) # estimated performance by acquistion function
                ns = self.run_config['search_space']['evolve']
                top_1 = est_values.argsort()[-1:][::-1]
                i = self.search_space.get_incumbent()
                cur_best = { 
                            "hpv": self.search_space.get_hpv(i), 
                            "schema": self.search_space.get_schema(i),
                            "gen": self.search_space.get_generation(i)
                        }
                for k in cands[top_1]:
                    cand = self.search_space.get_hpv_dict(k)
                    evolve_samples(self.search_space, ns, cur_best, cand)
                debug("{} candidates evolved. ({:.1f} sec)".format(ns, time.time() - start_t))


        cand_size = len(self.search_space.get_candidates())
        debug("Current # of candidates: {} ({:.1f} sec)".format(cand_size, time.time() - s_t))  

   
