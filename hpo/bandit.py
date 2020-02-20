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
import ws.shared.hp_cfg as hconf
from ws.shared.saver import *

from hpo.search_space import *
from hpo.space_mgr import *

from hpo.results import ResultsRepository

from hpo.utils.measures import RankIntersectionMeasure
from hpo.utils.converter import TimestringConverter

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
            hp_config = hconf.HyperparameterConfiguration(hp_config)

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
        self.warm_up_time = None
        self.run_config = run_config
        self.min_train_epoch = min_train_epoch
        if self.run_config:
            if "min_train_epoch" in self.run_config:
                self.min_train_epoch = self.run_config["min_train_epoch"]

            if "warm_up_time" in self.run_config:
                self.warm_up_time = self.run_config["warm_up_time"]

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
        log("HPO will be ended {}".format(criterion))

        self.print_exception_trace = False


    def reset(self, config=None):
        if config is None:
            config = self.run_config
        self.bandit = BanditConfigurator(self.search_space, config)
        self.search_space.reset()
        self.trainer.reset()
        self.repo = ResultsRepository(self.goal_metric)
        self.cur_runtime = 0.0
        
        self.eval_end_time = time.time()

    def stop(self):
        self.stop_flag = True
    
    def predict_time(self, cand_index, model, time_model):
        ''' Experimental feature. This is not be suggested to use. '''        
        import hpo.predict_time as pt

        et = pt.get_estimator(self.search_space, time_model)
            
        success, cand_est_times = et.estimate(self.search_space.get_candidates(), 
                                            self.search_space.get_completions())
        if success:
            chooser = self.bandit.choosers[model]
            chooser.set_eval_time_penalty(cand_est_times)

        if cand_est_times is not None:
            for i in range(len(self.search_space.get_candidates())):
                if self.search_space.get_candidates(i) == cand_index:
                    return int(cand_est_times[i])
        
        return None

    def choose(self, model, acq_func, search_space=None):
        if search_space is None:
            search_space = self.search_space
        ss = 1  # XXX:default steps of sampling
        num_done = len(search_space.get_completions())
        
        if "search_space" in self.run_config:
            if "resample_steps" in self.run_config['search_space']:
                ss = self.run_config['search_space']["resample_steps"]

        metrics = []
        chooser = self.bandit.choosers[model]
        start_time = time.time()
        exception_raised = False
        use_interim_result = True
        if self.warm_up_time != None:
            if self.warm_up_time < self.cur_runtime:
                use_interim_result = False
                debug("HPO does not utilize interim results")
            else:
                debug("HPO utilizes interim results to warm up")
        
        if self.sample_thread != None and self.sample_thread.is_alive():
            while len(search_space.get_candidates()) < self.min_candidates: # waiting until enough sample population
                time.sleep(1)
        debug("# of observations, candidates: {}, {}".format(len(search_space.get_completions()),
                                                        len(search_space.get_candidates())))
        next_index = chooser.next(search_space, acq_func, use_interim_result)


        if num_done > 0 and num_done % ss == 0:
            if 'increment' in self.run_config['search_space'] and \
                self.run_config['search_space']["increment"]:
                # samples will be transformed in parallel
                if self.sample_thread != None and self.sample_thread.is_alive():
                    debug("Under search space construction... ")
                else:
                    debug("Incremental sampling performed asynchronously.")
                    self.sample_thread = threading.Thread(target=self.sample, 
                                                        args=(chooser.estimates,))
                    self.sample_thread.start()
            else:
                # samples will be transformed sequentially                               
                self.sample(chooser.estimates)

        # for measure information sharing effect
        if self.calc_measure:
            mr = RankIntersectionMeasure(search_space.get_errors())
            if chooser.estimates:
                metrics = mr.compare_all(chooser.estimates['candidates'],
                                         chooser.estimates['acq_funcs'])

        opt_time = time.time() - start_time
        
        return next_index, opt_time, metrics

    def evaluate(self, cand_index, model, search_space=None):
        if search_space is None:
            search_space = self.search_space
        
        eval_start_time = time.time()
        exec_time = 0.0
        test_error = 1.0 # FIXME: base test error should be reasonable
        
        chooser = self.bandit.choosers[model]
        early_terminated = False
        if chooser.response_shaping == True:
            self.trainer.set_response_shaping(chooser.shaping_func)
        
        # set initial error for avoiding duplicate
        interim_error, cur_iter = self.trainer.get_interim_error(cand_index, 0)
        search_space.update_error(cand_index, test_error, cur_iter)
        
        train_result = self.trainer.train(cand_index, 
                                          estimates=chooser.estimates,
                                          space=search_space)

        if train_result == None or not 'test_error' in train_result:
            train_result = {}
            # return interim error for avoiding stopping
            train_result['test_error'] = interim_error            
            train_result['early_terminated'] = True
        
        if not 'test_acc' in train_result:
            if train_result['test_error'] == None:
                train_result['test_acc'] = float("inf")
            else:
                train_result['test_acc'] = 1.0 - train_result['test_error']

        self.eval_end_time = time.time()
        if not 'exec_time' in train_result:
            train_result['exec_time'] = self.eval_end_time - eval_start_time

        return train_result

    def pull(self, model, acq_func, result_repo, select_opt_time=0):
        exception_raised = False
        try:
            next_index, opt_time, metrics = self.choose(model, acq_func)
        
        except KeyboardInterrupt:
            self.stop_flag = True
            return 0.0, None
        except:
            warn("Exception occurred in the estimation processing. " +
                 "To avoid stopping, it selects the candidate randomly.")
            model = 'SOBOL'
            acq_func = 'RANDOM'
            next_index, opt_time, metrics = self.choose(model, acq_func)
            exception_raised = True

        total_opt_time = select_opt_time + opt_time
        # XXX: To solve time mismatch problem   
        if self.eval_end_time != None:
            total_opt_time = time.time() - self.eval_end_time 
        
        # evaluate the candidate
        eval_result = self.evaluate(next_index, model)
        test_error = eval_result['test_error']

        if 'test_acc' in eval_result:
            test_acc = eval_result['test_acc']           
        elif metrics == 'accuracy' and test_error != None:
            test_acc = 1.0 - test_error
        else:
            test_acc = None
                 
        exec_time = eval_result['exec_time']
        early_terminated = eval_result['early_terminated']
        train_epoch = None
        if 'train_epoch' in eval_result:
            train_epoch = eval_result['train_epoch']
        
        result_repo.append(next_index, test_error,
                           total_opt_time, exec_time, 
                           metrics=metrics, 
                           train_epoch=train_epoch,
                           test_acc=test_acc)
        self.cur_runtime += (total_opt_time + exec_time)
        self.search_space.update_error(next_index, test_error, train_epoch)
        
        optional = {
            'exception_raised': exception_raised,
        }
        if self.save_internal == True:
            est_log["estimated_values"] = self.bandit.choosers[model].estimates

        if self.goal_metric == "accuracy":
            return test_acc, optional
        elif self.goal_metric == "error":
            return test_error, optional

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

        for i in range(start_idx, num_runs): # loop for multiple runs           
            start_time = time.time()
            self.reset()
            incumbent = None

            if internal_records:
                internal_records[str(i)] = []

            for j in range(NUM_MAX_ITERATIONS):
                iter_start_time = time.time()
                use_interim_result = True
                if self.warm_up_time != None:
                    if self.warm_up_time < self.cur_runtime:
                        use_interim_result = False
						
                if mode == 'DIV':
                    arms = self.bandit.get_arms(spec)
                    mode, spec, _ = arms.select(j, use_interim_result)

                prepare_time = time.time() - iter_start_time
                curr_val, opt_log = self.pull(mode, spec, self.repo, prepare_time)
                if mode == 'DIV':
                    arms.update(j, curr_val, opt_log)
                
                if opt_log['exception_raised']:
                    mode = 'SOBOL'
                    spec = 'RANDOM'
                self.repo.update_trace(mode, spec)

                if self.stop_flag == True:
                    return self.current_results
                
                if internal_records:
                    internal_records[str(i)].append(opt_log)

                if num_runs == 1:
                    self.current_results[i] = self.repo.get_current_status()
                    temp_saver.save(self.current_results)
                # incumbent update
                if curr_val == None: # in case of error, skip belows
                    continue
                
                if incumbent == None:
                    incumbent = curr_val
                elif self.goal_metric == "accuracy" and incumbent < curr_val:
                    incumbent = curr_val
                elif self.goal_metric == "error" and incumbent > curr_val:
                    incumbent = curr_val

                # stopping criteria check
                if self.check_stop(incumbent, start_time):
                    if mode == 'DIV':
                        self.repo.feed_selection(arms)  
                    break

            wall_time = time.time() - start_time
            log("Best {} {:.4f} at run #{}. (wall time: {:.1f} secs)".format(self.goal_metric, 
                                                             incumbent, 
                                                             i, 
                                                             wall_time))

            if self.sample_thread != None and self.sample_thread.is_alive():
                self.sample_thread.join()
                self.sample_thread = None

            self.current_results[i] = self.repo.get_current_status()
            temp_saver.save(self.current_results)
            self.search_space.archive(i)

        if saver:
            saver.save(mode, spec, num_runs, self.current_results, internal_records)
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
                best_hpv = self.search_space.get_hpv_dict(best_model_index)
                log("Best performance at run #{} by {} is {}.".format(int(k)+1, best_hpv, best_error))
            except Exception as ex:
                warn("Report error ar run #{}".format(k))

    def sample(self, estimates):
        s_t = time.time()
        if not "search_space" in self.run_config:
            return
			
        if 'remove' in self.run_config['search_space']:
            start_t = time.time()
            ds = self.run_config['search_space']["remove"]
            remove_samples(self.search_space, ds, estimates)
            debug("Removed with {} ({:.1f} sec)".format(ds, time.time() - start_t))
			
        if 'intensify' in self.run_config['search_space']:
            if estimates == None:
                warn("No estimation available to intensify samples.")
            else:             
                start_t = time.time()
				
                cands = np.array(estimates['candidates']) # has index
                est_values = np.array(estimates['acq_funcs']) # estimated performance by acquistion function
                ns = self.run_config['search_space']['intensify']
                top_k = est_values.argsort()[-1*ns:][::-1]
				
                for k in cands[top_k]:
                    cand = self.search_space.get_hpv_dict(k)
                    intensify_samples(self.search_space, 1, cand)
                debug("{} samples intensified. ({:.1f} sec)".format(ns, time.time() - start_t))
				
        if 'evolution' in self.run_config['search_space']:
            if estimates == None:
                warn("No estimation available to evolve samples.")
            else:             
                start_t = time.time()
				
                cands = np.array(estimates['candidates']) # has index
                est_values = np.array(estimates['acq_funcs']) # estimated performance by acquistion function
                ns = self.run_config['search_space']['evolution']
                top_1 = est_values.argsort()[-1:][::-1]
                i = self.search_space.get_incumbent()
                cur_best = { "hpv": self.search_space.get_hpv(i), 
                             "schema": self.search_space.get_schema(i) }
                for k in cands[top_1]:
                    cand = self.search_space.get_hpv_dict(k)
                    evolve_samples(self.search_space, ns, cur_best, cand)
                debug("{} samples evolved. ({:.1f} sec)".format(ns, time.time() - start_t))
				
        if 'add' in self.run_config['search_space']:
            start_t = time.time()
            ns = self.run_config['search_space']["add"]                                
            append_samples(self.search_space, ns)
            debug("{} samples added ({:.1f} sec)".format(ns, time.time() - start_t))
			
        cand_size = len(self.search_space.get_candidates())
        debug("Current # of candidates: {} ({:.1f} sec)".format(cand_size, time.time() - s_t))
		  
   
