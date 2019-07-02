from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import time
import traceback
import pickle
import gzip

import numpy as np

from ws.shared.logger import *
import ws.shared.lookup as lookup
import ws.shared.hp_cfg as hconf
from ws.shared.saver import ResultSaver, TempSaver


from ws.hpo.sample_space import *
from ws.hpo.result import HPOResultFactory

from ws.hpo.utils.grid_gen import *
from ws.hpo.utils.measures import RankIntersectionMeasure
from ws.hpo.utils.converter import TimestringConverter

from ws.hpo.bandit_config import BanditConfigurator

from ws.hpo.connectors.remote_train import RemoteTrainConnector
import ws.hpo.trainers.trainer as trainer

NUM_MAX_ITERATIONS = 10000


def create_emulator(space,
                    run_mode, target_acc, time_expired, 
                    run_config=None,
                    save_internal=False,
                    num_resume=0,
                    id="HPO_emulator"):

    t = trainer.get_simulator(space, run_config)

    if run_config != None and "early_term_rule" in run_config:
        id = "{}.ETR-{}".format(id, run_config["early_term_rule"]) 

    machine = HPOBanditMachine(
        space, t, 
        run_mode, target_acc, time_expired, run_config, 
        num_resume=num_resume, 
        save_internal=save_internal,
        min_train_epoch=t.get_min_train_epoch(),
        id=id)
    
    return machine


def create_runner(trainer_url, space, 
                  run_mode, target_acc, time_expired, 
                  run_config, hp_config,
                  save_internal=False,
                  num_resume=0,
                  use_surrogate=None,
                  early_term_rule=None,
                  id="HPO_runner"
                  ):
    
    try:
        kwargs = {}
        if use_surrogate != None:            
            kwargs["surrogate"] = use_surrogate
            id += "-S_{}".format(use_surrogate)
        
        
        if isinstance(hp_config, dict):
            hp_config = hconf.HyperparameterConfiguration(hp_config)
        else:
            hp_config = hp_config

        cred = ""
        if "credential" in run_config:
            cred = run_config["credential"]
        
        rtc = RemoteTrainConnector(trainer_url, hp_config, cred, **kwargs)

        t = trainer.get_remote_trainer(rtc, space, run_config)

        if run_config and "early_term_rule" in run_config:
            early_term_rule = run_config["early_term_rule"]
            if early_term_rule != "None":
                id = "{}.ETR-{}".format(id, early_term_rule)

        machine = HPOBanditMachine(
            space, t, run_mode, target_acc, time_expired, run_config,
            num_resume=num_resume, 
            save_internal=save_internal, 
            id=id)
        
        return machine

    except Exception as ex:
        warn("Runner creation failed: {}".format(ex))


class HPOBanditMachine(object):
    ''' k-armed bandit machine of hyper-parameter optimization.'''
    def __init__(self, samples, trainer, 
                 run_mode, target_acc, time_expired, run_config,
                 num_resume=0, 
                 save_internal=False, 
                 calc_measure=False,
                 min_train_epoch=1,
                 id="HPO"):

        self.id = id

        self.samples = samples
        self.trainer = trainer
        self.save_name = samples.get_name()

        self.calc_measure = calc_measure
        
        self.target_accuracy = target_acc
        self.time_expired = TimestringConverter().convert(time_expired)
        self.eval_time_model = None

        self.run_mode = run_mode  # can be 'GOAL' or 'TIME'

        self.save_internal = save_internal
        
        self.temp_saver = None
        self.num_resume = num_resume

        self.stop_flag = False

        self.working_result = None
        self.total_results = None
        self.warm_up_time = None
        self.run_config = run_config
        self.min_train_epoch = min_train_epoch
        if self.run_config:
            if "min_train_epoch" in self.run_config:
                self.min_train_epoch = self.run_config["min_train_epoch"]

            if "warm_up_time" in self.run_config:
                self.warm_up_time = self.run_config["warm_up_time"]

        self.print_exception_trace = False

        self.saver = ResultSaver(self.save_name, self.run_mode, self.target_accuracy,
                                 self.time_expired, self.run_config, 
                                 postfix=".{}".format(self.id))

    def init_bandit(self, config=None):
        if config is None:
            config = self.run_config
        self.bandit = BanditConfigurator(self.samples, config)
        self.samples.reset()
        self.trainer.reset()
        self.working_result = None
        self.cur_runtime = 0.0
        
        self.eval_end_time = time.time()

    def force_stop(self):
        self.stop_flag = True
    
    def estimate_eval_time(self, cand_index, model):
        ''' Experimental feature. Not suggested to use now. '''        
        import ws.hpo.eval_time as eval_time
        samples = self.samples
        cand_est_times = None

        if self.eval_time_model and self.eval_time_model != "None":
            et = eval_time.get_estimator(samples, self.eval_time_model)
            
            if et is not None:
                success, cand_est_times = et.estimate(samples.get_candidates(), 
                                                    samples.get_completes())
                if success:
                    chooser = self.bandit.choosers[model]
                    chooser.set_eval_time_penalty(cand_est_times)

                if cand_est_times is not None:
                    for i in range(len(samples.get_candidates())):
                        if samples.get_candidates(i) == cand_index:
                            return int(cand_est_times[i])
        
        return None

    def select_candidate(self, model, acq_func, samples=None):
        if samples is None:
            samples = self.samples

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

        next_index = chooser.next(samples, acq_func, use_interim_result)

        # for measure information sharing effect
        if self.calc_measure:
            mr = RankIntersectionMeasure(samples.get_errors())
            if chooser.estimates:
                metrics = mr.compare_all(chooser.estimates['candidates'],
                                         chooser.estimates['acq_funcs'])

        opt_time = time.time() - start_time
        
        return next_index, opt_time, metrics

    def evaluate(self, cand_index, model, samples=None):
        if samples is None:
            samples = self.samples
        
        eval_start_time = time.time()
        exec_time = 0.0
        test_error = 1.0
        
        chooser = self.bandit.choosers[model]
        early_terminated = False
        if chooser.response_shaping == True:
            self.trainer.set_response_shaping(chooser.shaping_func)
        
        # set initial error for avoiding duplicate
        interim_error = self.trainer.get_interim_error(cand_index, 0)
        self.samples.update_error(cand_index, test_error, True)
        
        train_result = self.trainer.train(cand_index, 
                                          estimates=chooser.estimates,
                                          space=samples)
        
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
            next_index, opt_time, metrics = self.select_candidate(
                model, acq_func)
        
        except KeyboardInterrupt:
            self.stop_flag = True
            return 0.0, None
        except:
            warn("Exception occurred in the estimation processing. " +
                 "To avoid stopping, it selects the candidate randomly.")
            model = 'SOBOL'
            acq_func = 'RANDOM'
            next_index, opt_time, metrics = self.select_candidate(
                model, acq_func)
            exception_raised = True

        total_opt_time = select_opt_time + opt_time
        # XXX: To solve time mismatch problem   
        if self.eval_end_time != None:
            total_opt_time = time.time() - self.eval_end_time 
        # estimate an evaluation time of the next candidate
        #est_eval_time = self.estimate_eval_time(next_index, model)
        
        # evaluate the candidate
        eval_result = self.evaluate(next_index, model)
        test_error = eval_result['test_error']
        test_acc = 1.0 - test_error
        if 'test_acc' in eval_result:
            test_acc = eval_result['test_acc']            
        exec_time = eval_result['exec_time']
        early_terminated = eval_result['early_terminated']
        train_epoch = None
        if 'train_epoch' in eval_result:
            train_epoch = eval_result['train_epoch']
        best_epoch = None
        if 'best_epoch' in eval_result:
            eval_result['best_epoch']

        result_repo.append(next_index, test_error,
                           total_opt_time, exec_time, 
                           metrics=metrics, 
                           train_epoch=train_epoch,
                           best_epoch=best_epoch,
                           test_acc=test_acc)
        self.cur_runtime += (total_opt_time + exec_time)
        self.samples.update_error(next_index, test_error, early_terminated)
        
        curr_acc = test_acc
        optional = {
            'exception_raised': exception_raised,
        }
        if self.save_internal == True:
            est_log["estimated_values"] = self.bandit.choosers[model].estimates

        return curr_acc, optional

    def all_in(self, model, acq_func, num_trials, save_results=True):
        ''' executing a specific arm in an exploitative manner '''
        self.temp_saver = TempSaver(self.samples.get_name(),
                                    model, acq_func, num_trials, self.run_config)

        if self.num_resume > 0:
            self.total_results, start_idx = self.saver.load(
                model, acq_func, self.num_resume)
        else:
            self.total_results, start_idx = self.temp_saver.restore()

        est_records = {}
        for i in range(start_idx, num_trials):
            trial_start_time = time.time()
            self.init_bandit()
            wr = self.get_working_result()
            best_acc = 0.0
            est_records[str(i)] = []
            for j in range(NUM_MAX_ITERATIONS):

                curr_acc, opt_log = self.pull(model, acq_func, wr)
                if self.stop_flag == True:
                    return self.total_results
                if self.save_internal == True:
                    est_records[str(i)].append(opt_log)

                if best_acc < curr_acc:
                    best_acc = curr_acc
                
                if self.run_mode == 'GOAL':
                    if best_acc >= self.target_accuracy:
                        break
                elif self.run_mode == 'TIME':
                    duration = wr.get_elapsed_time()
                    #debug("current time: {} - {}".format(duration, self.time_expired))
                    if duration >= self.time_expired:
                        break
                    elif time.time() - trial_start_time >= self.time_expired:
                        debug("Trial time mismatch: {}".format(self.time_expired - duration))
                        break

            trial_sim_time = time.time() - trial_start_time
            log("{} found best accuracy {:.2f}% at run #{}. ({:.1f} sec)".format(
                self.id, best_acc * 100, i, trial_sim_time))            
            if best_acc < self.target_accuracy:
                wr.force_terminated()

            self.total_results[i] = wr.get_current_status()
            self.temp_saver.save(self.total_results)
            self.working_result = None
            
        if save_results is True:
            if self.save_internal == True:
                est_records = est_records
            else:
                est_records = None
            self.saver.save(model, acq_func,
                            num_trials, self.total_results, est_records)

        self.temp_saver.remove()
        self.show_best_hyperparams()

        return self.total_results

    def mix(self, strategy, num_trials, save_results=True):
        ''' executing the bandit with many arms by given mixing strategy '''
        model = 'DIV'
        self.temp_saver = TempSaver(self.samples.get_name(),
                                    model, strategy, num_trials, self.run_config)

        if self.num_resume > 0:
            self.total_results, start_idx = self.saver.load(
                model, strategy, self.num_resume)
        else:
            self.total_results, start_idx = self.temp_saver.restore()

        est_records = {}
        for i in range(start_idx, num_trials):            
            trial_start_time = time.time()
            self.init_bandit()
            arm = self.bandit.get_arm(strategy)
            wr = self.get_working_result()
            best_acc = 0.0

            est_records[str(i)] = []

            for j in range(NUM_MAX_ITERATIONS):
                iter_start_time = time.time()
                use_interim_result = True
                if self.warm_up_time != None:
                    if self.warm_up_time < self.cur_runtime:
                        use_interim_result = False
                model, acq_func, _ = arm.select(j, use_interim_result)

                curr_acc, opt_log = self.pull(model, acq_func, wr, 
                                             time.time() - iter_start_time)
                if self.stop_flag == True:
                    return self.total_results
                
                if opt_log['exception_raised']:
                    model = 'SOBOL'
                    acq_func = 'RANDOM'

                wr.update_trace(model, acq_func)
                arm.update(j, curr_acc, opt_log)
                
                if self.save_internal == True:
                    est_records[str(i)].append(opt_log)

                if best_acc < curr_acc:
                    best_acc = curr_acc

                if self.run_mode == 'GOAL':
                    if best_acc >= self.target_accuracy:
                        break
                elif self.run_mode == 'TIME':
                    duration = wr.get_elapsed_time()
                    #debug("current time: {} - {}".format(duration, self.time_expired))
                    if duration >= self.time_expired:
                        break
                    elif time.time() - trial_start_time >= self.time_expired:
                        debug("Trial time mismatch: {}".format(self.time_expired - duration))
                        break

            trial_sim_time = time.time() - trial_start_time
            log("{} found best accuracy {:.2f}% at run #{}. ({:.1f} sec)".format(
                self.id, best_acc * 100, i, trial_sim_time))
            if best_acc < self.target_accuracy:
                wr.force_terminated()

            wr.feed_arm_selection(arm)
            self.total_results[i] = wr.get_current_status()
            self.working_result = None # reset working result
            self.temp_saver.save(self.total_results)

        if save_results is True:
            if self.save_internal == True:
                est_records = est_records
            else:
                est_records = None
            self.saver.save('DIV', strategy,
                            num_trials, self.total_results, est_records)

        self.temp_saver.remove()
        self.show_best_hyperparams()

        return self.total_results

    def get_current_results(self):
        results = []
        if self.total_results:
            results += self.total_results

        if self.working_result != None:
            results.append(self.working_result.get_current_status())

        return results

    def get_working_result(self):
        if self.working_result == None:
            self.working_result = HPOResultFactory() 
        return self.working_result

    def show_best_hyperparams(self):
        for k in self.total_results.keys():
            result = self.total_results[k]
            
            acc_max_index = np.argmax(result['accuracy'])
            max_model_index = result['model_idx'][acc_max_index]
            #debug(max_model_index)
            max_hpv = self.samples.get_hpv(max_model_index)
            log("Achieved {:.4f} at run #{} by {}".format(result['accuracy'][acc_max_index], k, max_hpv))
            #accs = [ round(acc, 4) for acc in result['accuracy'] ]
            #log("selected accuracies: {}".format(accs))        

   
