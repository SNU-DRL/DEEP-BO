from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy

import numpy as np

from ws.shared.logger import *
from ws.shared.saver import BatchResultSaver
from hpo.utils.converter import TimestringConverter

import hpo.bandit as bandit

from hpo.result import BatchHPOResultFactory
from hpo.batch_cand import CandidateSelector
import hpo.space_mgr as space


def get_simulator(sync_mode, dataset_name, run_mode, target_acc, time_expired, config):
    sim = None
    if sync_mode == 'ASYNC':
        sim = AsynchronusBatchSimulator
    else:
        sim = SynchronusBatchSimulator
    time_expired = TimestringConverter().convert(time_expired)
    cluster = sim(run_mode, config, dataset_name, target_acc, time_expired)
    return cluster


class BatchHPOSimulator(object):

    def __init__(self, run_mode, config, dataset, target_acc, time_expired):
        
        self.target_acc = target_acc
        self.time_expired = time_expired
        self.elapsed_time = 0.0
        self.cur_sync_time = 0.0
        self.run_mode = run_mode

        self.cur_samples = None
        self.cur_iters = 0
        self.data_type = dataset
        self.config = config
        self.early_term_rule = "None"

        self.failover = 'premature' # can be 'None', 'random', 'premature' or 'next_candidate'

        if 'failover' in config:
            self.failover = config['failover']

        if "early_term_rule" in config:
            self.early_term_rule = config['early_term_rule']

        postfix = "FO-{}.ETR-{}".format(self.failover, self.early_term_rule)
        self.saver = BatchResultSaver(self.data_type, self.run_mode, self.target_acc,
                                    self.time_expired, self.config, postfix=postfix)
        
        self.bandits = self.create_bandits(config)

    def create_bandits(self, config):
        try:
            bandits = config['bandits']
        except:
            bandits = []
            for m in ['GP-HLE', 'RF-HLE']:
                for a in ['EI', 'PI', 'UCB']:
                    b = {'mode': m, 'spec': a}
                    bandits.append(b)
            bandits.append({'mode': 'SOBOL', 'spec': 'RANDOM'})

        bandit_list = []
        
        for i in range(len(bandits)):
            b = {}
            conf = None
            save_internal = False
            
            if 'config' in bandits[i]:
                conf = bandits[i]['config']
                if self.config:
                    b['title'] = "{}-{}".format(self.config['title'].replace(" ", "_"), i)
                    conf['title'] = b['title']
            else:    
                conf = {}            
            
            
            conf['arms'] = self.config['arms']
            if 'failover' in self.config:
                conf['failover'] = self.config['failover']
            if 'eta' in self.config:
                conf['eta'] = self.config['eta']

            if 'save_internal' in bandits[i]:
                save_internal = bandits[i]['save_internal']

            grid_order = None 
            
            if 'search_space' in conf and 'order' in conf['search_space']:
                grid_order = conf['search_space']['order']                 
            
            if "early_term_rule" in config:
                conf['early_term_rule'] = config['early_term_rule']

            samples = space.create_lookup_space(self.data_type, grid_order)
            emulator = bandit.create_emulator(samples, self.run_mode, self.target_acc, self.time_expired, 
                                 run_config=conf, save_internal=save_internal)
            b['m_id'] = i
            b['machine'] = emulator
            b['mode'] = bandits[i]['mode']
            b['spec'] = bandits[i]['spec']
            b['config'] = conf
            b['local_result'] = emulator.get_working_result()
            b['local_time'] = 0.0
            b['cur_work_index'] = None

            if i == 0:
                self.cur_samples = emulator.samples

            bandit_list.append(copy.deepcopy(b)) #XXX:mysterious behavior in here
        return bandit_list

    def update_history(self, bandit):

        # update the history of iteration which was selected by each machine
        machine_index = bandit['m_id']
        done_index = bandit['local_result'].get_value('model_idx', -1)
        # debug("machine #{} has evaluated work index #{}".format(machine_index, done_index))

        self.cur_samples.update_error(done_index)
        

    def get_optimizer(self, bandit, cur_iters, num_random_start=2):
        if bandit['mode'] == 'DIV':
            model, acq_func, _ = bandit['arm'].select(cur_iters)            
        else:
            model = bandit['mode']
            acq_func = bandit['spec']

        # random starts
        if cur_iters < num_random_start:
            model = 'SOBOL'
            acq_func = 'RANDOM'

        return model, acq_func

    def run(self, num_trials, save_results=True):
        raise NotImplementedError("Chlid class should be implemented.")

    def reset_bandits(self):
        # initialize all bandits
        for b in self.bandits:
            m = b['machine']
            conf = None
            if hasattr(m, 'config'):
                conf = m.config
            elif 'config' in b:
                conf = b['config']
            else:
                raise ValueError("Invaild configuration")
            m.init_bandit(conf)

            if b['mode'] == 'DIV':
                b['arm'] = m.bandit.get_arm(b['spec'])

            b['opt_time'] = 0.0
            b['eval_time'] = 0.0
            b['cur_start_time'] = 0
            b['cur_end_time'] = 0
            b['cur_iters'] = 0
            # the count how many times the optimizer selects working items.
            b['num_duplicates'] = 0

            b['local_result'] = m.get_working_result()

            b['samples'] = m.samples  # reset sampling space

            if self.cur_samples is None:
                self.cur_samples = copy.deepcopy(b['samples'])

        debug('initialized for new trial.')


class AsynchronusBatchSimulator(BatchHPOSimulator):

    def __init__(self, run_mode, config, dataset, target_acc, time_expired):
        super(AsynchronusBatchSimulator, self).__init__(
            run_mode, config, dataset, target_acc, time_expired)
        
    def run(self, num_trials, save_results=True):
        start_idx = 0
        results = {}

        for t in range(start_idx, num_trials):
            self.cur_samples = None
            trial_start_time = time.time()

            self.reset_bandits()

            cur_async_time = 0.0
            best_acc = 0.0

            res = BatchHPOResultFactory()
            cur_shelves = []
            timeout = False
            i = 0
            while timeout is False:
                debug("asynchronous optimization in bandit #{} with shelve: {}".format(
                    i, cur_shelves))
                acc, exec_index = self.async_next(i, cur_async_time, cur_shelves)
                cur_shelves.append({'model_idx': exec_index, 'start_time': cur_async_time})

                work_items = [b['cur_work_index'] for b in self.bandits]
                debug("current working items:  {}".format(work_items))

                # update history
                end_time_list = [b['cur_end_time'] for b in self.bandits]
                if cur_async_time < min(end_time_list):
                    i = np.argmin(end_time_list)
                    exec_index = self.bandits[i]['cur_work_index']
                    
                    cur_shelves = [ x for x in cur_shelves if not (exec_index == x.get('model_idx'))]
                    cur_async_time = end_time_list[i]
                    debug('item #{} ends in machine #{} at {:.1f}'.format(
                        exec_index, i, cur_async_time))
                    self.bandits[i]['cur_start_time'] = cur_async_time
                    self.update_history(self.bandits[i])
                    debug("number of completions: {}".format(len(self.cur_samples.get_completions())))
                else:
                    i = (i + 1) % len(self.bandits)

                if best_acc < acc:
                    best_acc = acc

                if self.run_mode == "GOAL" and best_acc >= self.target_acc:
                    timeout = True
                    res.update_batch_result(self.bandits)
                    break
                elif cur_async_time >= self.time_expired:
                    timeout = True
                    res.update_batch_result(self.bandits)
                    break

            trial_sim_time = time.time() - trial_start_time
            log("Best accuracy at run #{}: {:.2f}%, batch simulation time: {:.1f}".format(
                t, best_acc * 100, trial_sim_time))

            results[t] = res.get_current_status()

        if save_results is True:
            self.saver.save('async', num_trials, results)
        return results
  
    def async_next(self, i, cur_time, cur_shelves):
        b = self.bandits[i]

        b['samples'] = copy.deepcopy(self.cur_samples)
        debug('The number of observations in history: {}'.format(
            len(b['samples'].get_completions())))
        model, acq_func = self.get_optimizer(b, b['cur_iters'])
        cs_func = CandidateSelector(model, acq_func).select(self.failover)
        next_index, opt_time, model, acq_func = cs_func(b, cur_shelves, cur_time)

        #est_exec_time = b['machine'].estimate_eval_time(next_index, model)
        eval_result = b['machine'].evaluate(next_index, model, b['samples'])
        test_error = eval_result['test_error']
        exec_time = eval_result['exec_time']
        acc = eval_result['test_acc']
        train_epoch = None
        if 'train_epoch' in eval_result:
            train_epoch = eval_result['train_epoch']
                
        b['local_result'].append(next_index, test_error,
                                 opt_time, exec_time,
                                 train_epoch=train_epoch)
        b['samples'].update_error(next_index, test_error)

        b['local_result'].update_trace(model, acq_func)
        if b['mode'] == 'DIV':
            b['arm'].update(b['cur_iters'], acc, None)

        duration = b['local_result'].get_total_duration(-1)
        b['cur_work_index'] = next_index
        b['cur_end_time'] += duration
        b['cur_iters'] += 1

        return acc, next_index


class SynchronusBatchSimulator(BatchHPOSimulator):

    def __init__(self, run_mode, config, dataset, target_acc, time_expired):
        super(SynchronusBatchSimulator, self).__init__(
            run_mode, config, dataset, target_acc, time_expired)

    def run(self, num_trials, save_results=True):
        start_idx = 0
        results = {}

        for t in range(start_idx, num_trials):
            best_acc = 0.0
            trial_start_time = time.time()
            self.cur_iters = 0
            self.cur_sync_time = 0.0
            self.reset_bandits()
            repo = None
            for b in self.bandits:
                m = b['machine']
                m.init_bandit(b['config'])
                if b['mode'] == 'DIV':
                    b['arm'] = m.bandit.get_arm(b['spec'])

                b['num_duplicates'] = 0
                if repo is None:
                    repo = m.get_working_result()
                    self.cur_samples = m.samples
                    
            while self.sync_next(repo):
                cur_iter_acc = repo.get_value('accuracy', -1)
                if best_acc < cur_iter_acc:
                    best_acc = cur_iter_acc

                if self.run_mode == "GOAL" and best_acc >= self.target_acc:
                    break

                self.cur_iters += 1

            trial_sim_time = time.time() - trial_start_time
            log("Best accuracy at run #{}: {:.2f}%, batch simulation time: {}".format(
                t, best_acc * 100, trial_sim_time))

            results[t] = repo.get_current_status()

        if save_results is True:
            self.saver.save('sync', num_trials, results)
        return results

    def sync_next(self, repo):
        if self.cur_sync_time < self.time_expired:

            cur_shelves = []
            opt_times = []
            
            # candidate selection stage
            for b in self.bandits:
                model, acq_func = self.get_optimizer(b, self.cur_iters)
                b['samples'] = self.cur_samples
                cs_func = CandidateSelector(model, acq_func).select(self.failover)
                next_index, opt_time, model, acq_func = cs_func(
                    b, cur_shelves, self.cur_sync_time)
                cur_shelves.append({'model_idx': next_index, 'start_time': self.cur_sync_time})
                opt_times.append(opt_time)

            debug("evaluation candidates: {}".format(cur_shelves))

            # candidate evaluation stage
            max_duration = 0.0
            best_acc = 0.0
            best_bandit = None
            worst_bandit = None

            i = 0
            for b in self.bandits:
                samples = copy.deepcopy(self.cur_samples)
                optimizer, acquistion_func = self.get_optimizer(
                    b, self.cur_iters)
                next_index = cur_shelves[i]['model_idx']

                #est_exec_time = b['machine'].estimate_eval_time(next_index, optimizer)
                eval_result = b['machine'].evaluate(next_index, optimizer, samples)
                test_error = eval_result['test_error']
                exec_time = eval_result['exec_time']
                train_epoch = None
                if 'train_epoch' in eval_result:
                    train_epoch = eval_result['train_epoch']
                acc = eval_result['test_acc']
                total_opt_time = opt_times[i]
                b['local_result'].append(next_index, test_error,
                                        total_opt_time, exec_time,
                                        train_epoch=train_epoch)
                b['samples'].update_error(next_index, test_error)
                
                i += 1

                duration = b['local_result'].get_total_duration(-1)

                debug('synchronized at {:.1f} - machine #{} found {:.4f}  ({:.0f} secs)'.format(
                    self.cur_sync_time, b['m_id'], acc, duration))

                if b['mode'] == 'DIV':
                    b['local_result'].update_trace(optimizer, acquistion_func)
                    b['arm'].update(self.cur_iters, acc, None)

                if max_duration < duration:
                    max_duration = duration
                    worst_bandit = b

                if acc > self.target_acc:
                    debug('Target accuracy {} is achieved by {}'.format(
                        self.target_acc, acc))
                    return False

                if best_acc < acc:
                    best_acc = acc
                    best_bandit = b

            if best_bandit is not None:
                select_index = best_bandit['local_result'].get_value(
                    'model_idx', -1)
                test_error = best_bandit['local_result'].get_value('error', -1)
                exec_time = worst_bandit['local_result'].get_value(
                    'exec_time', -1)
                opt_time = worst_bandit['local_result'].get_value(
                    'opt_time', -1)

                repo.append(select_index, test_error,
                                 opt_time, exec_time)
                repo.count_duplicates(cur_shelves)

            self.cur_sync_time += max_duration

            # XXX: update the iteration result after all machines are finished.
            for b in self.bandits:
                self.update_history(b)
            debug("number of completions at {}: {}".format(self.cur_sync_time, len(self.cur_samples.get_completions())))
            return True

        else:
            debug('Timeout')
            return False


