##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This code is written for research and educational purposes only to
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
from __future__ import absolute_import

try:
    xrange
except NameError:
    xrange = range

import math
import numpy as np
import scipy.linalg as spla
import random

import ws.hpo.choosers.gp_util as gp_util
from ws.hpo.choosers.acq_func import *
from ws.hpo.choosers.util import *

from ws.shared.logger import *
from ws.shared.resp_shape import *

def init(expt_dir, arg_string):
    args = unpack_args(arg_string)

    return GPChooser(expt_dir, **args)


"""
Chooser module for the Gaussian process expected improvement
acquisition function.  Candidates are sampled densely in the unit
hypercube and then the highest EI point is selected.  Slice sampling
is used to sample Gaussian process hyperparameters for the GP.
"""

class GPChooser:
    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=10,
                 max_obs=200, noiseless=False, 
                 response_shaping=False, shaping_func="hybrid_log", trade_off=0, v=1.0):
        
        self.cov_func = getattr(gp_util, covar)

        self.mcmc_iters = int(mcmc_iters)
        self.max_obs = max_obs
        self.D = -1
        self.hyper_iters = 1
        self.noiseless = bool(int(noiseless))
        self.trade_off = float(trade_off)
        self.v = float(v)
        
        # convert string 'True' or 'False' to boolean
        try:
            self.response_shaping = eval(response_shaping)
        except:
            self.response_shaping = False

        self.shaping_func = shaping_func

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale = 1  # zero-mean log normal prior
        self.max_ls = 2  # top-hat prior on length scales

        self.acq_funcs = ['EI', 'PI', 'UCB']

        self.mean_value = None
        self.estimates = None

        self.time_penalties = []

        self.est_eval_time = None

    def _real_init(self, dims, values):
        # print 'pkl not exists'
        # Input dimensionality.
        self.D = dims

        # Initial length scales.
        self.ls = np.ones(self.D)

        # Initial amplitude.
        self.amp2 = np.std(values) + 1e-4

        # Initial observation noise.
        self.noise = 1e-3

        # Initial mean.
        self.mean = np.mean(values)

    def cov(self, x1, x2=None):
        if x2 is None:
            # print self.amp2
            return self.amp2 * (self.cov_func(self.ls, x1, None)
                                + 1e-6 * np.eye(x1.shape[0]))
        else:
            return self.amp2 * self.cov_func(self.ls, x1, x2)

    def add_time_acq_funcs(self, time_penalties):
        if type(time_penalties) != list:
            time_penalties = [time_penalties]

        for time_penalty in time_penalties:
            self.time_penalties.append(time_penalty)

        for time_penalty in time_penalties:            
            self.acq_funcs.append(time_penalty['name'])

    def set_eval_time_penalty(self, est_eval_time):
        self.est_eval_time = est_eval_time

    def get_observations(self, samples, use_interim, sample_size):

        #completes = samples.get_completes(use_interim)
        comp = samples.get_grid("completes", use_interim)
        errs = samples.get_errors("completes", use_interim)

        size = len(errs)
        if size > sample_size:
            debug("Subsampling {} observations to make GP computational bound".format(sample_size))
            nc = []
            ne = []
            indices = np.random.choice(size, sample_size)
            for i in indices:
                nc.append(comp[i])
                ne.append(errs[i])
            return np.array(nc), np.array(ne)
        else:
            return comp, errs

    def next(self, samples, af, use_interim=True):
        
        comp_grid, errs = self.get_observations(samples, use_interim, self.max_obs)

        candidates = samples.get_candidates(use_interim)
        cand_grid = samples.get_grid("candidates", use_interim)
        
        # Don't bother using fancy GP stuff at first.
        if len(errs) < 2:
            return int(random.choice(candidates))

        # Perform the real initialization.
        if self.D == -1:
            num_dims = samples.get_grid_dim() # # of grid dimension
            self._real_init(num_dims, errs)
        
        if self.response_shaping is True:
            # transform errors to log10(errors) for enhancing optimization performance
            if self.shaping_func == "log_err":
                #debug("before scaling: {}".format(errs))
                errs = np.log10(errs)
                v_func = np.vectorize(apply_log_err)
                errs = v_func(errs)
            elif self.shaping_func == "hybrid_log":
                v_func = np.vectorize(apply_hybrid_log)
                errs = v_func(errs, threshold=.3)
            else:
                errs = np.log10(errs)
                
        #debug("errors: {}".format(errs))       

        # Sample from hyperparameters.
        af_values, mean_m, mean_v = self.do_modeling(comp_grid, cand_grid, errs, af)

        best_cand = np.argmax(af_values)
        self.mean_value = mean_m[best_cand]
        self.estimates = {
            'candidates' : candidates.tolist(),
            'acq_funcs' : af_values.tolist(),
            'means': mean_m.tolist(),
            'vars' : mean_v.tolist()
        }
        
        return int(candidates[best_cand])

    def do_modeling(self, comp, cand, errs, acq_func_type):

        # Sample from hyperparameters.
        afs = np.zeros((cand.shape[0], self.mcmc_iters))
        ms = np.zeros((cand.shape[0], self.mcmc_iters))
        vs = np.zeros((cand.shape[0], self.mcmc_iters))
        acq_func = None

        # If there are no pending, don't do anything fancy.

        # Current best.
        cur_best = np.min(errs)

        # The primary covariances for prediction.
        comp_cov = self.cov(comp)
        cand_cross = self.cov(comp, cand)

        # Compute the required Cholesky.
        obsv_cov = comp_cov + self.noise * np.eye(comp.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)

        # Solve the linear systems.
        alpha = spla.cho_solve((obsv_chol, True), errs - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)

        
        for i in xrange(self.mcmc_iters):
            self.sample_hypers(comp, errs)
            ms[:, i] = func_m
            vs[:, i] = func_v 
                   
            afs[:, i] = self.compute_acq_func(acq_func_type, 
                                              cur_best, func_m, func_v, 
                                              self.est_eval_time, 
                                              self.time_penalties)

        acq_funcs = np.mean(afs, axis=1)
        mean_m = np.mean(ms, axis=1)
        mean_v = np.mean(vs, axis=1)

        return acq_funcs, mean_m, mean_v

    def compute_acq_func(self, af, best, func_m, func_v, 
                                est_eval_time=None, time_penalties=[]):
        
        acq_func = get_acq_func(af)
        
        if est_eval_time is None:
            return acq_func(best, func_m, func_v)

        if af in [ tp['name'] for tp in self.time_penalties ]:

            penalty = (item for item in self.time_penalties if item["name"] == af).next()
            time_penalty_type = penalty['type']
            penalty_rate = 0.0
            if 'rate' in penalty.keys():
                penalty_rate = penalty['rate']
            
            try:
                af_vals = acq_func(best, func_m, func_v)
                return apply_eval_time_penalty(time_penalty_type, af_vals, 
                        est_eval_time, penalty_rate)

            except:
                warn("Exception occurs when acquires next candidate with {}".format(af))
                return acq_func(best, func_m, func_v)

    def sample_hypers(self, comp, vals):
        if self.noiseless:
            self.noise = 1e-3
            self._sample_noiseless(comp, vals)
        else:
            self._sample_noisy(comp, vals)
        self._sample_ls(comp, vals)

    def _sample_ls(self, comp, vals):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf

            cov = self.amp2 * (self.cov_func(ls, comp, None) + 1e-6 * np.eye(comp.shape[0])) + self.noise * np.eye(
                comp.shape[0])
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - self.mean, solve)
            return lp

        self.ls = slice_sample(self.ls, logprob, compwise=True)

    def _sample_noisy(self, comp, vals):
        def logprob(hypers):
            mean = hypers[0]
            amp2 = hypers[1]
            noise = hypers[2]

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            cov = amp2 * (self.cov_func(self.ls, comp, None) +
                          1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0])
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.noise_scale / noise) ** 2))

            # Roll in amplitude lognormal prior
            lp -= 0.5 * (np.log(amp2) / self.amp2_scale) ** 2

            return lp

        hypers = slice_sample(np.array([self.mean, self.amp2, self.noise]),
                                   logprob, compwise=False)
        self.mean = hypers[0]
        self.amp2 = hypers[1]
        self.noise = hypers[2]

    def _sample_noiseless(self, comp, vals):
        def logprob(hypers):
            mean = hypers[0]
            amp2 = hypers[1]
            noise = 1e-3

            if amp2 < 0:
                return -np.inf

            cov = amp2 * (self.cov_func(self.ls, comp, None) +
                          1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0])
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)

            # Roll in amplitude lognormal prior
            lp -= 0.5 * (np.log(amp2) / self.amp2_scale) ** 2

            return lp

        hypers = slice_sample(np.array([self.mean, self.amp2, self.noise]), logprob,
                                   compwise=False)
        self.mean = hypers[0]
        self.amp2 = hypers[1]
        self.noise = 1e-3

    def optimize_hypers(self, comp, vals):
        my_gp = gp_util.GaussianProcess(self.cov_func.__name__)
        my_gp.real_init(comp.shape[1], vals)
        my_gp.optimize_hypers(comp, vals)
        self.mean = my_gp.mean
        self.ls = my_gp.ls
        self.amp2 = my_gp.amp2
        self.noise = my_gp.noise

        return




