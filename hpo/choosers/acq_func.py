import numpy as np
import scipy.stats as sps


def get_acq_func(af):
    if af == 'EI':
        acq_func = compute_ei
            
    elif af == 'PI':
        acq_func = compute_pi

    elif af =='UCB':
        acq_func = compute_ucb
    else:
        raise ValueError('Unsupported acquisition function type: {}'.format(af))

    return acq_func


def compute_ei(best, func_m, func_v, trade_off=0, epsilon=0.0001):

    # Expected improvement
    func_s = np.sqrt(func_v) + epsilon
    u = (best - func_m - trade_off) / func_s
    ncdf = sps.norm.cdf(u)
    npdf = sps.norm.pdf(u)
    ei = func_s * (u * ncdf + npdf)

    return ei


def compute_pi(best, func_m, func_v, trade_off=0, epsilon=0.0001):

    # Probability of improvement
    func_s = np.sqrt(func_v) + epsilon
    u = (best - func_m - trade_off) / func_s

    pi = sps.norm.cdf(u)

    return pi


def compute_ucb(best, func_m, func_v, trade_off=0, epsilon=0.0001):
    
    func_s = np.sqrt(func_v) + epsilon

    ucb = -func_m + func_s
    return ucb 

##
# Evaluation time penalty functions for acquisition function 


def apply_eval_time_penalty(time_penalty, af_vals, est_eval_time, penalty_rate):

    if time_penalty == None or time_penalty == "None":
        return af_vals
    elif time_penalty == 'linear':
        return linear_penalty(af_vals, est_eval_time, penalty_rate)
    elif time_penalty == 'top_k':
        return top_k_penalty(af_vals, est_eval_time, penalty_rate)
    elif time_penalty == 'per_second':
        return per_second_penalty(af_vals, est_eval_time)
    elif time_penalty == 'per_log_second':
        return per_log_second_penalty(af_vals, est_eval_time)
    else:
        raise ValueError("unsupported penalty methods: {}".format(time_penalty))

def linear_penalty(af_vals, et, alpha):
    
    if np.min(et) < 0:
        et = et - np.min(et)

    if np.min(af_vals) < 0:
        af_vals = af_vals - np.min(af_vals)

    et = et / np.max(et)
    af_vals = af_vals / np.max(af_vals)
    new_af_vals = af_vals - alpha * et

    return new_af_vals


def top_k_penalty(af_vals, et, top_k):

    if np.min(af_vals) < 0:
        af_vals = af_vals - np.min(af_vals)

    af_vals = af_vals / np.max(af_vals)

    new_af_vals = af_vals
    idxs = af_vals.argsort()[-top_k:][::-1]
    et_20 = et[idxs]
    et_min_idx = np.where(et == np.min(et_20))[0]
    new_af_vals[et_min_idx] = 10

    return new_af_vals


def per_second_penalty(af_vals, et):

    if np.min(et) < 0:
        et = et - np.min(et)

    et = et + 10
    new_af_vals = af_vals / et

    return new_af_vals


def per_log_second_penalty(af_vals, et):

    if np.min(et) < 0:
        et = et - np.min(et)

    et = et + 10
    new_af_vals = af_vals / np.log(et)

    return new_af_vals