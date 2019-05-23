import math

##
# Functions for response shaping

LOG_ERR_LOWER_BOUND = -5.0

def apply_no_shaping(x):
    return x


def apply_log_err(x):    

    if x < LOG_ERR_LOWER_BOUND:
        x = LOG_ERR_LOWER_BOUND
    
    scale_x = (x - LOG_ERR_LOWER_BOUND) / abs(LOG_ERR_LOWER_BOUND)
    return scale_x


def apply_hybrid_log(err, threshold=0.3, err_lower_bound=0.00001):    
    log_th = math.log10(threshold)
    beta = threshold - log_th

    if err > threshold:
        return err  # linear scale
    else:
        if err > 0:
            log_applied = math.log10(err)
        else:
            log_applied = math.log10(err_lower_bound)
        return  log_applied + beta # log scale