from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ws.shared.logger import *

def get_simulator(space, run_config):
    
    etr = None
    if run_config and "early_term_rule" in run_config:
        etr = run_config["early_term_rule"]

    expired_time = None
    if run_config and "warm_up_time" in run_config:
        expired_time = run_config["warm_up_time"]

    from ws.hpo.trainers.emul.trainer import TrainEmulator
    from ws.hpo.trainers.emul.grad_etr import GradientETRTrainer
    from ws.hpo.trainers.emul.median_etr import VizMedianETRTrainer 
    from ws.hpo.trainers.emul.threshold_etr import ThresholdingETRTrainer 
    from ws.hpo.trainers.emul.threshold_etr import MultiThresholdingETRTrainer     
    from ws.hpo.trainers.emul.knock_etr import KnockETRTrainer
    from ws.hpo.trainers.emul.interval_etr import IntervalETRTrainer
    from ws.hpo.trainers.emul.hybrid_etr import HybridETRTrainer
    from ws.hpo.trainers.emul.kickstart_etr import KickStarterETRTrainer
    from ws.hpo.trainers.emul.curve_predict_etr import CurvePredictETRTrainer

    try:
        if not hasattr(space, 'lookup'):
            raise ValueError("Invalid surrogate space")
        lookup = space.lookup
                
        if etr == None or etr == "None":
            return TrainEmulator(lookup)
        elif etr == "DecaTercet":
            return MultiThresholdingETRTrainer(lookup, 0.1)
        elif etr == "PentaTercet":
            return MultiThresholdingETRTrainer(lookup, 0.2) 
        elif etr == "TetraTercet":
            return MultiThresholdingETRTrainer(lookup, 0.25)
        elif etr == "VizMedian":
            return VizMedianETRTrainer(lookup)
        elif etr == "VizPenta":
            return ThresholdingETRTrainer(lookup, 0.2)
        elif etr == "VizPentaOpt":
            return ThresholdingETRTrainer(lookup, 0.2, 
                                            eval_end_ratio=0.85)        
        elif etr == "Knock":
            return KnockETRTrainer(lookup)
        elif etr == "Interval":
            return IntervalETRTrainer(lookup)
        elif etr == "IntervalPentaOpt":
            return HybridETRTrainer(lookup)            
        elif etr == "KickStarter":
            return KickStarterETRTrainer(lookup, expired_time=expired_time)
        
        elif etr == 'ExtrapolaFantasy':
            return CurvePredictETRTrainer(lookup)
        elif etr == 'Extrapola':
            return CurvePredictETRTrainer(lookup, use_fantasy=False)
        else:
            debug("Invalid ETR: {}".format(etr))
            return TrainEmulator(lookup)

    except Exception as ex:
        warn("Early termination trainer creation failed: {}".format(ex))
        return TrainEmulator(lookup)


def get_remote_trainer(rtc, space, run_config):
    from ws.hpo.trainers.remote.trainer import RemoteTrainer    
    from ws.hpo.trainers.remote.grad_etr import GradientETRTrainer
    from ws.hpo.trainers.remote.threshold_etr import MultiThresholdingETRTrainer

    try:
        if run_config and "early_term_rule" in run_config:
            etr = run_config["early_term_rule"]
            if etr == 'Gradient':
                return GradientETRTrainer(rtc, space, **run_config)
            elif etr == "DecaTercet":
                return MultiThresholdingETRTrainer(rtc, space, 0.1, **run_config)
            elif etr == "PentaTercet":
                return MultiThresholdingETRTrainer(rtc, space, 0.2, **run_config) 
            elif etr == "TetraTercet":
                return MultiThresholdingETRTrainer(rtc, space, 0.25, **run_config)                
            else:
                debug("Invalid ETR: {}".format(etr))
                return RemoteTrainer(rtc, space, **run_config)            
        else:
            return RemoteTrainer(rtc, space, **run_config)

    except Exception as ex:
        warn("Remote trainer creation failed: {}".format(ex))
        raise ValueError("Invalid trainer implementation")

