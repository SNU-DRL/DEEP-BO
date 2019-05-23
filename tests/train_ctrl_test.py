import os
import sys
import time

# For path arrangement (set the parent directory as the root folder)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import ws.shared.hp_cfg as hp_cfg

from ws.shared.logger import *
from ws.hpo.connectors.remote_train import RemoteTrainConnector

def test_train_main():
    # XXX: You should start add_task.py of worker first.

    set_log_level('debug')
    json_cfg = {
        "dataset": "integers",
        'model': "add", 
        "hyperparams": {
            "arg1": {
                "type": "int",
                "value_type": "discrete",
                "range": [
                    1,
                    1000
                ]
            },
            "arg2": {
                "type": "int",
                "value_type": "discrete",
                "range": [
                    1,
                    1000
                ]
            }
        },
        "param_order" : ["arg1", "arg2"],
        "config": {}
    }
    cfg = hp_cfg.HyperparameterConfiguration(json_cfg)

    base_url = 'http://127.0.0.1:5000'
    rtc = RemoteTrainConnector(base_url, cfg)
    if rtc.validate():
        hpv = {"arg1": 10, "arg2": 10}
        job_id = rtc.create_job(hpv)
        if job_id is not None:
            print("Job {} created successfully.".format(job_id))
            if rtc.start(job_id):
                print("Job started")
                
                # Wait until job is done.
                while rtc.get_job("active") != None:
                    time.sleep(1)
                result = rtc.get_job(job_id)
                print("Return of job {} is {}.".format(job_id, result['cur_loss']))
            else:
                print("Starting a training job failed.")
        else:
            print("Creating job failed")
                


if __name__ == '__main__':    
    test_train_main()
