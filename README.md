# DEEP-BO for Hyperparameter Optimization of Deep Networks

## Abstract

The performance of deep neural networks (DNN) is very sensitive to the particular choice of hyperparameters. To make it worse, the shape of the learning curve can be significantly affected when a technique like batchnorm is used. As a result, hyperparameter optimization of deep networks can be much more challenging than traditional machine learning models. In this work, we start from well known Bayesian Optimization solutions and provide enhancement strategies specifically designed for hyperparameter optimization of deep networks. 
The resulting algorithm is named as DEEP-BO (Diversified, Early-termination-Enabled, and Parallel Bayesian Optimization). When evaluated over six DNN benchmarks, DEEP-BO easily outperforms or shows comparable performance as some of the well-known solutions including GP-Hedge, Hyperband, BOHB, Median Stopping Rule, and Learning Curve Extrapolation.


## DEEP-BO Overview
**Key Features**
  * Expandable structure which can easily add new HPO algorithm
  * Noble diversification strategy for effective and robust HPO
  * Compound early termination rule.  
  * Scalable and asynchronous HPO jobs on a microservices architecture
    * Thanks to Web Oriented Architecture via RESTful Web API

We unified three practical hyperparameter optimization frameworks of machine learning:

* [Spearmint](https://github.com/JasperSnoek/spearmint) 
* [SMAC](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/)
* [Hyperopt](https://github.com/hyperopt/hyperopt)

Our solution diversify BO algorithms to minimize a catastrophic failure which happens to be on any DNN problem by rotating combinations of their modeling algorithms and acquisition functions.

## Installation

### Prerequisites

I strongly suggest to make new virtual environment by using [Anaconda](https://www.anaconda.com/download/).
This project only tested on Python 2.7.x.

```bash
    conda create -n hpo python=2.7
```

After creating the environment, activate your environment as follows:

```bash
    source activate hpo
```

The following additional packages are required to install:

* pandas
* scikit-learn
* future
* hyperopt
* validators
* weave
* flask-restful
* requests


If you are working on Linux, install these packages as follows:

```bash

(hpo)device:path$ conda install pandas scikit-learn future numpy scipy
(hpo)device:path$ pip install hyperopt validators flask-restful requests
```

(Optional) Speeding up the gradient calculation in GP, install below package:
```bash
(hpo)device:path$ conda install -c conda-forge weave
```

If you want to evalute samples, more packages for deep learning are required:

* tensorflow(-gpu)
* karas(-gpu)

-------------------

## Runner Interface

You can run hyperparameter optimization through hpo_runner.py.
See help for more options as below:

```bash
(hpo)$ python hpo_runner.py --help
usage: hpo_runner.py [-h] [-m MODE] [-s SPEC] [-e EXP_CRT] [-eg EXP_GOAL]
                     [-et EXP_TIME] [-etr EARLY_TERM_RULE] [-rd RCONF_DIR]
                     [-hd HCONF_DIR] [-rc RCONF] [-w WORKER_URL]
                     [-l LOG_LEVEL] [-r RERUN] [-p PKL]
                     hp_config num_trials

positional arguments:
  hp_config             hyperparameter configuration name.
  num_trials            The total repeats of the experiment.

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  The optimization mode. Set a model name to use a
                        specific model only. Set DIV to sequential
                        diverification mode. Set BATCH to parallel mode.
                        ['SOBOL', 'GP', 'RF', 'TPE', 'GP-NM', 'GP-HLE',
                        'RF-HLE', 'TPE-HLE', 'DIV', 'ADA', 'BATCH'] are available.
                        default is DIV.
  -s SPEC, --spec SPEC  The detailed specification of the given mode. (e.g.
                        acquisition function) ['RANDOM', 'EI', 'PI',
                        'UCB', 'SEQ', 'RANDOM', 'HEDGE', 'BO-HEDGE', 'BO-
                        HEDGE-T', 'BO-HEDGE-LE', 'BO-HEDGE-LET', 'EG', 'EG-
                        LE', 'GT', 'GT-LE', 'SKO', 'SYNC', 'ASYNC'] are
                        available. default is SEQ.
  -e EXP_CRT, --exp_crt EXP_CRT
                        Expiry criteria of the trial. Set "TIME" to run each
                        trial until given exp_time expired.Or Set "GOAL" to
                        run until each trial achieves exp_goal.Default setting
                        is TIME.
  -eg EXP_GOAL, --exp_goal EXP_GOAL
                        The expected target goal accuracy. When it is
                        achieved, the trial will be terminated automatically.
                        Default setting is 0.9999.
  -et EXP_TIME, --exp_time EXP_TIME
                        The time each trial expires. When the time is up, it
                        is automatically terminated. Default setting is 86400.
  -etr EARLY_TERM_RULE, --early_term_rule EARLY_TERM_RULE
                        Early termination rule. Default setting is None.
  -rd RCONF_DIR, --rconf_dir RCONF_DIR
                        Run configuration directory. Default setting is
                        ./run_conf/
  -hd HCONF_DIR, --hconf_dir HCONF_DIR
                        Hyperparameter configuration directory. Default
                        setting is ./hp_conf/
  -rc RCONF, --rconf RCONF
                        Run configuration file in ./run_conf/. Default setting
                        is arms.json
  -w WORKER_URL, --worker_url WORKER_URL
                        Remote training worker node URL. Set the valid URL if
                        remote training required.
  -l LOG_LEVEL, --log_level LOG_LEVEL
                        Print out log level. ['debug', 'warn', 'error', 'log']
                        are available. default is warn

```

## Run Hyperparameter Optimization through Web API

For both easy of evaluation and scalability, we adapt Web-Oriented Architecture (WOA).
Firstly, we create a train node which will be served as a daemon like a sample code which described as train_node.py.
HPO runner can communicate with the train node through pre-defined Web API.

For run with a single processor, you first run a train node. (The practical samples are existed in /samples directory)
If the training node is ready to serve, you will get the URL(including port number if required).
Pass this URL through -w option of hpo_runner.

If the URL is http://localhost:6000, run script as follows: 
```bash
(hpo)$ python hpo_runner.py -e=TIME -et=36000 -w http://localhost:6000 MNIST-LeNet1 1 
```

**CAUTIONS**: configuration files (e.g. /hp_conf/*.json, /run_conf/*.json) are properly set.
(Details will be updated soon) 


## Prallel BO with diversification 
For scalable parallel HPO, the hyperparameter optimization algorithm itself also can be a node like hpo_node.py.
We also introduce the name node which manages the shared history and associated parallel workers.
In this manner, we can scaling Bayesian Optimization as described in the paper.
(Details will be updated soon) 


