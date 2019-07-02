# DEEP-BO for Hyperparameter Optimization of Deep Networks

This repo contains code accompaning the paper, [DEEP-BO for Hyperparameter Optimization of Deep Networks](https://arxiv.org/abs/1905.09680)  

## Abstract

The performance of deep neural networks (DNN) is very sensitive to the particular choice of hyperparameters. To make it worse, the shape of the learning curve can be significantly affected when a technique like batchnorm is used. As a result, hyperparameter optimization of deep networks can be much more challenging than traditional machine learning models. In this work, we start from well known Bayesian Optimization solutions and provide enhancement strategies specifically designed for hyperparameter optimization of deep networks. 
The resulting algorithm is named as DEEP-BO (Diversified, Early-termination-Enabled, and Parallel Bayesian Optimization). When evaluated over six DNN benchmarks, DEEP-BO easily outperforms or shows comparable performance as some of the well-known solutions including GP-Hedge, Hyperband, BOHB, Median Stopping Rule, and Learning Curve Extrapolation.


## DEEP-BO Overview
**Key Features**
  * Simple diversification strategy over BO modeling algorithms and acquisition functions
  * Scalable BO executing HPO jobs asynchronously based on a microservices architecture
    * Thanks to Web Oriented Architecture via RESTful Web API
  * Compound early termination rule to stop unuseful evaluation(s) without harmful side-effect.  

We have unified three practical hyperparameter optimization frameworks of machine learning:

* [Spearmint](https://github.com/JasperSnoek/spearmint) 
* [SMAC](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/)
* [Hyperopt](https://github.com/hyperopt/hyperopt)


## Installation

### Prerequisites

Firstly, creating new virtual environment by using [Anaconda](https://www.anaconda.com/download/) is strongly suggested.
Even though the maintenance period for python 2.7.x is almost over, we still recommend creating this environment based on python 2.7.x.
Running in python 3.x may be supported soon, but so far this project has only been tested in Python 2.7.x.

```bash
    conda create -n hpo python=2.7
```

After creating the environment, activate your environment as follows:

```bash
    source activate hpo
```

The following additional packages are required to install properly:

* pandas
* scikit-learn
* future
* hyperopt
* validators
* weave
* flask-restful
* requests
* httplib2

If you are working on Linux, install these packages as follows:

```bash

(hpo)device:path$ conda install pandas scikit-learn future numpy scipy
(hpo)device:path$ pip install hyperopt validators flask-restful requests
```

(Optional) Speeding up the gradient calculation in GP, install weave package as follows:
```bash
(hpo)device:path$ conda install -c conda-forge weave
```

If you want to run a sample, following deep learning packages may be required additionally:

* tensorflow(-gpu)
* karas(-gpu)
* [pytorch](https://pytorch.org/get-started/locally/)

In case of no GPU available, you can install above packages as follows:
```bash
(hpo)device:path$ conda install tensorflow karas 
(hpo)device:path$ conda install pytorch-cpu torchvision-cpu -c pytorch
```

-------------------

For more information, Refer to [Wiki](https://github.com/snu-adsl/DEEP-BO/wiki).

