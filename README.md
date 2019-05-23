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

For more information, See [Wiki](https://github.com/snu-adsl/DEEP-BO/wiki)




## Prallel BO with diversification 
For scalable parallel HPO, the hyperparameter optimization algorithm itself also can be a node like hpo_node.py.
We also introduce the name node which manages the shared history and associated parallel workers.
In this manner, we can scaling Bayesian Optimization as described in the paper.
(Details will be updated soon) 


