# DEEP-BO for Hyperparameter Optimization of Deep Networks

This repo contains code accompaning the paper, [Basic Enhancement Strategies When Using Bayesian Optimization for Hyperparameter Tuning of Deep Neural Networks](https://ieeexplore.ieee.org/document/9037259/)  

![Graphical Abstract](https://github.com/snu-adsl/DEEP-BO/blob/master/GA.png)

## Abstract

Compared to the traditional machine learning models, deep neural networks (DNN) are known to be highly sensitive to the choice of hyperparameters. While the required time and effort for manual tuning has been rapidly decreasing for the well developed and commonly used DNN architectures, undoubtedly DNN hyperparameter optimization will continue to be a major burden whenever a new DNN architecture needs to be designed, a new task needs to be solved, a new dataset needs to be addressed, or an existing DNN needs to be improved further. For hyperparameter optimization of general machine learning problems, numerous automated solutions have been developed where some of the most popular solutions are based on Bayesian Optimization (BO). In this work, we analyze four fundamental strategies for enhancing BO when it is used for DNN hyperparameter optimization. Specifically, diversification, early termination, parallelization, and cost function transformation are investigated. Based on the analysis, we provide a simple yet robust algorithm for DNN hyperparameter optimization - DEEP-BO (Diversified, Early-termination-Enabled, and Parallel Bayesian Optimization). When evaluated over six DNN benchmarks, DEEP-BO mostly outperformed well-known solutions including GP-Hedge, BOHB, and the speed-up variants that use Median Stopping Rule or Learning Curve Extrapolation. In fact, DEEP-BO consistently provided the top, or at least close to the top, performance over all the benchmark types that we have tested. This indicates that DEEP-BO is a robust solution compared to the existing solutions.


## Citation

If you are use of any material in this repository, we ask to cite:

```

@ARTICLE{9037259, 
author={H. {Cho} and Y. {Kim} and E. {Lee} and D. {Choi} and Y. {Lee} and W. {Rhee}}, 
journal={IEEE Access}, 
title={Basic Enhancement Strategies When Using Bayesian Optimization for Hyperparameter Tuning of Deep Neural Networks}, 
year={2020}, volume={8}, number={}, pages={52588-52608},}
```

-------

## Installation

### Prerequisite

While this project had been developed to support both python 2.x and python 3.x, the use of python 3.x is preferred because the maintenance period for python 2.x is over.
Also, this project has been tested on both Linux and Windows. 
If you want to run on Windows, we strongly suggest to use Linux Subsystem for Linux.
Before installing it with your existing project, we strongly suggest that creating new virtual environment is preferred by using [Anaconda](https://www.anaconda.com/download/).

```bash
    conda create -n hpo python=3.6
```

After creating the above environment, activate this environment as follows:

```bash
    source activate hpo
```

### Package Dependencies

DEEP-BO requires the following packages:

* For any thinking node (basic requirements)
  * future
  * flask-restful
  * requests
  * httplib2
  * validators
  * pyYAML

* For DEEP-BO (especially for tuner node and HPO runner)
  * pandas
  * scikit-learn
  * numpy
  * scipy
  * hyperopt
  * pyDOE

* For visualization (for web viewers)
  * matplotlib
  * jupyter notebook
  * hiplot
It is better to install using conda than pip to avoid dependency conflicts but conda manges some limitted packages.
Therefore, if you are working in a conda environment as I suggested, install them seperately as follows:

```bash
(hpo)device:path$ pip install future validators flask-restful requests httplib2 pyYAML
(hpo)device:path$ conda install -c conda-forge pandas numpy scipy scikit-learn notebook
(hpo)device:path$ pip install hyperopt pyDOE matplotlib hiplot
```

When python2 is used, the gradient calculation in GP can be sped up when installing the weave package as follows:
```bash
(hpo)device:path$ conda install -c conda-forge weave
```

### Deep learning packages for sample code

If you want to optimize hyperparameters in a [sample code](github.com/snu-adsl/DEEP-BO/tree/master/samples), following deep learning packages are required respectively:
* For keras implementations
  * [tensorflow(-gpu)](https://www.tensorflow.org/install)
  * [keras(-gpu)](https://keras.io/#installation)
* For pytorch implementation
  * [pytorch, , torchvision](pytorch.org/get-started/locally/?source=Google&medium=PaidSearch&utm_campaign=1711784041&utm_adgroup=68039908078&utm_keyword=install%20pytorch&utm_offering=AI&utm_Product=PYTorch&gclid=CjwKCAjwvtX0BRAFEiwAGWJyZGmAYDWXuZMLuN-G2qFYvEb428fr3uOh5yaR2WAKgCZEe5nCke1f0BoCm08QAvD_BwE)
  * mlconfig, tqdm 


See the other details in the [guide](https://github.com/snu-adsl/DEEP-BO/wiki). 


