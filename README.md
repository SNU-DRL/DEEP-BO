# DEEP-BO for Hyperparameter Optimization of Deep Networks

This repo contains code accompaning the paper, [DEEP-BO for Hyperparameter Optimization of Deep Networks](https://arxiv.org/abs/1905.09680)  

## Abstract

The performance of deep neural networks (DNN) is very sensitive to the particular choice of hyperparameters. To make it worse, the shape of the learning curve can be significantly affected when a technique like batchnorm is used. As a result, hyperparameter optimization of deep networks can be much more challenging than traditional machine learning models. In this work, we start from well known Bayesian Optimization solutions and provide enhancement strategies specifically designed for hyperparameter optimization of deep networks. 
The resulting algorithm is named as DEEP-BO (Diversified, Early-termination-Enabled, and Parallel Bayesian Optimization). When evaluated over six DNN benchmarks, DEEP-BO easily outperforms or shows comparable performance as some of the well-known solutions including GP-Hedge, Hyperband, BOHB, Median Stopping Rule, and Learning Curve Extrapolation.


## Citation

If you are use of any material in this repository, we ask to cite:

```

@article{cho2019deep,
  title={DEEP-BO for Hyperparameter Optimization of Deep Networks},
  author={Cho, Hyunghun and Kim, Yongjin and Lee, Eunjung and Choi, Daeyoung and Lee, Yongjae and Rhee, Wonjong},
  journal={arXiv preprint arXiv:1905.09680},
  year={2019}
}

```

-------

## Installation

### Prerequisite

While this project supports on both python 2.x and python 3.x, but it has been tested in Python 2.7.x on Linux.
Before installing it with your existing project, we strongly suggest that creating new virtual environment is preferred by using [Anaconda](https://www.anaconda.com/download/).
Even though the maintenance period for python 2.x is over, we still recommend creating this environment based on python 2.x.

```bash
    conda create -n hpo python=2.7
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

It is better to install using conda than pip to avoid dependency conflicts but conda manges some limitted packages.
Therefore, if you are working in a conda environment as I suggested, install them seperately as follows:

```bash
(hpo)device:path$ pip install future validators flask-restful requests httplib2 pyYAML
(hpo)device:path$ conda install pandas numpy scipy scikit-learn
(hpo)device:path$ pip install hyperopt pyDOE
```

Optionally, you can speed up the gradient calculation in GP when installing the weave package as follows:
```bash
(hpo)device:path$ conda install -c conda-forge weave
```

### Deep learning packages for sample code

If you want to run a sample code, following deep learning packages are required:

* [tensorflow(-gpu)](https://www.tensorflow.org/install)
* [keras(-gpu)](https://keras.io/#installation)

In case of CPU only, you can install above packages as follows:

```bash
(hpo)device:path$ conda install tensorflow=1.15 keras
```

See the other details in the [guide](https://github.com/snu-adsl/DEEP-BO/wiki). 


