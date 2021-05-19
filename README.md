# Binimial Mixture Model wrapped in a Package with Tests

## Introduction

In this repository, I wrap the Binomial Mixture Model in a package that I developed in my
other (private) repository [Binomial-Mixture-Model-with-EM-Algorithm](https://github.com/jingluan-xw/Binomial-Mixture-Model-with-EM-Algorithm). Details about this model, the step-by-step development,
etc can be found in that repository and links therein. The mathematical details about this model is explained in [this article](https://jingluan-xw.medium.com/binomial-mixture-model-with-expectation-maximum-em-algorithm-feeaf0598b60). Please contact me if you like to get access to that
repository.

This repository includes the following
subdirectories and files.

* binomial_mixture_model/ contains a python script named "BinomialMixture.py" in which the `BinomialMixture` class is defined. A detailed description about its parameters, attributes and methods can be found in the end. The other python script named "number_components.py" defines
a function `cv_n_components` which allows a user to try different values for the number of binomial components in the BinomialMixture model and uses cross validation to compare the resulted log-likelihood on the validation data set.


* tests/ contains a file named "test_BMM.py" for testing the `BinomialMixture` model. The input to this test function is defined in the file "conftest.py". It is by no means a thorough test for the model. As I say earlier, this is just meant for a practice of package setup and pytests for myself.

* "setup.py" contains information about the package including its name (="binomial_mixture_model") version, author, contact, etc.

## Author  
Jing Luan -- jingluan.xw _at_ gmail

## Prerequisites

* pytorch
* pytest
* setuptools
* scipy
* numpy
* scikit-learn
* matplotlib

## Installation

Once again, this is not a fully tested or well established package yet. However, if you strongly intend to use it and test it for your own application, you are very welcome to do so. To install this package, you need to download the whole repository into a local directory on your machine, then navigate to that directory in a terminal and run

* pip install .

Note the period "." above refers to the local directory which asks "pip" to install whatever package available in the local directory. After installing the package, you may use the class "BinomialMixture" by "from binomial_mixture_model.BinomialMixture import BinomialMixture". You may use the function "cv_n_components" by "from binomial_mixture_model.number_components import cv_n_components".

Anytime you like to uninstall the package. Just do

* pip uninstall binomial_mixture_model

## Acknowledgement

Jing Luan is an IAS fellow supported by the Institute for Advanced Study while she develops this package.

## BinomialMixture

_class_ **BinomialMixture** _(n_components, tolerance=1e-5, max_step=1000, verbose=True, random_state=None)_

Binomial Mixture: Representation of a mixture of one or more than one different binomial distributions. The class allows to estimate the parameters of a Binomial mixture distribution.

### Parameters:

* __n_components: int__: the number of mixture components.
* __tolerance: float, default 1e-5__: one of the conditions for stopping the Expectation-Maximization iteration. This condition says that when the log-likelihood changes by less or equal to `tolerance` per iteration, the iteration can be stopped.
* __max_step: int, default 1000__: the other condition for stopping the Expectation-Maximization iterations. This condition says that when the number of iterations exceed `max_step` the iteration is stopped.
* __verbose: bool, default True__: prints the intermediate fitting results during the iteration.
* __random_state: int, default None__: the random seed used to initialize the parameters in the model.

### Attributes:

* __pi_list: array like, shape(n_components, )__: the weights (prevalences) of each mixture component.
* __theta_list: array like, shape(n_components, )__: the binomial probabilities of each binomial component.

### Methods:

* __fit(X[,y])__: estimate model parameters with the Expectation-Maximization algorithm.
* __predict(X)__: predict the posterior probabilities for each sample belonging to each mixture component.
* __calc_AIC_BIC(X)__: calculate the Akaike Information Criterion and the Bayesian Information Criterion.
* __p_value(X, side="right")__: after a binomial mixture model is fitted, this method calculates the p values of all samples (either training set or test set). The argument `side`, if set to "right" returns right-sided p-values, if set to "left" returns left-side p-values, if set to `both` returns double-sided p-values.
