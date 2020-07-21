"""
In this file, we define the following functions:

1) A function that applies K-fold cross validation in order to evaluate
the performances of the BinomialMixture model with different values for
n_components. Here n_components refers to the number of binomial distributions
in the model.

2) A function that plots the performance metric, which is the log_likelihood,
versus n_components.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from binomial_mixture_model.BinomialMixture import BinomialMixture


def cv_n_components(N_ls_all, n_ls_all, K_list, n_splits=10, random_state=None):
    """
    Evaluates the performance of BinomialMixture with Cross Validation.

    Input
    -----
    N_ls_all: 1D array.
              A list of total numbers of experiements. This list would be
              splitted into train and validation sets by cross validation.
    n_ls_all: 1D array.
              The list of numbers of successful experiments corresponding to
              N_ls_all. This list would be splitted into train and Validation
              sets by the same fashion as that for N_ls_all.
    K_list: A list of integers. Its length is num_K.
            The list of values of n_components to be evaluated by cross
            validation.
    n_splits: int.
              The number of folds for the cross validation.
    random_state: int. Default value is None.
              For reproducibility. It is used for both the KFold Cross
              Validation and the initialization of parameters for the
              BinomialMixture Model.
    Output
    ------
    metric_list: A list of lists. Its shape is (num_K, 2).
                 Each inner list contains two numbers. The first one is the mean
                 log_likelihood (metric) for the n_splits training sets.
                 The second one is the standard deviation of the log_likelihood
                 (metric) for the n_splits training sets.
    metric_val_list: A list of lists. Its shape is (num_K, 2).
                 The same as metric_list except that this list is for the
                 validation sets.
    """
    N_ls_all = t.FloatTensor(N_ls_all)
    n_ls_all = t.FloatTensor(n_ls_all)
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    metric_list = []
    metric_val_list = []
    for K in K_list:
        metric_kfold = []
        metric_val_kfold = []
        for train_ndx, val_ndx in kf.split(N_ls_all):
            # Assign train and validation sets.
            N_ls = N_ls_all[train_ndx]
            n_ls = n_ls_all[train_ndx]
            N_ls_valid = N_ls_all[val_ndx]
            n_ls_valid = n_ls_all[val_ndx]
            # Sizes
            S = len(N_ls)
            S_val = len(N_ls_valid)
            # Fit BMM.
            BM = BinomialMixture(n_components=K, tolerance=1e-6,
                                 max_step=int(5e4), verbose=False,
                                 random_state=random_state)
            BM.fit(N_ls, n_ls)
            # Calculate the metric.
            log_likelihood = BM.calc_logL(N_ls, n_ls)
            log_likelihood_val = BM.calc_logL(N_ls_valid, n_ls_valid)
            metric_kfold.append(-log_likelihood.item()/S)
            metric_val_kfold.append(-log_likelihood_val.item()/S_val)
        # Metric for the training data.
        metric_mean = np.mean(metric_kfold)
        metric_std = np.std(metric_kfold)
        metric_list.append([metric_mean, metric_std])
        # Metric for the validation data.
        metric_mean_val = np.mean(metric_val_kfold)
        metric_std_val = np.std(metric_val_kfold)
        metric_val_list.append([metric_mean_val, metric_std_val])
    return metric_list, metric_val_list


# metric_val_array = np.array(metric_val_list)
# metric_array = np.array(metric_list)
# fig = plt.figure(figsize=(8,5))
# plt.errorbar(K_list, metric_val_array[:,0], yerr=metric_val_array[:,1], label="validation",
#              fmt="o")
# plt.errorbar(K_list, metric_array[:,0], yerr=metric_array[:,1], label="train")
# plt.xlabel("K", size=14)
# plt.ylabel("Metric", size=14)
# plt.tick_params(labelsize=14)
# plt.legend(loc='upper right', fontsize=14)
# plt.title("5-Fold Cross Validation", fontsize=16)
# plt.show()
