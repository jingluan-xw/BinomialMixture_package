import pytest
import numpy as np
from binomial_mixture_model.BinomialMixture import BinomialMixture


def test_BMM(data_params):
    """
    Tests the class BinomialMixture
    """
    N_ls, n_ls, params = data_params
    BM = BinomialMixture(n_components=2, tolerance=1e-6,
                         max_step=int(5e4), verbose=False,
                         random_state=123)
    BM.fit(N_ls, n_ls)
    pi_array = BM.pi_list.numpy()
    theta_array = BM.theta_list.numpy()
    assert np.abs(np.sum(pi_array)-1) < 1e-3, (
        f"Sum of pi is not equal to unity."
    )
    assert np.max(theta_array) < 1 and np.min(theta_array) > 0,(
        f"theta out of range."
    )
