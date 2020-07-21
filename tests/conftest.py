import pytest
import numpy as np
import torch
from torch.distributions.binomial import Binomial
if torch.cuda.is_available():
    import torch.cuda as t
else:
    import torch as t


@pytest.fixture
def data_params():
    """
    Returns random data and parameters used to generate the data.
    """
    S = int(1e5)
    theta_1 = 0.5
    theta_2 = 0.3
    pi_1 = 0.6
    pi_2 = 1 - pi_1
    N_ls_all = t.FloatTensor([100 for _ in range(S)])
    theta_ls = t.FloatTensor(np.random.choice([theta_1, theta_2],
                                               size=S, p=[pi_1, pi_2]))
    n_ls_all = Binomial(N_ls_all, theta_ls).sample()
    return N_ls_all, n_ls_all, [pi_1, pi_2, theta_1, theta_2]
