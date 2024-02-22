# The code is designed to work with either PyTorch tensors or numpy arrays.

import numpy as np


class StreamNorm:
    def __init__(self, alpha, mel_means, mel_vars):
        """
        StreamNorm implements exponential moving average normalization.
        The values supplied to init() should be computed from training data.
        The stored values are then updated over time as new frames are normalized.
        """
        # alpha should be a scalar, a float, in [0.0,1.0), perhaps 0.001
        np.testing.assert_(alpha >= 0.0)
        np.testing.assert_(alpha < 1.0)
        self.alpha = alpha

        # mel_dim (referred to in comments) is typically 80
        # mel_means should be a vector of floats, shape (mel_dim,)
        self.mel_means = mel_means

        # mel_vars should be a vector of positive floats, shape (mel_dim,)
        z = np.zeros_like(mel_vars)
        np.testing.assert_array_less(
            z, mel_vars, "\nERROR : All variances should be positive\n"
        )
        self.mel_vars = mel_vars

    def normalize(
        self, new_vec
    ):  # vector of floats, shape (mel_dim,), PyTorch Tensor or numpy array
        """
        Formulae from:
        https://stats.stackexchange.com/questions/6874/exponential-weighted-moving-skewness-kurtosis # noqa: E501
        https://nestedsoftware.com/2018/03/27/calculating-standard-deviation-on-streaming-data-253l.23919.html # noqa: E501
        https://silo.tips/download/incremental-calculation-of-weighted-mean-and-variance (end of) # noqa: E501
        """
        # Update the internal means and vars with the new vector using the formulae just
        # mentioned.
        # Note that since mel_vars is always updated by the old_vars plus a squared
        # quantity, and since the old_vars begin positive (asserted above) and since
        # 0<=alpha<1 (asserted above), the new vars are guaranteed positive in all
        # circumstances, so the sqrt will never fail.
        diff = new_vec - self.mel_means
        self.mel_means = self.mel_means + self.alpha * diff
        self.mel_vars = (1 - self.alpha) * (self.mel_vars + self.alpha * diff * diff)

        # normalize new_vec using the internal stats
        norm_vec = (new_vec - self.mel_means) / np.sqrt(self.mel_vars)

        return norm_vec
