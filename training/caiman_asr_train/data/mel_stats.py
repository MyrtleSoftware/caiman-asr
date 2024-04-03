from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class MelStats:
    """
    Log mel feature statistics.
    """

    means: Tensor
    vars: Tensor

    def __post_init__(self):
        """
        Check that the means and vars are valid.
        """
        mel_means, mel_vars = self.means, self.vars
        # mel_means should be a vector of floats, shape (mel_dim,)
        assert mel_means.shape == mel_vars.shape
        assert len(mel_means.shape) == 1, "Means/vars should be vectors, not matrices"

        # mel_vars should be a vector of positive floats, shape (mel_dim,)
        z = np.zeros_like(mel_vars)
        np.testing.assert_array_less(
            z, mel_vars, "\nERROR : All variances should be positive\n"
        )

    @property
    def stddevs(self) -> Tensor:
        """
        Return the standard deviations of the mel features.
        """
        return self.vars**0.5

    @classmethod
    def from_dir(cls, dir: str | None) -> "MelStats":
        """
        Load mel stats from a directory.
        """
        if dir is None:
            raise ValueError("dir must be a string, not None")
        means = torch.load(f"{dir}/melmeans.pt")
        vars = torch.load(f"{dir}/melvars.pt")
        return cls(means, vars)

    def __repr__(self) -> str:
        means_str = f"<tensor: {self.means.shape}>"
        vars_str = f"<tensor: {self.vars.shape}>"
        return f"{self.__class__.__name__}(means={means_str}, vars={vars_str})"
