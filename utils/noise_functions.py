# Acknowledgement: The code was provided by Felix Zimmermann for
# https://github.com/koflera/LearningRegularizationParameterMaps/tree/main.

import torch
import torch.nn as nn
import numpy as np

from math import sqrt


class SamplingOperator2D(nn.Module):
    """Module that selects the non-zero k-space coefficients
    from a zero-filled k-space data sampled on a Cartesian grid."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, kspace_data, mask):

        # check that for all temporal points the number of samples lines
        # is the same
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask)
            nb, nc, Nx, _ = kspace_data.shape
            n_sampled_lines = int(torch.tensor(torch.sum(mask[0, 0, :]).abs().item()))

            # restrict k-space data to acquired k-space coefficients
            kspace_data = torch.masked_select(kspace_data, mask.to(torch.bool)).view(
                nb,
                nc,
                Nx,
                n_sampled_lines,
            )

        return kspace_data


def add_gaussian_noise(kdata, mask, noise_var=0.05, seed=0):
    """
    add gaussian noise with chosen variance to k-space data.

    N.B. z = x + i*y in C^N ~ N(0,sigma**2 * Id )  is equivalent to
                x ~ N(0, sigma**2 / 2 * Id) and y ~ N(0, sigma**2 / 2 * Id)
    """

    # torch.manual_seed(seed)
    # np.random.seed(seed)

    sampling_op = SamplingOperator2D()

    kdata_noisy = kdata.clone()

    supp = torch.where(kdata != 0)

    kdata = sampling_op(kdata, mask)

    # compute mean and std
    mu_r, std_r = torch.mean(kdata.real, dim=(2, 3), keepdim=True), torch.std(
        kdata.real, dim=(2, 3), keepdim=True
    )
    mu_i, std_i = torch.mean(kdata.imag, dim=(2, 3), keepdim=True), torch.std(
        kdata.imag, dim=(2, 3), keepdim=True
    )

    # center k-space data
    kdata_r = (kdata.real - mu_r) / std_r
    kdata_i = (kdata.imag - mu_i) / std_i

    noise_r = torch.randn_like(kdata_r)
    noise_i = torch.randn_like(kdata_i)
    noise = noise_r + 1j * noise_i

    kdata_r = kdata_r + sqrt(noise_var / 2) * noise_r
    kdata_i = kdata_i + sqrt(noise_var / 2) * noise_i

    kdata = (mu_r + std_r * kdata_r) + 1j * (mu_i + std_i * kdata_i)
    kdata_noisy[supp] = kdata.flatten()

    return kdata_noisy, noise
