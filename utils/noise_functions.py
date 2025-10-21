# Acknowledgement: The code was provided by Felix Zimmermann for
# https://github.com/koflera/LearningRegularizationParameterMaps/tree/main.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from math import sqrt


class CartesianSamplingOperator2D(nn.Module):
    """Module that selects the non-zero k-space coefficients
    from a zero-filled k-space data sampled on a Cartesian grid."""

    def forward(self, kspace_data: torch.Tensor, mask: torch.Tensor):

        # check that for all temporal points the number of samples lines
        # is the same
        if mask is not None:
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


class LowFieldSamplingOperator2D(nn.Module):
    """Module that selects the non-zero k-space coefficients
    from a zero-filled k-space data sampled on a low-field grid."""

    def forward(self, kspace_data: torch.Tensor, mask: torch.Tensor):
        if mask is not None:
            nb, nc, _, _ = kspace_data.shape
            n_k0 = int(torch.tensor(torch.sum(mask[0, :, 0]).abs().item()))
            n_k1 = int(torch.tensor(torch.sum(mask[0, 0, :]).abs().item()))

            # restrict k-space data to acquired k-space coefficients
            kspace_data = torch.masked_select(kspace_data, mask.to(torch.bool)).view(
                nb,
                nc,
                n_k0,
                n_k1
            )

        return kspace_data


def add_gaussian_noise(kdata: torch.Tensor, mask, sampling_op, noise_var=0.05, rng=None):
    """
    add gaussian noise with chosen variance to k-space data.

    N.B. z = x + i*y in C^N ~ N(0,sigma**2 * Id )  is equivalent to
                x ~ N(0, sigma**2 / 2 * Id) and y ~ N(0, sigma**2 / 2 * Id)
    """

    # torch.manual_seed(seed)
    # np.random.seed(seed)

    kdata_noisy = kdata.clone()

    supp = torch.where(kdata != 0)

    # kdata = kdata + noise_var * torch.std(kdata) * torch.randn(
    #     kdata.shape, dtype=kdata.dtype, device=kdata.device, generator=rng
    # )

    kdata: torch.Tensor = sampling_op(kdata, mask)  # NOTE: shape will change here

    # compute mean and std
    mu_r = torch.mean(kdata.real, dim=(2, 3), keepdim=True)
    std_r = torch.std(kdata.real, dim=(2, 3), keepdim=True)
    mu_i = torch.mean(kdata.imag, dim=(2, 3), keepdim=True)
    std_i = torch.std(kdata.imag, dim=(2, 3), keepdim=True)

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
