# Acknowledgement: The code was provided by Felix Zimmermann for
# https://github.com/koflera/LearningRegularizationParameterMaps/tree/main.

import torch


def normal_pdf(length, sensitivity):
    return torch.exp(-sensitivity * (torch.arange(length) - length / 2) ** 2)


def cartesian_mask(shape, acc: float, sample_n: int = 10):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    Note:
            function borrowed from Jo Schlemper from
            https://github.com/js3611/Deep-MRI-Reconstruction/blob/master/utils/compressed_sensing.py

    TODO: Improve efficiency by using torch instead of numpy to avoid
    switching to cpu.
    """
    N = torch.prod(torch.tensor(shape[:-2])).item()
    Nx, Ny = shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.0) ** 2)
    lmda = Nx / (2.0 * acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1.0 / Nx

    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2: Nx // 2 + sample_n // 2] = 0
        pdf_x /= torch.sum(pdf_x)
        n_lines -= sample_n

    mask = torch.zeros((N, Nx))
    for i in range(N):
        idx = torch.multinomial(pdf_x, n_lines, replacement=False)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2: Nx // 2 + sample_n // 2] = 1

    mask = torch.as_strided(mask, (N, Nx, Ny), (Nx, 1, 0))

    mask = mask.reshape(shape)
    mask = torch.rot90(
        torch.fft.ifftshift(mask),
        k=1,
        dims=(-2, -1),
    ).to(torch.complex64)

    return mask


def low_field_mask(shape, super_resolution_factor: float):
    """
    Prepare a low-field mask for training by simulating k-space data.

    Parameters
    ----------
    shape : tuple
        Shape of the mask to be generated. Should be of the form (..., nx, ny).
    super_resolution_factor : float
        A number greater than 1 that determines the factor by which
        each side of the k-space is reduced.

    Returns
    -------
    torch.Tensor
        The generated low-field mask.
    """
    # Create the mask using the cartesian_mask function
    mask = torch.zeros(shape)
    Nx, Ny = shape[-2], shape[-1]
    reduced_Nx = int(Nx / super_resolution_factor)
    reduced_Ny = int(Ny / super_resolution_factor)
    start_x = (Nx - reduced_Nx) // 2
    start_y = (Ny - reduced_Ny) // 2
    mask[..., start_x:start_x + reduced_Nx, start_y:start_y + reduced_Ny] = 1.0
    mask = torch.fft.ifftshift(mask).to(torch.complex64)
    return mask


# def low_field_mask(
#     target_image: torch.Tensor, super_resolution_factor: float, noise_variance: float, seed: int = 0
# ) -> tuple[torch.Tensor, mrpro.operators.LinearOperator, torch.Tensor, torch.Tensor]:
#     """Prepare data for training by simulating k-space data.

#     Parameters
#     ----------
#     target_image : torch.Tensor
#         Target image to be used for simulation.
#     super_resolution_factor : float
#         Factor determining by how much the number of pixels is increased (i.e. how much smaller
#         each pixel will be).
#     noise_variance : float
#         Variance of the Gaussian noise to be added.
#     seed : float
#         seed of the random number generator.

#     Returns
#     -------
#     tuple
#         A tuple containing k-space data, forward operator adjoint reconstruction.
#     """
#     # randomly choose trajectories to define the fourier operator
#     ny, nx = target_image.shape[-2:]
#     recon_matrix = mrpro.data.SpatialDimension(z=1, y=ny, x=nx)
#     encoding_matrix = mrpro.data.SpatialDimension(z=1, y=ny, x=nx)

#     n_k1, n_k0 = ny // super_resolution_factor, nx // super_resolution_factor

#     traj = mrpro.data.traj_calculators.KTrajectoryCartesian()(
#         n_k0=int(n_k0),
#         k0_center=int(n_k0 // 2),
#         k1_idx=torch.arange(-n_k1 // 2, n_k1 // 2)[..., None, None, :, None],
#         k1_center=0,
#         k2_idx=torch.tensor(0),
#         k2_center=0,
#     )

#     fourier_operator = mrpro.operators.FourierOp(
#         traj=traj,
#         recon_matrix=recon_matrix,
#         encoding_matrix=encoding_matrix,
#     )

#     (kdata,) = fourier_operator(target_image)

#     rng = torch.Generator().manual_seed(seed)
#     kdata = kdata + noise_variance * kdata.std() * torch.randn(
#         kdata.shape, dtype=kdata.dtype, device=kdata.device, generator=rng
#     )

#     kdata, target_image = normalize_kspace_data_and_image(kdata, target_image)  # type: ignore[assignment]

#     (adjoint_recon,) = fourier_operator.H(kdata)

#     return kdata, fourier_operator, adjoint_recon, target_image
