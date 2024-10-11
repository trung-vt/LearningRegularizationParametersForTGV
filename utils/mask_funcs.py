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
        torch.fft.ifftshift(torch.tensor(mask)),
        k=1,
        dims=(-2, -1),
    ).to(torch.complex64)

    return mask
