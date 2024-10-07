import torch
import torch.nn as nn
from typing import Literal, Union


class Cart2DEncObj(nn.Module):
    """

    Implementation of operators needed for MR reconstruction for static 2D
    Cartesian MR imaging.
    The forward operator is given by
    A = (I_Nc \otimes E) C,
    see e.g.
    https://web.eecs.umich.edu/~fessler/papers/files/talk/19/isbi-tutorial.pdf
    slide 9.

    This class contains the application of all different operators
    C, C^H, E, E^H, .. A, A^H etc required for constructing a reconstruction
    network.
    For single-coil MRI, csm can simply be set to None.

    Input shapes:
        x 	 	- 	(mb,  Nx, Ny)
        csm  	- 	(mb, Nc, Nx, Ny)
        mask  	- 	(mb,  Nx, Ny)
        k 	 	- 	(mb,  Nc, Nx, Ny)

    """

    def __init__(self, norm="ortho"):

        self.norm = norm

        super(Cart2DEncObj, self).__init__()

    def check_input(
            self, input: Union[torch.Tensor, None], input_name: str,
            func_name: str, expected_ndim: Literal[3, 4]
    ) -> None:
        """
        Raises an error if the shape is not as expected.
        """
        if input is not None and input.ndim != expected_ndim:
            raise ValueError(
                f"Expected input {input_name} for function {func_name} " +
                f"to have {expected_ndim} dimensions " +
                f"(mb, {'Nc, ' if expected_ndim == 4 else ''}Nx, Ny), " +
                f"but got {input.ndim} dimensions instead: {input.shape}."
            )

    def apply_C(self, x, csm):
        self.check_input(x, "x", func_name="apply_C", expected_ndim=3)
        return csm * x.unsqueeze(1) if csm is not None else x.unsqueeze(1)

    def apply_CH(self, xc, csm):
        # TODO: Why does xc here have 4 dimensions?
        self.check_input(xc, "xc", func_name="apply_CH", expected_ndim=4)
        self.check_input(csm, "csm", func_name="apply_CH", expected_ndim=4)
        return (
            torch.sum(csm.conj() * xc, dim=1, keepdim=False)
            if csm is not None
            else xc.squeeze(1)
        )

    def apply_E(self, x):
        # TODO: Why does x here have 4 dimensions?
        self.check_input(x, "x", func_name="apply_E", expected_ndim=4)
        return torch.fft.fftn(x, dim=(-2, -1), norm=self.norm)

    def apply_EH(self, k):
        self.check_input(k, "k", func_name="apply_EH", expected_ndim=4)
        return torch.fft.ifftn(k, dim=(-2, -1), norm=self.norm)

    def apply_EC(self, x, csm):
        self.check_input(x, "x", func_name="apply_EC", expected_ndim=3)
        self.check_input(csm, "csm", func_name="apply_EC", expected_ndim=4)
        return self.apply_E(self.apply_C(x, csm))

    def apply_mask(self, k, mask):
        self.check_input(k, "k", func_name="apply_mask", expected_ndim=4)
        self.check_input(mask, "mask", func_name="apply_mask", expected_ndim=3)
        return k * mask.unsqueeze(1) if mask is not None else k

    def apply_A(self, x, csm, mask):
        self.check_input(x, "x", func_name="apply_A", expected_ndim=3)
        self.check_input(csm, "csm", func_name="apply_A", expected_ndim=4)
        self.check_input(mask, "mask", func_name="apply_A", expected_ndim=3)
        return self.apply_mask(self.apply_E(self.apply_C(x, csm)), mask)

    def apply_AH(self, k, csm, mask):
        self.check_input(k, "k", func_name="apply_AH", expected_ndim=4)
        self.check_input(csm, "csm", func_name="apply_AH", expected_ndim=4)
        self.check_input(mask, "mask", func_name="apply_AH", expected_ndim=3)
        return self.apply_CH(self.apply_EH(self.apply_mask(k, mask)), csm)

    def apply_AHA(self, x, csm, mask):
        self.check_input(x, "x", func_name="apply_AHA", expected_ndim=3)
        self.check_input(csm, "csm", func_name="apply_AHA", expected_ndim=4)
        self.check_input(mask, "mask", func_name="apply_AHA", expected_ndim=3)
        return self.apply_AH(self.apply_A(x, csm, mask), csm, mask)
