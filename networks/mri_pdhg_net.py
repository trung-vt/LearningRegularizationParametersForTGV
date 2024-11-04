import torch
import torch.nn as nn
# import torch.nn.functional as F
import math
from typing import Union, Optional, Literal, Tuple, Dict, Callable

from pdhg.mri_pdhg import MriPdhgTorch


def getpadding(
        size: int, multipleof: int = 1, minpad: int = 0) -> Tuple[int, int]:
    if (size + 2 * minpad) % multipleof == 0:
        pad = 2 * minpad
    else:
        pad = (2 * minpad) + multipleof - (size + (2 * minpad)) % multipleof
    return (math.ceil(pad / 2), math.floor(pad / 2))


class MriPdhgNet(nn.Module):
    def __init__(
            self, device: Union[str, torch.device],
            pdhg_algorithm: Literal["tv", "tgv"],
            T: int,
            cnn: nn.Module,
            params_config: Dict[str, Union[float, torch.Tensor]],
            # Bounds for the lambda values
            low_bound: Union[float, torch.Tensor] = 0.0,
            up_bound: Optional[Union[float, torch.Tensor]] = None,
            padding: Literal[
                "constant", "reflect", "replicate", "circular", "zeros"
            ] = "reflect"):

        super().__init__()
        self.device = device
        self.pdhg_solver = MriPdhgTorch(
            device=device, pdhg_algorithm=pdhg_algorithm)
        self.T = T
        self.cnn = cnn
        self.set_params(params_config)
        low_bound = None if low_bound is None else torch.as_tensor(low_bound)
        up_bound = None if up_bound is None or up_bound == 0.0 \
            else torch.as_tensor(up_bound)
        self.register_buffer(
            name="low_bound", tensor=low_bound, persistent=False)
        self.register_buffer(
            name="up_bound", tensor=up_bound, persistent=False)
        self.padding = padding

    def set_L(self) -> None:
        norm_of_op_A = 1
        # norm_of_grad_op_nabla = (dim * 4) ** (1 / 2)
        norm_of_grad_op_nabla = torch.sqrt(torch.tensor(8.0)).to(self.device)
        # Expect L = sqrt(9) = 3
        # self.L = (norm_of_op_A ** 2 + norm_of_grad_op_nabla ** 2) ** (1 / 2)
        self.L = torch.tensor(3.0, device=self.device)
        print(f"Norm of operator A: {norm_of_op_A}")
        print(f"Norm of gradient operator nabla: {norm_of_grad_op_nabla}")
        print(f"L: {self.L}")
        assert 0.0 <= self.L <= 3.0, f"Require 0 <= L <= 3. Got L={self.L}"

    def get_learnable_param(
            self, param: Union[float, torch.Tensor]) -> torch.Tensor:
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, device=self.device)
        return nn.Parameter(param, requires_grad=True).to(self.device)

    def set_sigma_and_tau(
            self, params_config: Dict[str, Union[float, torch.Tensor]]
    ) -> None:
        # # sigma, tau scaling constants:
        # # sigma = sigmoid(alpha)/L * exp(beta)
        # # tau = sigmoid(alpha)/L / exp(beta)
        # # ensuring sigma*tau*L^2 < 1 for convergence (?)
        # # TODO: Can it be equal???
        self.learns_sigma_and_tau = params_config["learns_sigma_and_tau"]
        self.learns_alpha = params_config["learns_alpha"]
        if self.learns_sigma_and_tau:
            if self.learns_alpha:
                self.alpha_raw = self.get_learnable_param(
                    params_config.get("initial_alpha", 10.0))
            self.beta = self.get_learnable_param(
                params_config.get("initial_beta", 0.0))
        else:
            self.constant_sigma = params_config.get(
                "constant_sigma", 1.0 / self.L)
            self.constant_tau = params_config.get(
                "constant_tau", 1.0 / self.L)
            assert self.constant_sigma >= 0.0, "Require sigma >= 0.0. " + \
                f"Got sigma={self.constant_sigma}"
            assert self.constant_tau >= 0.0, "Require tau >= 0.0. " + \
                f"Got tau={self.constant_tau}"
            assert self.constant_sigma * self.constant_tau <= self.L ** 2, \
                "Require sigma * tau <= L^2 for convergence. " + \
                f"Got sigma={self.constant_sigma}, " + \
                f"tau={self.constant_tau}, " + \
                f"L={self.L}"

    # def set_theta(
    #         self, params_config: Dict[str, Union[float, torch.Tensor]]
    # ) -> None:
    #     # #  theta=sigmoid(theta_raw) as theta should be in \in [0,1].
    #     # #  Starting theta close to 1.
    #     self.learns_theta = params_config["learns_theta"]
    #     if self.learns_theta:
    #         self.theta_raw = self.get_learnable_param(
    #             params_config.get("initial_theta_raw", 10.0))
    #     else:
    #         self.constant_theta = params_config.get("constant_theta", 1.0)
    #         assert 0.0 <= self.constant_theta <= 1.0, \
    #             f"Require theta in [0, 1]. Got theta={self.constant_theta}"

    def set_params(
            self, params_config: Dict[str, Union[float, torch.Tensor]]
    ) -> None:
        self.set_L()
        self.set_sigma_and_tau(params_config)
        # self.set_theta(params_config)

    @property
    def alpha(self) -> torch.Tensor:
        if self.learns_alpha:
            return torch.sigmoid(self.alpha_raw)
        else:
            return 1.0

    # Overriding the sigma, tau, theta properties to allow learning
    @property
    def sigma(self) -> Union[float, torch.Tensor]:
        if self.learns_sigma_and_tau:
            # return torch.sigmoid(self.alpha) / self.L * torch.exp(self.beta)
            # return 1.0 / self.L * torch.exp(self.beta)
            return self.alpha / self.L * torch.exp(self.beta)
        else:
            return self.constant_sigma
            # return 1.0 / self.L

    @property
    def tau(self) -> Union[float, torch.Tensor]:
        if self.learns_sigma_and_tau:
            # return torch.sigmoid(self.alpha) / self.L / torch.exp(self.beta)
            # return 1.0 / self.L / torch.exp(self.beta)
            return self.alpha / self.L / torch.exp(self.beta)
        else:
            return self.constant_tau
            # return 1.0 / self.L

    # @property
    # def theta(self) -> Union[float, torch.Tensor]:
    #     if self.learns_theta:
    #         return torch.sigmoid(self.theta_raw)
    #         # return 1.0
    #     else:
    #         return self.constant_theta
    #         # return 1.0

    def prepare_for_cnn(
            # self, x: torch.Tensor) -> Tuple[torch.Tensor, slice]:
            self, x: torch.Tensor) -> torch.Tensor:
        # convert to 2-channel view:
        #   (Nb,Nx,Ny,Nt) (complex) --> (Nb,2,Nx,Ny,Nt) (real)
        x_real = torch.view_as_real(x)
        # (Nb,2,Nx,Ny,Nt) -> (Nb,Nt,Nx,Ny,2)
        x_real = torch.moveaxis(x_real, -1, 1)

        # padmultiple = 4
        # minpad = 4
        # padsizes = [
        #     getpadding(n, padmultiple, minpad) for n in x_real.shape[2:]]
        # # pad takes the padding size
        # # starting from last dimension moving forward
        # pad = [s for p in padsizes[::-1] for s in p]
        # crop = [Ellipsis] + [slice(p[0], -p[1]) for p in padsizes]
        # x_padded_real = nn.functional.pad(x_real, pad, mode=self.padding)
        # return x_padded_real, crop
        return x_real

    def scale_lambda(self, lambdas: torch.Tensor) -> torch.Tensor:
        low_bound = 0.0 if self.low_bound is None else self.low_bound
        if self.up_bound is not None:
            return low_bound + \
                (self.up_bound - low_bound) * torch.sigmoid(lambdas)
        else:
            return low_bound + torch.nn.functional.softplus(lambdas, beta=5)

    def get_lambda_cnn(
            self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # x_padded_real, crop = self.prepare_for_cnn(x)
        # lambda_cnn = self.cnn(x_padded_real)
        # lambda_cnn = lambda_cnn[crop]
        # lambda_scaled = self.scale_lambda(lambda_cnn)
        # lambda0_w, lambda1_v = torch.chunk(lambda_scaled, 2, 1)

        x_real = self.prepare_for_cnn(x)
        # print("No padding!")
        lambda_cnn = self.cnn(x_real)
        lambda_scaled = self.scale_lambda(lambda_cnn)
        lambda0_w, lambda1_v = torch.chunk(lambda_scaled, 2, 1)

        # NOTE: Assume channel size is 1 (grayscale image).
        # Remove the channel dimension,
        #   i.e. (batch_size, 1, Nx, Ny) -> (batch_size, Nx, Ny)
        lambda0_w = lambda0_w.squeeze(1)
        lambda1_v = lambda1_v.squeeze(1)
        return {
            "lambda0_w": lambda0_w,
            "lambda1_v": lambda1_v
        }

    def forward(
            self,
            batch_kdata: torch.Tensor,  # measured data
            batch_kmask: torch.Tensor,  # undersampling mask
            batch_x: torch.Tensor,  # input image
            batch_csmap: torch.Tensor = None,  # coil sensitivity map
            return_dict: bool = False,
            tqdm_progress_bar: Optional[Callable] = None
    ) -> Tuple[
        Union[torch.Tensor, Dict[str, torch.Tensor]],
        Union[torch.Tensor, Dict[str, torch.Tensor]]
    ]:
        # estimate lambda map(s) from the image
        lambda_reg = self.get_lambda_cnn(batch_x.clone())
        if self.pdhg_solver.pdhg_algorithm == "tv":
            lambda_reg = lambda_reg["lambda1_v"]
        batch_x_reconstructed = self.pdhg_solver(
            num_iters=self.T, lambda_reg=lambda_reg,
            kdata=batch_kdata, kmask=batch_kmask,
            csmap=batch_csmap, state=batch_x.clone(),
            sigma=self.sigma, tau=self.tau,
            # theta=self.theta,
            theta=1.0,  # TODO: Allow learning theta?
            return_dict=return_dict, tqdm_progress_bar=tqdm_progress_bar)
        return batch_x_reconstructed, lambda_reg
