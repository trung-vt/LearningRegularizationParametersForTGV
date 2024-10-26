import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Any, Literal, Callable, Tuple

from encoding_objects.cart_2d_enc_obj import Cart2DEncObj
from pdhg.pdhg import TvPdhgTorch, TgvPdhgTorch, TvLambdaReg, TgvLambdaReg


class MriPdhgTorch(nn.Module):
    def __init__(
            self,
            device: Union[str, torch.device],
            pdhg_algorithm: Literal["tv", "tgv"]):
        super().__init__()
        self.encoding_object = Cart2DEncObj()
        if pdhg_algorithm == "tv":
            pdhg_torch = TvPdhgTorch
        elif pdhg_algorithm == "tgv":
            pdhg_torch = TgvPdhgTorch
        else:
            raise ValueError(
                "pdhg_algorithm should be 'tv' or 'tgv', " +
                f"but got '{pdhg_algorithm}'")
        self.pdhg_algorithm = pdhg_algorithm
        self.pdhg_solver = pdhg_torch(device=device)

    def forward(
            self, num_iters: int,  # number of iterations
            lambda_reg: Union[
                float, torch.Tensor, Dict[
                    Literal["lambda1_v", "lambda0_w"],
                    Union[float, torch.Tensor]]],
            kdata: torch.Tensor,  # measured data
            kmask: torch.Tensor,
            state: Union[torch.Tensor, Dict[str, Any]],
            csmap: Optional[torch.Tensor] = None,   # coil sensitivity map
            sigma: Union[float, torch.Tensor] = 1.0 / 3,
            tau: Union[float, torch.Tensor] = 1.0 / 3,
            theta: Union[float, torch.Tensor] = 1.0,
            return_dict: bool = False,
            device: Optional[Union[torch.device, str]] = None,
            tqdm_progress_bar: Optional[Callable] = None
    ) -> Union[torch.Tensor, Dict[
            Literal["x", "v", "p", "q", "r", "x_bar", "v_bar"], torch.Tensor]]:
        def A(x: torch.Tensor) -> torch.Tensor:
            return self.encoding_object.apply_A(x=x, csm=csmap, mask=kmask)

        def AH(k: torch.Tensor) -> torch.Tensor:
            return self.encoding_object.apply_AH(k=k, csm=csmap, mask=kmask)
        result = self.pdhg_solver(
            num_iters=num_iters, A=A, AH=AH, lambda_reg=lambda_reg,
            z=kdata, state=state, sigma=sigma, tau=tau, theta=theta,
            return_dict=return_dict, tqdm_progress_bar=tqdm_progress_bar,
            device=device)
        return result


class MriPdhgTorchTest(MriPdhgTorch):
    def __init__(
            self,
            device: Union[str, torch.device],
            pdhg_algorithm: Literal["tv", "tgv"],
            num_iters: int,  # number of iterations
            kdata: torch.Tensor,  # measured data
            kmask: torch.Tensor,
            state: Union[torch.Tensor, Dict[str, Any]],
            csmap: Optional[torch.Tensor] = None,   # coil sensitivity map
            sigma: Union[float, torch.Tensor] = 1.0 / 3,
            tau: Union[float, torch.Tensor] = 1.0 / 3,
            theta: Union[float, torch.Tensor] = 1.0,
            tqdm_progress_bar: Optional[Callable] = None
    ):
        super().__init__(device=device, pdhg_algorithm=pdhg_algorithm)
        self.num_iters = num_iters
        self.kdata = kdata
        self.kmask = kmask
        self.state = state
        self.csmap = csmap
        self.sigma = sigma
        self.tau = tau
        self.theta = theta
        self.tqdm_progress_bar = tqdm_progress_bar

    def get_denoised(
            self, lambda_reg: Union[TvLambdaReg, TgvLambdaReg]
    ) -> torch.Tensor:
        return super().forward(
            num_iters=self.num_iters, lambda_reg=lambda_reg, kdata=self.kdata,
            kmask=self.kmask, state=self.state, csmap=self.csmap,
            sigma=self.sigma, tau=self.tau, theta=self.theta,
            return_dict=False, tqdm_progress_bar=self.tqdm_progress_bar)
