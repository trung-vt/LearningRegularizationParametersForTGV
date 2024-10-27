import torch
from typing import Callable, Union, Tuple, Any

from networks.mri_pdhg_net import MriPdhgNet
from utils.metrics import ImageMetricsEvaluator
from pdhg.pdhg import TvLambdaReg, TgvLambdaReg


class MriIteration:
    def __init__(
            self,
            model: Union[MriPdhgNet, Callable[
                [Any], Tuple[torch.Tensor, Union[TvLambdaReg, TgvLambdaReg]]]],
            metrics_evaluator: ImageMetricsEvaluator):
        self.model = model
        self.metrics_evaluator = metrics_evaluator

    def perform_iteration(
            self,
            data: Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        x_corrupted, x_true, kdata_corrupted, undersampling_kmask = data
        coil_sensitive_map = None

        x_reconstructed, lambda_reg = self.model(
            batch_kdata=kdata_corrupted,
            batch_kmask=undersampling_kmask,
            batch_x=x_corrupted,
            batch_csmap=coil_sensitive_map)

        loss = torch.nn.functional.mse_loss(
            torch.view_as_real(x_reconstructed),
            torch.view_as_real(x_true)
        )

        psnr, ssim = self.metrics_evaluator.compute_torch_complex(
            x=x_reconstructed, x_true=x_true)

        # x_reconstructed_abs = x_reconstructed.abs()
        # min_val = min(x_reconstructed_abs.min(), min_val)
        # max_val = max(x_reconstructed_abs.max(), max_val)

        return loss, psnr, ssim
