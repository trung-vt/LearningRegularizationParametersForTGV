import torch
from typing import Callable, Optional, Union, Tuple, Any

from networks.mri_pdhg_net import MriPdhgNet
from scripts.mri.logger import Logger
from utils.metrics import ImageMetricsEvaluator
from pdhg.pdhg import TvLambdaReg, TgvLambdaReg


def perform_iteration(
        data: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        model: Union[MriPdhgNet, Callable[
            [Any], Tuple[torch.Tensor, Union[TvLambdaReg, TgvLambdaReg]]]],
        metrics_evaluator: ImageMetricsEvaluator,
) -> None:
    x_corrupted, x_true, \
        kdata_corrupted, undersampling_kmask = data
    coil_sensitive_map = None

    x_reconstructed, lambda_reg = model(
        batch_kdata=kdata_corrupted,
        batch_kmask=undersampling_kmask,
        batch_x=x_corrupted,
        batch_csmap=coil_sensitive_map)

    psnr, ssim = metrics_evaluator.compute_torch_complex(
        x=x_reconstructed, x_true=x_true)

    return x_reconstructed, x_true, psnr, ssim


def perform_epoch(
        data_iterator,
        model: Union[MriPdhgNet, Callable[
            [Any], Tuple[torch.Tensor, Union[TvLambdaReg, TgvLambdaReg]]]],
        logger: Logger,
        is_training: bool,
        metrics_evaluator: ImageMetricsEvaluator,
        learning_rate_scheduler: Optional[
            torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        sets_tqdm_postfix: bool = False
) -> torch.Tensor:
    running_metrics = torch.zeros(3)    # loss, psnr, ssim
    total_metrics = torch.zeros(3)      # loss, psnr, ssim
    min_val = torch.inf
    max_val = -torch.inf

    for idx, data in enumerate(data_iterator):
        x_reconstructed, x_true, psnr, ssim = perform_iteration(
            data, model, metrics_evaluator)

        loss = torch.nn.functional.mse_loss(
            torch.view_as_real(x_reconstructed),
            torch.view_as_real(x_true)
        )

        new_metrics = torch.tensor([loss.detach(), psnr, ssim])
        total_metrics += new_metrics
        logger.update_and_log_metrics(
            stage="intermediate", iter_idx=idx,
            new_metrics=new_metrics, running_metrics=running_metrics)

        x_reconstructed_abs = x_reconstructed.abs()
        min_val = min(x_reconstructed_abs.min(), min_val)
        max_val = max(x_reconstructed_abs.max(), max_val)

        if is_training:
            # https://lightning.ai/docs/pytorch/stable/common/optimization.html
            # (Specific for PyTorch Lightning, but same principles apply?)
            # It is good practice to call optimizer.zero_grad() before
            #   self.manual_backward(loss).
            # You can call lr_scheduler.step() at arbitrary intervals.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            learning_rate_scheduler.step()
            optimizer.step()

        if sets_tqdm_postfix:
            # Assume the metrics are ordered correctly: loss, psnr, ssim
            data_iterator.set_postfix({
                "loss": f"{new_metrics[0].item():.4f}",
                # "spatial": f"{spatial_min:.2f}/{spatial_max:.2f}"
                "PSNR": f"{new_metrics[1].item():.2f}",
                "SSIM": f"{new_metrics[2].item():.4f}"
            })

    avg_metrics = total_metrics / len(data_iterator)
    print(f"min_val = {min_val}")
    print(f"max_val = {max_val}")
    return avg_metrics
