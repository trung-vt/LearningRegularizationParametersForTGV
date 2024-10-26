import torch
import pandas as pd
from typing import Callable, Literal, Any, Union, Tuple, Optional, List

from pdhg.pdhg import TvLambdaReg, TgvLambdaReg


class ScalarSearcher:
    def __init__(
            self, get_denoised: Callable[
                [Union[TvLambdaReg, TgvLambdaReg]], torch.Tensor],
            compute_metrics: Callable[[torch.Tensor], pd.DataFrame],
            best_metric: Literal["MSE", "PSNR", "SSIM"],
    ):
        self.get_denoised = get_denoised
        self.compute_metrics = compute_metrics
        self.best_metric = best_metric

    def compute_best_metric_tv(
            self,
            lambda_reg: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        denoised = self.get_denoised(lambda_reg)
        psnr, ssim = self.compute_metrics(denoised)
        return psnr if self.best_metric == "PSNR" else ssim
        # metrics_df = self.compute_metrics(denoised)
        # assert len(metrics_df) == 1, \
        #     f"Expected 1 row, got {len(metrics_df)} rows."
        # return metrics_df[self.best_metric].max()

    def compute_best_metric_tgv(
            self,
            lambda0_w: Union[float, torch.Tensor],
            lambda1_v: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        lambda_reg = {
            "lambda0_w": lambda0_w,
            "lambda1_v": lambda1_v
        }
        denoised = self.get_denoised(lambda_reg)
        psnr, ssim = self.compute_metrics(denoised)
        return psnr if self.best_metric == "PSNR" else ssim
        # metrics_df = self.compute_metrics(denoised)
        # assert len(metrics_df) == 1, \
        #     f"Expected 1 row, got {len(metrics_df)} rows."
        # return metrics_df[self.best_metric].max()

    def brute_force_and_denoise_tv(
            self, search_1d: Callable[[Any], float],
            range_lambda: Union[Tuple[float, float], List[float]],
            num_search_iters: int = None,
            func: Optional[Callable[[Any], Any]] = None,
    ) -> Tuple[torch.Tensor, TvLambdaReg, Any]:
        best_lambda = search_1d(
            range_lambda,
            self.compute_best_metric_tv, max, num_search_iters, func)
        best_denoised = self.get_denoised(best_lambda)
        best_metrics = self.compute_metrics(best_denoised)
        return best_denoised, best_lambda, best_metrics

    def brute_force_and_denoise_tgv(
            self, search_2d: Callable[[Any], Tuple[float, float]],
            range_lambda0_w: Union[Tuple[float, float], List[float]],
            range_lambda1_v: Union[Tuple[float, float], List[float]],
            num_search_iters: int = None,
            func: Optional[Callable[[Any], Any]] = None,
    ) -> Tuple[torch.Tensor, TgvLambdaReg, Any]:
        best_lambda0_w, best_lambda1_v = search_2d(
            range_lambda0_w, range_lambda1_v,
            self.compute_best_metric_tgv, max, num_search_iters, func)
        best_lambda_reg = {
            "lambda0_w": best_lambda0_w,
            "lambda1_v": best_lambda1_v,
        }
        best_denoised = self.get_denoised(best_lambda_reg)
        best_metrics = self.compute_metrics(best_denoised)
        return best_denoised, best_lambda_reg, best_metrics
