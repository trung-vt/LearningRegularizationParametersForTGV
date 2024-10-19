import torch
from torchmetrics.image import \
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import List, Union, Optional, Tuple, Literal


class ImageMetricsEvaluator:

    def __init__(
            self,
            device: Union[str, torch.device],
            metrics: List[str] = ["MSE", "PSNR", "SSIM"],
            data_range: Optional[Union[int, float]] = None,
            complex_to_real_conversion: Literal["abs", "view_as_real"] = "abs",
            clips_before_comparing: bool = False):

        self.data_range = data_range
        self.metrics = metrics
        self.clips_before_comparing = clips_before_comparing
        self.complex_to_real_conversion = complex_to_real_conversion
        print(
            "NOTE: Complex-to-real conversion method:",
            complex_to_real_conversion)

        self.psnr_torch = PeakSignalNoiseRatio(
            data_range=self.data_range).to(device)
        self.ssim_torch = StructuralSimilarityIndexMeasure(
            data_range=self.data_range).to(device)

    def convert_complex_to_real_batch(self, x: torch.Tensor) -> torch.Tensor:
        if self.complex_to_real_conversion == "abs":
            x_real = x.abs()
            # We want a batch. For example, (1, 1, 320, 320).
            while x_real.ndim < 4:
                x_real = x_real.unsqueeze(0)
        elif self.complex_to_real_conversion == "view_as_real":
            # view_as_real adds an extra dimension for the real and imaginary
            #   parts of the complex number.
            #   For example, (320, 320) -> (320, 320, 2).
            #   moveaxis(-1, 0) moves the last dimension to the first.
            #   For example, (320, 320, 2) -> (2, 320, 320).
            x_real = torch.view_as_real(x).moveaxis(-1, 0)
            # We want a pair of batches. For example, (2, 1, 1, 320, 320).
            while x_real.ndim < 5:
                x_real = x_real.unsqueeze(1)
        return x_real

    def compute_torch_unknown_range_single_channel(
            self, x: torch.Tensor, x_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data_range = torch.max(x_true) - torch.min(x_true)
        self.psnr_torch.data_range = data_range
        self.ssim_torch.data_range = data_range
        psnr = self.psnr_torch(x, x_true).to(x.device)
        ssim = self.ssim_torch(x, x_true).to(x.device)
        return psnr, ssim

    def compute_torch_complex(
            self, x: torch.Tensor, x_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_x_real = self.convert_complex_to_real_batch(x)
        batch_x_true_real = self.convert_complex_to_real_batch(x_true)

        if self.complex_to_real_conversion == "abs":
            psnr, ssim = self.compute_torch_unknown_range_single_channel(
                batch_x_real, batch_x_true_real)
        elif self.complex_to_real_conversion == "view_as_real":
            # Take the average of the 2 images produced by the original
            #   real and imaginary parts.
            psnr_0, ssim_0 = self.compute_torch_unknown_range_single_channel(
                batch_x_real[0], batch_x_true_real[0])
            psnr_1, ssim_1 = self.compute_torch_unknown_range_single_channel(
                batch_x_real[1], batch_x_true_real[1])
            psnr = (psnr_0 + psnr_1) / 2
            ssim = (ssim_0 + ssim_1) / 2
        return psnr, ssim
