import torch
from typing import Callable, Optional, Tuple, Any

from scripts.logger import Logger


def perform_epoch(
        data_loader: torch.utils.data.DataLoader,
        perform_iteration: Callable[[Any], Tuple[torch.Tensor, float, float]],
        is_training: bool,
        logger: Logger,
        learning_rate_scheduler: Optional[
            torch.optim.lr_scheduler._LRScheduler] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tqdm: Optional[Callable] = None,
        sets_tqdm_postfix: bool = False,
) -> torch.Tensor:
    """
    Perform an epoch of training or validation.

    Parameters
    ----------
    data_iterator : torch.utils.data.DataLoader | Any
        The data iterator to iterate over the dataset.
        Can be a `DataLoader` or any other iterable object.
        Usually also a `tqdm` object for progress bar.
    """
    running_metrics = torch.zeros(3)    # loss, psnr, ssim
    total_metrics = torch.zeros(3)      # loss, psnr, ssim
    # min_val = torch.inf
    # max_val = -torch.inf
    if tqdm is not None:
        data_iterator = tqdm(data_loader)
    else:
        data_iterator = data_loader
    for idx, data in enumerate(data_iterator):
        # from data.mri.data_generator import DataGenerator
        # print(f"Random int: {DataGenerator.get_random_int(4, 8)}")
        # print(f"Random float: {DataGenerator.get_random_float(0.00, 0.2)}")
        loss, psnr, ssim = perform_iteration(data)

        new_metrics = torch.tensor([loss.detach(), psnr, ssim])
        total_metrics += new_metrics
        logger.update_and_log_metrics(
            stage="intermediate", iter_idx=idx,
            new_metrics=new_metrics, running_metrics=running_metrics)

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
            assert tqdm is not None
            # Assume the metrics are ordered correctly: loss, psnr, ssim
            postfix_dict = {
                "loss": f"{new_metrics[0].item():.4f}",
                # "spatial": f"{spatial_min:.2f}/{spatial_max:.2f}"
                "PSNR": f"{new_metrics[1].item():.2f}",
                "SSIM": f"{new_metrics[2].item():.4f}",
                "R": data_loader.dataset.current_acceleration_factor_R,
                "sigma": data_loader.dataset.current_gaussian_noise_std_dev,
            }
            data_iterator.set_postfix(postfix_dict)

    avg_metrics = total_metrics / len(data_iterator)
    # print(f"min_val = {min_val}")
    # print(f"max_val = {max_val}")
    return avg_metrics
