import torch
from typing import Callable


class MriBaseDataset(torch.utils.data.Dataset):
    def __init__(
            self, all_x_true_complex: torch.Tensor, scale_factor: float):
        super().__init__()
        print(f"\n\nGround truth data shape: {all_x_true_complex.shape}")
        self.all_rescaled_x_true_complex = all_x_true_complex * scale_factor
        min_abs_val = self.all_rescaled_x_true_complex.abs().min()
        max_abs_val = self.all_rescaled_x_true_complex.abs().max()
        print(f"min abs val of ground truth: {min_abs_val}")
        print(f"max abs val of ground truth: {max_abs_val}")

    def __getitem__(self, idx):
        single_rescaled_x_true_complex = self.all_rescaled_x_true_complex[idx]
        return single_rescaled_x_true_complex

    def __len__(self):
        return len(self.all_rescaled_x_true_complex)


class MriPreProcessedDataset(MriBaseDataset):
    def __init__(
        self,
        all_rescaled_x_true_complex: torch.Tensor,
        all_x_corrupted: torch.Tensor,
        all_kdata_corrupted: torch.Tensor,
        all_undersampling_kmasks: torch.Tensor,
        acc_factor_R: float, gaussian_noise_sigma: float
    ):
        super().__init__(
            all_x_true_complex=all_rescaled_x_true_complex, scale_factor=1)
        self.all_rescaled_x_corrupted = all_x_corrupted
        self.all_kdata_corrupted = all_kdata_corrupted
        self.all_undersampling_kmasks = all_undersampling_kmasks
        self.acc_factor_R = acc_factor_R
        self.gaussian_noise_sigma = gaussian_noise_sigma
        print(f"\nCorrupted data shape: {all_x_corrupted.shape}")
        print(f"min abs val of corrupted: {all_x_corrupted.abs().min()}")
        print(f"max abs val of corrupted: {all_x_corrupted.abs().max()}")
        print(f"\nCorrupted kdata shape: {all_kdata_corrupted.shape}")
        print(f"\nkmasks shape: {all_undersampling_kmasks.shape}")
        print()

    def __getitem__(self, idx):
        single_rescaled_x_true_complex = self.all_rescaled_x_true_complex[idx]
        single_rescaled_x_corrupted = self.all_rescaled_x_corrupted[idx]
        kdata_corrupted = self.all_kdata_corrupted[idx]
        undersampling_kmask = self.all_undersampling_kmasks[idx]
        return (
            single_rescaled_x_corrupted, single_rescaled_x_true_complex,
            kdata_corrupted, undersampling_kmask)


class MriDynamicallyGeneratedDataset(MriBaseDataset):
    def __init__(
            self, all_x_true_complex: torch.Tensor, scale_factor: float,
            get_corrupted_data: Callable):
        super().__init__(
            all_x_true_complex=all_x_true_complex, scale_factor=scale_factor)
        self.get_corrupted_data = get_corrupted_data

    def __getitem__(self, idx):
        single_rescaled_x_true_complex = self.all_rescaled_x_true_complex[idx]
        single_rescaled_x_corrupted, kdata_corrupted, undersampling_kmask = \
            self.get_corrupted_data(
                x_true=single_rescaled_x_true_complex)
        return (
            single_rescaled_x_corrupted,
            single_rescaled_x_true_complex,
            kdata_corrupted, undersampling_kmask)
