import torch
from typing import Optional, Dict, Any, Union

from utils.mask_funcs import cartesian_mask
from utils.noise_functions import add_gaussian_noise
from encoding_objects.cart_2d_enc_obj import Cart2DEncObj


class DataGenerator:
    def __init__(
            self, data_config: Union[Dict[str, Any], None],
            device: Union[str, torch.device]):
        if data_config is not None:
            self.min_acceleration_factor_R = data_config[
                "min_acceleration_factor_R"]
            self.max_acceleration_factor_R = data_config[
                "max_acceleration_factor_R"]
            self.min_gaussian_noise_variance = data_config[
                "min_standard_deviation_sigma"]
            self.max_gaussian_noise_variance = data_config[
                "max_standard_deviation_sigma"]
        self.EncObj = Cart2DEncObj()
        self.device = device

    @staticmethod
    def get_random_int(min_val: int, max_val: int) -> int:
        return torch.randint(min_val, max_val, (1,)).item()

    @staticmethod
    def get_random_float(min_val: float, max_val: float) -> float:
        return torch.rand(1).item() * (max_val - min_val) + min_val

    def get_undersampling_kmask(self, shape: torch.Size) -> torch.Tensor:
        mask = cartesian_mask(
            shape=shape,
            acc=DataGenerator.get_random_int(
                self.min_acceleration_factor_R,
                self.max_acceleration_factor_R))
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask).to(torch.complex64)
        return mask

    def add_random_k_noise(
            self, kdata: torch.Tensor, undersampling_kmask: torch.Tensor
    ) -> torch.Tensor:
        return add_gaussian_noise(
            kdata=kdata, mask=undersampling_kmask,
            noise_var=DataGenerator.get_random_float(
                self.min_gaussian_noise_variance,
                self.max_gaussian_noise_variance))[0]

    def get_corrupted_kdata(
            self,
            x_true: torch.Tensor,
            acceleration_factor_R: int,
            gaussian_noise_standard_deviation_sigma: float,
            coil_sensitivity_map: Optional[torch.Tensor] = None
    ):
        undersampling_kmask = cartesian_mask(
            shape=x_true.shape,     # NOTE: Don't use shape of kdata!!!
            acc=acceleration_factor_R).to(self.device)
        undersampled_kdata = self.EncObj.apply_A(
            x=x_true, csm=coil_sensitivity_map, mask=undersampling_kmask)
        corrupted_kdata = add_gaussian_noise(
            kdata=undersampled_kdata,
            mask=undersampling_kmask,
            noise_var=gaussian_noise_standard_deviation_sigma**2)[0]
        return corrupted_kdata, undersampling_kmask

    def get_corrupted_data(
            self,
            x_true: torch.Tensor,
            acceleration_factor_R: Optional[float] = None,
            gaussian_noise_standard_deviation_sigma: Optional[float] = None,
            coil_sensitivity_map: Optional[torch.Tensor] = None
    ):
        assert x_true.dim() == 2, \
            f"Expected 2D tensor, got {x_true.dim()}D shape {x_true.shape}"
        # Add batch dimension for the encoding object to work.
        x_true = x_true.unsqueeze(0)
        sigma = gaussian_noise_standard_deviation_sigma
        if acceleration_factor_R is None:
            acceleration_factor_R = DataGenerator.get_random_float(
                self.min_acceleration_factor_R,
                self.max_acceleration_factor_R)
        if sigma is None:
            sigma = DataGenerator.get_random_float(
                self.min_gaussian_noise_variance,
                self.max_gaussian_noise_variance)
        corrupted_kdata, undersampling_kmask = self.get_corrupted_kdata(
            x_true=x_true,
            acceleration_factor_R=acceleration_factor_R,
            gaussian_noise_standard_deviation_sigma=sigma,
            coil_sensitivity_map=coil_sensitivity_map)
        corrupted_x = self.EncObj.apply_AH(
            k=corrupted_kdata, csm=coil_sensitivity_map,
            mask=undersampling_kmask)
        # Remove batch dimension before returning.
        return (
            corrupted_x.squeeze(0),
            corrupted_kdata.squeeze(0),
            undersampling_kmask.squeeze(0))
