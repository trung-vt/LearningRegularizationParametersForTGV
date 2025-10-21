from box import Box
import torch
from typing import Optional, Dict, Any, Union

from utils.mask_funcs import cartesian_mask, low_field_mask
from utils.noise_functions import CartesianSamplingOperator2D, LowFieldSamplingOperator2D, add_gaussian_noise
from encoding_objects.cart_2d_enc_obj import Cart2DEncObj


class DataGenerator:
    def __init__(
            self,
            data_config: Box,
            device: Union[str, torch.device]):
        self.data_config = data_config
        if not data_config:
            data_config = Box(undersampling='cartesian')  # Default to cartesian if no config provided
        if data_config.undersampling == 'cartesian':
            self.sampling_op = CartesianSamplingOperator2D()
        elif data_config.undersampling == 'low_field':
            self.sampling_op = LowFieldSamplingOperator2D()
        self.EncObj = Cart2DEncObj()
        self.device = device

    @staticmethod
    def get_random_int(min_val: int, max_val: int) -> int:
        """

        Parameters
        ----------
        min_val : int
            Minimum value of the random integer.
        max_val : int
            Maximum value of the random integer.
            Note: The random integer will be in the range [min_val, max_val]
            (inclusive).
        """
        return torch.randint(min_val, max_val + 1, (1,)).item()

    @staticmethod
    def get_random_float(min_val: float, max_val: float) -> float:
        """

        Parameters
        ----------
        min_val : float
            Minimum value of the random float.
        max_val : float
            Maximum value of the random float.
            The random float will be in the range [min_val, max_val].
        """
        return (torch.rand(1) * (max_val - min_val) + min_val).item()

    def get_gaussian_noise_standard_deviation_sigma(self, config: Box) -> float:
        if 'sigma' not in config:
            config.sigma = DataGenerator.get_random_float(
                self.data_config.min_standard_deviation_sigma,
                self.data_config.max_standard_deviation_sigma)
        return config.sigma

    def get_acceleration_factor(self, config: Box) -> int:
        if 'R' not in config:
            config.R = DataGenerator.get_random_int(
                self.data_config.min_acceleration_factor_R,
                self.data_config.max_acceleration_factor_R)
        return config.R

    def get_super_resolution_factor(self, config: Box) -> float:
        if 'super_res' not in config:
            config.super_res = self.data_config.super_resolution_factor
        return config.super_res

    def get_corrupted_kdata(
            self,
            x_true: torch.Tensor,
            config: Box = None,
            coil_sensitivity_map: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, Box]:
        if self.data_config.undersampling == 'cartesian':
            undersampling_kmask = cartesian_mask(
                shape=x_true.shape,  # NOTE: Assume shape of k-data is same as image shape
                acc=self.get_acceleration_factor(config)).to(self.device)
        elif self.data_config.undersampling == 'low_field':
            undersampling_kmask = low_field_mask(
                shape=x_true.shape,
                super_resolution_factor=self.get_super_resolution_factor(config)).to(self.device)
        else:
            raise ValueError(
                f"Unknown undersampling method: {self.data_config.undersampling}. "
                "Expected one of ['cartesian', 'low_field'].")

        undersampled_kdata = self.EncObj.apply_A(
            x=x_true, csm=coil_sensitivity_map, mask=undersampling_kmask)
        corrupted_kdata = add_gaussian_noise(
            kdata=undersampled_kdata,
            mask=undersampling_kmask,
            sampling_op=self.sampling_op,
            noise_var=self.get_gaussian_noise_standard_deviation_sigma(config)**2,
        )[0]
        return corrupted_kdata, undersampling_kmask, config

    def get_corrupted_data(
            self,
            x_true: torch.Tensor,
            config: Box = None,
            coil_sensitivity_map: Optional[torch.Tensor] = None
    ):
        assert x_true.dim() == 2, \
            f"Expected 2D tensor, got {x_true.dim()}D shape {x_true.shape}"
        # Add batch dimension for the encoding object to work.
        x_true = x_true.unsqueeze(0)  # (coils, Nx, Ny) --> (1, coils, Nx, Ny)
        if not config:
            config = Box()
        corrupted_kdata, undersampling_kmask, config = self.get_corrupted_kdata(
            x_true=x_true,
            config=config,
            coil_sensitivity_map=coil_sensitivity_map)
        corrupted_x = self.EncObj.apply_AH(
            k=corrupted_kdata, csm=coil_sensitivity_map,
            mask=undersampling_kmask)
        return (
            corrupted_x.squeeze(0),
            corrupted_kdata.squeeze(0),
            undersampling_kmask.squeeze(0),
            config
        )

    def get_zero_shot_corrupted_data(
            self,
            x_true: torch.Tensor,
            kmask: torch.Tensor,
            config: Box = None,
            coil_sensitivity_map: Optional[torch.Tensor] = None
    ):
        assert x_true.dim() == 2, \
            f"Expected 2D tensor, got {x_true.dim()}D shape {x_true.shape}"
        # Add batch dimension for the encoding object to work.
        x_true = x_true.unsqueeze(0)  # (coils, Nx, Ny) --> (1, coils, Nx, Ny)
        if not config:
            config = Box()
        corrupted_kdata, undersampling_kmask, config = self.get_zero_shot_data(
            x_true=x_true,
            config=config,
            coil_sensitivity_map=coil_sensitivity_map)
        corrupted_x = self.EncObj.apply_AH(
            k=corrupted_kdata, csm=coil_sensitivity_map,
            mask=undersampling_kmask)
        return (
            corrupted_x.squeeze(0),
            corrupted_kdata.squeeze(0),
            undersampling_kmask.squeeze(0),
            config
        )
