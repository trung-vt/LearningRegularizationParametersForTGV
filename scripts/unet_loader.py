import torch
from typing import Any, Dict, Optional, Union

from networks.unet import UNet
from networks.unet_2d import UNet2d


class UnetLoader:

    def __init__(
            self, unet_config: Dict[str, Any],
            device: Optional[Union[str, torch.device]] = None):
        """
        Parameters
        ----------
        config_choice : str or dict
            If a dict, the configuration is stored in the dict.
            If not, it should be a string as explained in
            {config.config_loader.get_model_choice_string_description()}.
        is_training : bool
            Whether the model is used for training or testing.
        device : str or torch.device, optional
            The device to use for training or testing.
            If not provided, the device is loaded from the config.
        """
        self.unet_config = unet_config
        if device is not None:
            self.device = device
        print(f"Loading model on device: {self.device}")

    # New implementation
    def init_unet_2d(self, uses_complex_numbers: bool) -> UNet2d:
        unet_config = self.unet_config
        unet = UNet2d(
            in_channels=unet_config["in_channels"],
            out_channels=unet_config["out_channels"],
            init_filters=unet_config["init_filters"],
            n_blocks=unet_config["n_blocks"],
            activation=unet_config["activation"],
            downsampling_kernel=unet_config["downsampling_kernel"],
            downsampling_mode=unet_config["downsampling_mode"],
            upsampling_kernel=unet_config["upsampling_kernel"],
            upsampling_mode=unet_config["upsampling_mode"],
        ).to(self.device)
        if uses_complex_numbers:
            self.prepare_unet_for_complex_numbers(unet)
        return unet

    # From previous paper
    def init_unet(self, uses_complex_numbers: bool) -> UNet:
        unet_config = self.unet_config
        unet = UNet(
            dim=2,
            n_ch_in=unet_config["in_channels"],
            n_ch_out=unet_config["out_channels"],
            n_filters=unet_config["init_filters"],
            n_enc_stages=unet_config["n_blocks"],
            n_convs_per_stage=2,
            kernel_size=3,
            res_connection=False,
            bias=True,
            padding_mode="zeros"
        ).to(self.device)
        if uses_complex_numbers:
            self.prepare_unet_for_complex_numbers(unet)
        return unet

    @staticmethod
    def prepare_unet_for_complex_numbers(unet):
        with torch.no_grad():
            # Make sure the bias works with complex numbers?
            def bias_to_zero(m):
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.data.fill_(0)
            unet.apply(bias_to_zero)
            # force initial lambdas to be closer to lower bound / zero
            unet.c1x1.bias.fill_(-1.0)
