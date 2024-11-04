import torch
from typing import Union, Optional, Literal
from pathlib import Path

from utils.makepath import makepath as mkp
from config.config_loader import load_config
from networks.mri_pdhg_net import MriPdhgNet
# from networks.denoising_pdhg_net import DenoisingPdhgNet
from scripts.unet_loader import UnetLoader


class ModelLoader:

    def __init__(
            self, config_choice: Union[str, dict],
            is_training: bool,
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
        self.is_training = is_training
        self.config = load_config(config_choice, is_training)
        if device is not None:
            self.config["device"] = device
        print(f"Loading model on device: {self.config['device']}")
        self.unet_loader = UnetLoader(
            unet_config=self.config["unet"],
            device=self.config["device"]
        )

    def device(self) -> Union[str, torch.device]:
        return self.config["device"]

    def get_full_model_path(self) -> str:
        log_config = self.config["log"]
        model_folder = log_config["save_dir"]
        if self.is_training:
            if log_config["is_state_dict"]:
                mid_substr = "state_dict_"
            else:
                mid_substr = log_config["mid_substr"]
            model_checkpoint = log_config["checkpoint"]
            model_filename = f"model_{mid_substr}{model_checkpoint}.pt" + \
                ("h" if log_config["is_state_dict"] else "")
        else:
            model_filename = log_config["model_filename"]
        full_model_path = mkp(model_folder, model_filename)
        return full_model_path

    def load_pretrained_model(
            self, root_dir: Optional[Union[str, Path]], model: torch.nn.Module
    ) -> torch.nn.Module:
        is_state_dict = self.config["log"]["is_state_dict"]
        full_path = mkp(root_dir, self.get_full_model_path())
        if is_state_dict:
            state_dict = torch.load(
                full_path,
                map_location=self.device()
            )
            model.load_state_dict(state_dict)
            print(f"Loaded model state dict from {full_path}")
        else:
            model = torch.load(
                full_path,
                map_location=self.device()
            )
            print(f"Loaded model from {full_path}")
        return model

    # def load_pretrained_denoising_model(
    #         self, root_dir: Optional[Union[str, Path]]) -> DenoisingPdhgNet:
    #     denoising_pdhg_net = self.init_new_denoising_model()
    #     return self.load_pretrained_model(root_dir, denoising_pdhg_net)

    def load_pretrained_mri_model(
            self, root_dir: Union[str, Path] = ".") -> MriPdhgNet:
        mri_pdhg_net = self.init_new_mri_model()
        return self.load_pretrained_model(root_dir, mri_pdhg_net)

    # def init_new_denoising_model(self) -> DenoisingPdhgNet:
    #     pdhg_config = self.config["pdhg"]
    #     regularisation = pdhg_config["regularisation"]
    #     pdhg_net = self.init_denoising_pdhg_net(regularisation)
    #     print(f"PDHG net device: {pdhg_net.device}")
    #     # unet = self.unet_loader.init_unet_2d(
    #     unet = self.unet_loader.init_unet(
    #         uses_complex_numbers=True)
    #     pdhg_net.cnn = unet.to(self.device())
    #     return pdhg_net.to(self.device())

    def init_new_mri_model(self) -> MriPdhgNet:
        pdhg_config = self.config["pdhg"]
        regularisation = pdhg_config["regularisation"]
        pdhg_net = self.init_mri_pdhg_net(regularisation)
        print(f"PDHG net device: {pdhg_net.device}")
        # print("Using old U-Net implementation!")
        # unet = self.unet_loader.init_unet(
        print("Using my U-Net implementation!")
        unet = self.unet_loader.init_unet_2d(
            uses_complex_numbers=True)
        pdhg_net.cnn = unet.to(self.device())
        return pdhg_net.to(self.device())

    # def init_denoising_pdhg_net(
    #         self, pdhg_algorithm: Literal["tv", "tgv"]) -> DenoisingPdhgNet:
    #     pdhg_config = self.config["pdhg"]
    #     pdhg_net = DenoisingPdhgNet(
    #         device=self.device(),
    #         pdhg_algorithm=pdhg_algorithm,
    #         T=pdhg_config["T"],
    #         cnn=None,
    #         params_config=pdhg_config["params"],
    #         # Bounds for the lambda values
    #         low_bound=pdhg_config["low_bound"],
    #         up_bound=pdhg_config["up_bound"],
    #         # NOTE: could not use "constant" or "zeros" padding for some reason
    #         padding="reflect"
    #     )
    #     return pdhg_net.to(self.device())

    def init_mri_pdhg_net(
            self, pdhg_algorithm: Literal["tv", "tgv"]) -> MriPdhgNet:
        pdhg_config = self.config["pdhg"]
        pdhg_net = MriPdhgNet(
            device=self.device(),
            pdhg_algorithm=pdhg_algorithm,
            T=pdhg_config["T"],
            cnn=None,
            params_config=pdhg_config["params"],
            # Bounds for the lambda values
            low_bound=pdhg_config["low_bound"],
            up_bound=pdhg_config["up_bound"],
            # NOTE: could not use "constant" or "zeros" padding for some reason
            padding="reflect"
        )
        return pdhg_net.to(self.device())
