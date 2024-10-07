import torch
from typing import Union, Optional, Literal

from utils.makepath import makepath as mkp
from config.config_loader import load_config
from networks.mri_pdhg_net import MriPdhgNet
from networks.unet import UNet


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

    def load_pretrained_model(self) -> MriPdhgNet:
        is_state_dict = self.config["log"]["is_state_dict"]
        full_path = self.get_full_model_path()
        if is_state_dict:
            model = self.init_new_model()
            state_dict = torch.load(full_path)
            model.load_state_dict(state_dict)
        else:
            model = torch.load(full_path)
        return model.to(self.device())

    def init_new_model(self) -> MriPdhgNet:
        pdhg_config = self.config["pdhg"]
        regularisation = pdhg_config["regularisation"]
        pdhg_net = self.init_mri_pdhg_net(regularisation)
        print(f"PDHG net device: {pdhg_net.device}")
        unet = self.init_unet()
        pdhg_net.cnn = unet.to(self.device())
        return pdhg_net.to(self.device())

    def init_mri_pdhg_net(
            self, pdhg_algorithm: Literal["tv", "tgv"]) -> MriPdhgNet:
        pdhg_config = self.config["pdhg"]
        tgv_pdhg_net = MriPdhgNet(
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
        return tgv_pdhg_net.to(self.device())

    def init_unet(self) -> UNet:
        unet_config = self.config["unet"]
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
        ).to(self.device())
        with torch.no_grad():
            # Make sure the bias works with complex numbers?
            def bias_to_zero(m):
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.data.fill_(0)
            unet.apply(bias_to_zero)
            # force initial lambdas to be closer to lower bound / zero
            unet.c1x1.bias.fill_(-1.0)
        return unet
