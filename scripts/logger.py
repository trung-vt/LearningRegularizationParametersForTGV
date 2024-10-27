import os
import torch
import yaml
import wandb
from pathlib import Path
from typing import Dict, Literal, Optional, Any, Union

from utils.makepath import makepath as mkp


class Logger:
    def __init__(
            self, action: Union[Literal["train", "val", "test"], str],
            config: Dict[str, Any], force_overwrite: bool = False) -> None:
        """

        Parameters
        ----------
        action : Union[Literal["train", "val", "test"], str]
            The action to be taken: train, val, or test.
            It could also be a string that begins with one of the above.
            For example, when we test with R=4 and sigma=0.05, the action
            could be "test-R_4-sigma_0_05".
            "train", "val", and "test" are the general actions.
        """
        self.action = action
        self.general_action = action.split("-")[0]
        self.config = config
        log_config = self.config["log"]
        self.save_dir = log_config["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        self.force_overwrite = force_overwrite
        self.current_epoch = self.config["train"]["start_epoch"]
        print(
            f"Action: {self.action}\n" +
            f"Save directory: {self.save_dir}\n" +
            f"Force overwrite: {self.force_overwrite}\n" +
            f"Current epoch: {self.current_epoch}"
        )
        print("Please initialize the logging options.")

    def init_model_saving_options(self, log_config: Dict[str, Any]) -> None:
        self.local_model_saving_interval = \
            log_config["local_model_saving_interval"]
        ratio = log_config["wandb_to_local_ratio"]
        self.wandb_model_saving_interval = \
            ratio * self.local_model_saving_interval
        self.saves_model_by_epoch = log_config["saves_model_by_epoch"]
        self.model_num = log_config["checkpoint"]
        print("Model saving options initialized.")

    def init_metrics_logging_options(self) -> None:
        force_overwrite = self.force_overwrite
        # Only write if not file with the same name does not exist.
        self.epoch_csv_file = self.init_log_file(
            freq="epoch", headers="epoch,loss,psnr,ssim",
            force_overwrite=force_overwrite)
        self.intermediate_csv_file = self.init_log_file(
            freq="intermediate", headers="epoch,iter,loss,psnr,ssim",
            force_overwrite=force_overwrite)

        self.log_freq_by_iter = {}
        self.log_freq_by_iter["intermediate"] = self.config["log"][
            f"intermediate_{self.general_action}_metrics_log_freq_by_iter"
        ]
        self.log_freq_by_iter["epoch"] = self.config["data"][
            f"{self.general_action}_num_samples"
        ] // self.config["data"]["batch_size"]

        print("Metrics logging options initialized.")

    def init_log_file(
            self, freq: str, headers: str, force_overwrite: bool = False
    ) -> Path:
        csv_file = mkp(self.save_dir, f"{self.action}_{freq}_metrics.csv")
        # Train or Val
        # Only write if not file with the same name does not exist.
        if os.path.exists(csv_file):
            print(f"File '{csv_file}' already exists.")
            if force_overwrite:
                print(f"Overwriting the file '{csv_file}'...")
            elif self.action == "test":
                return csv_file
            else:
                raise FileExistsError(
                    "Overwrite is not allowed. " +
                    "Set 'force_overwrite' to True to overwrite.")
        with open(csv_file, "w") as f:
            f.write(f"{headers}\n")
        print(f"File '{csv_file}' initialized.")
        return csv_file

    @staticmethod
    def get_model_size_in_mb(model: torch.nn.Module) -> float:
        # Calculate the total size in bytes
        param_size = sum(
            param.numel() * param.element_size()
            for param in model.parameters())
        buffer_size = sum(
            buffer.numel() * buffer.element_size()
            for buffer in model.buffers())

        total_size_in_bytes = param_size + buffer_size

        # Convert bytes to megabytes
        total_size_in_mb = total_size_in_bytes / (1024 ** 2)

        return total_size_in_mb

    def convert_path_to_str(
            self, config: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        out_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                out_config[key] = self.convert_path_to_str(value)
            elif isinstance(value, Path):
                out_config[key] = str(value)
            else:
                out_config[key] = value
        return out_config

    def log_config_local(self, pdhg_net: torch.nn.Module) -> None:
        if self.action == "test":
            raise ValueError("In testing mode. Will not log config to local.")
        config = self.convert_path_to_str(self.config)
        save_dir = self.config["log"]["save_dir"]
        print(f"Saving config in {save_dir}...")

        def log_to_files():
            with open(mkp(save_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f)
            with open(mkp(save_dir, "config.txt"), "w") as f:
                f.write(str(config))
            with open(mkp(save_dir, "unet.txt"), "w") as f:
                n_trainable_params = sum(
                    p.numel() for p in pdhg_net.parameters()
                )
                f.write(f"Trainable parameters: {n_trainable_params}\n\n")
                unet = pdhg_net.cnn
                f.write(str(unet))
            with open(mkp(
                    save_dir, "pdhg_net.txt"), "w") as f:
                f.write(str(pdhg_net))
        log_to_files()
        print("Config saved")

    def log_data_info(self, data_loader: torch.utils.data.DataLoader) -> None:
        dataset = data_loader.dataset
        save_dir = self.config["log"]["save_dir"]
        print(f"Saving config in {save_dir}...")
        with open(
                mkp(save_dir, f"data_loader_{self.action}.txt"), "w") as f:
            f.write(f"Batch size: {data_loader.batch_size}\n\n")
            f.write(f"Number of batches: {len(data_loader)}\n\n")
            f.write(f"Number of samples: {len(dataset)}\n\n")

    # Optional: Use wandb to log the training process
    # !wandb login
    def init_wandb(self) -> None:
        if self.action == "test":
            raise ValueError("In testing mode. Will not log config to wandb.")
        log_config = self.config["log"]
        project_name = log_config["project"]
        os.environ['WANDB_NOTEBOOK_NAME'] = project_name
        # https://docs.wandb.ai/quickstart
        os.environ['WANDB_MODE'] = log_config["wandb_mode"]
        wandb.login()
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,

            # entity=log_config["wandb_entity"],
            # id=log_config["wandb_run_id"],

            config=self.config,     # track hyperparameters and run metadata
        )

    def log_metrics(
            self, stage: Literal["intermediate", "epoch"],
            metrics: torch.Tensor, iter_idx: Union[int, None]
    ) -> None:
        # NOTE: Assume the metrics are ordered correctly: loss, psnr, ssim
        if wandb.run is not None:
            wandb.log({
                f"{self.action}_{stage}_loss": metrics[0],
                f"{self.action}_{stage}_PSNR": metrics[1],
                f"{self.action}_{stage}_SSIM": metrics[2]
            })
        metrics_str = ""
        # NOTE: Assume the csv file has the correct headers:
        # "epoch,loss,psnr,ssim" or "epoch,iter,loss,psnr,ssim"
        # for key, value in metrics.items():
        for i in range(len(metrics)):
            value = metrics[i].item()
            metrics_str = metrics_str + f",{value}"
        if iter_idx is None:    # Epoch
            with open(self.epoch_csv_file, "a") as f:
                f.write(f"{self.current_epoch+1}{metrics_str}\n")
        else:   # Intermediate
            with open(self.intermediate_csv_file, "a") as f:
                f.write(f"{self.current_epoch+1},{iter_idx+1}{metrics_str}\n")

    def update_and_log_metrics(
            self, stage: Literal["intermediate", "epoch"], iter_idx: int,
            new_metrics: torch.Tensor, running_metrics: torch.Tensor,
            log_freq: Optional[int] = None
    ) -> None:
        running_metrics += new_metrics
        log_freq = self.log_freq_by_iter[stage]
        if (iter_idx+1) % log_freq == 0:
            avg_metrics = running_metrics / log_freq
            self.log_metrics(
                stage=stage, metrics=avg_metrics, iter_idx=iter_idx)
            # Reset running metrics
            running_metrics.zero_()

    def save_model(
            self, pdhg_net: torch.nn.Module, idx: Union[int, None],
            is_final: bool) -> None:
        if self.action != "train":
            raise ValueError("Not in training mode. Will not save model.")
        if is_final:
            model_name = "model_state_dict_final"
            model_path = mkp(self.save_dir, f"{model_name}.pth")
            torch.save(pdhg_net.state_dict(), model_path)
            if wandb.run is not None:
                wandb.log_model(model_path, name=model_name)
            return

        if idx % self.local_model_saving_interval == 0:
            self.model_num += 1
            current_model_name = f"model_state_dict_{self.model_num}"
            model_path = mkp(
                self.save_dir, f"{current_model_name}.pth")
            torch.save(pdhg_net.state_dict(), model_path)

        if idx % self.wandb_model_saving_interval == 0:
            if wandb.run is not None:
                wandb.log_model(model_path, name=current_model_name)
