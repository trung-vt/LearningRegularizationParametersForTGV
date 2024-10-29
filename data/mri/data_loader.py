import torch
from pathlib import Path
from typing import Literal, Optional, Union, Dict, Any

from data.mri.dataset import \
    MriBaseDataset, MriPreProcessedDataset, MriDynamicallyGeneratedDataset
from data.mri.data_generator import DataGenerator
from data.mri.naming import get_test_file_name
from utils.makepath import makepath as mkp


def get_dataset(
        data_config: Dict[str, Any],
        dataset_type: Literal["base", "preprocessed", "dynamically_generated"],
        action: Literal["train", "val", "test"],
        device: Union[str, torch.device],
        root_dir: Union[str, Path] = ".",
        acceleration_factor_R: Optional[int] = None,
        gaussian_noise_standard_deviation_sigma: Optional[float] = None
) -> Union[
        MriBaseDataset, MriPreProcessedDataset, MriDynamicallyGeneratedDataset
]:
    dir = mkp(root_dir, data_config["data_path"])
    scale_factor = data_config["data_scale_factor"]

    file = data_config[f"{action}_file_name"]
    num_samples = data_config[f"{action}_num_samples"]
    file_path = mkp(dir, file)
    x_true_complex = torch.load(file_path, map_location=device)
    x_true_complex = x_true_complex[:num_samples]

    if dataset_type == "base":   # gets scaled ground truth data
        dataset = MriBaseDataset(
            all_x_true_complex=x_true_complex,
            scale_factor=scale_factor)

    elif dataset_type == "preprocessed":
        generated_dir = mkp(dir, action)
        sigma = gaussian_noise_standard_deviation_sigma

        def get_generated_data(data_name: str) -> torch.Tensor:
            filename = get_test_file_name(
                data_name=data_name, action=action,
                acceleration_factor_R=acceleration_factor_R,
                gaussian_noise_standard_deviation_sigma=sigma)
            return torch.load(
                mkp(generated_dir, filename), map_location=device)

        scaled_x_true = x_true_complex * scale_factor
        # scaled_x_true = torch.load(
        #     scaled_x_true_file_path, map_location=device)

        dataset = MriPreProcessedDataset(
            all_rescaled_x_true_complex=scaled_x_true,
            all_x_corrupted=get_generated_data("x_corrupted"),
            all_kdata_corrupted=get_generated_data("kdata_corrupted"),
            all_undersampling_kmasks=get_generated_data(
                "undersampling_kmasks"),
            acc_factor_R=acceleration_factor_R,
            gaussian_noise_sigma=sigma)

    elif dataset_type == "dynamically_generated":
        data_util = DataGenerator(data_config=data_config, device=device)
        dataset = MriDynamicallyGeneratedDataset(
            all_x_true_complex=x_true_complex, scale_factor=scale_factor,
            get_corrupted_data=data_util.get_corrupted_data)

    else:
        raise ValueError(
            "Unknown dataset type. Expected one of " +
            "['base', 'preprocessed', 'dynamically_generated'], " +
            f"but got '{dataset_type}'.")

    return dataset


def get_data_loader(
        data_config: Dict[str, Any],
        action: Literal["train", "val", "test"],
        dataset_type: Literal["base", "preprocessed", "dynamically_generated"],
        device: Union[str, torch.device],
        root_dir: Union[str, Path] = ".",
        acceleration_factor_R: Optional[int] = None,
        gaussian_noise_standard_deviation_sigma: Optional[float] = None,
        sets_generator: bool = False
) -> torch.utils.data.DataLoader:

    sigma = gaussian_noise_standard_deviation_sigma
    dataset = get_dataset(
        data_config=data_config,
        dataset_type=dataset_type,
        action=action,
        device=device,
        root_dir=root_dir,
        acceleration_factor_R=acceleration_factor_R,
        gaussian_noise_standard_deviation_sigma=sigma
    )

    batch_size = data_config["batch_size"] if action != "test" else 1
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(action == "train"),  # shuffle only for training set

        # # NOTE: Sometimes you need to explicitly set the generator
        # #         to the same device for the code to work.
        generator=torch.Generator(device=device) if sets_generator else None,

        # num_workers=0,
        # pin_memory=True
    )
    print(f"{action}_data_loader contains {len(data_loader)} batches.")
    return data_loader
