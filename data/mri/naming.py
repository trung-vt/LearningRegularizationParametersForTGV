from typing import Literal


def get_test_dir() -> str:
    return "test"


def get_test_file_name(
        data_name: Literal[
            "x_corrupted", "kdata_corrupted", "undersampling_kmasks"],
        action: Literal["train", "val", "test"],
        args: str,
        gaussian_noise_standard_deviation_sigma: float
) -> str:
    return f"{data_name}_{action}-" + \
        f"{args}-" + \
        f"sigma_{gaussian_noise_standard_deviation_sigma:.2f}".replace(
            ".", "_") + ".pt"
