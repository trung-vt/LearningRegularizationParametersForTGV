from typing import Dict, Any, List


class DataConfig:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_loading_method(self) -> str:
        return self.config["loading_method"]

    def get_sigmas(self, sigmas_str: str) -> List[float]:
        # Convert string "[0.1, 0.15, 0.2]" to list [0.1, 0.15, 0.2]
        sigmas = [float(sigma) for sigma in sigmas_str[1:-1].split(", ")]
        return sigmas

    def get_min_sigma(self) -> float:
        return self.config["min_sigma"]

    def get_max_sigma(self) -> float:
        return self.config["max_sigma"]

    def get_dataset_name(self) -> str:
        return self.config["dataset"]

    def get_base_path(self) -> str:
        return self.config["data_path"]

    def get_generated_data_path(self) -> str:
        return self.config["generated_data_path"]

    def get_samples_num(self, dataset_type: str) -> int:
        if f"{dataset_type}_num_samples" not in self.config:
            return None
        return self.config[f"{dataset_type}_num_samples"]

    def get_ratio(self, dataset_type: str) -> float:
        if f"{dataset_type}_ratio" not in self.config:
            return None
        ratio = self.config[f"{dataset_type}_ratio"]
        assert 0 <= ratio <= 1, f"Expected ratio to be in [0, 1]. Got {ratio}"
        return self.config[f"{dataset_type}_ratio"]

    def get_first_sample_index(self, dataset_type: str) -> int:
        key = f"{dataset_type}_first_sample_index"
        if key not in self.config:
            # return None
            return 0    # Assume start from the first sample
        return self.config[key]

    def get_img_size(self) -> int:
        return self.config["resize_square"]

    def get_sigmas_list(self) -> List[float]:
        sigmas = self.config["sigmas"]
        if isinstance(sigmas, list):
            return sigmas
        assert isinstance(sigmas, str), \
            f"Expected sigmas to be a list or a string. Got {type(sigmas)}"
        return self.get_sigmas(sigmas)

    def is_dynamic(self) -> bool:
        return self.config["is_dynamic"]

    def get_batch_size(self) -> int:
        return self.config["batch_size"]

    def get_random_seed(self) -> int:
        return self.config["random_seed"]
