import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

from utils.makepath import makepath as mkp


def get_model_choice_string_description() -> str:
    return (
        "The path to a config file, or " +
        "the ID of one of the model configurations " +
        "as described in the report (choose from " +
        "'u_tv', 'u_tgv_type_1', 'u_tgv_type_2'). " +
        "Currently supported file formats: 'yaml' or 'yml'; 'json'."
    )


def load_config(
        config_choice: Union[str, Dict[str, Any]],
        is_training: bool,
        root_dir: Optional[Union[str, Path]] = mkp("."),
) -> Dict[str, Any]:
    """
    Load the configuration from a file or a prepared configuration.

    Parameters
    ----------
    config_choice : str or dict
        If a dict, the configuration is stored in the dict.
        If not, it should be a string as explained in
        {get_model_choice_string_description()}.
    action : str
        What the model is used for. Choose from 'train' or 'test'.
    """
    if config_choice is None:
        raise ValueError(
            "Please provide a config dict, file path or a model ID.")
    if isinstance(config_choice, dict):
        config = config_choice
        print("Config loaded from dict")
    # choose from the prepared configurations
    else:
        config_choice = Path(config_choice)
        extension = config_choice.suffix
        if extension in [".yaml", ".yml"]:    # parse yaml file
            with open(config_choice, "r") as f:
                # Load all Python objects including pathlib.Path
                config = yaml.load(f, Loader=yaml.FullLoader)

                # # Only load simple, safe data types such as
                # #   dictionaries, lists, strings, integers, and floats
                # config = yaml.safe_load(f)

                print(f"Config loaded from file {config_choice}")
        elif extension == ".json":  # parse json file
            with open(config_choice, "r") as f:
                config = json.load(f)
                print(f"Config loaded from file {config_choice}")
        else:
            raise ValueError(
                f"The config choice '{config_choice}' is unsupported. " +
                "Currently supported file formats are " +
                "'.yaml' or '.yml', '.json'.")
    # print(f"root_dir: {root_dir}")
    # Adjust the paths in the config
    if "log" in config:
        if "save_dir" in config["log"]:
            save_dir = mkp(root_dir, config["log"]["save_dir"])
            # print(f"save_dir: {save_dir}")
            config["log"]["save_dir"] = save_dir
    if "data" in config:
        if "data_path" in config["data"]:
            data_path = mkp(root_dir, config["data"]["data_path"])
            # print(f"data_path: {data_path}")
            config["data"]["data_path"] = data_path
        if "generated_data_path" in config["data"]:
            generated_data_path = mkp(
                root_dir, config["data"]["generated_data_path"])
            # print(f"generated_data_path: {generated_data_path}")
            config["data"]["generated_data_path"] = generated_data_path
    # print(f"Config: {config}")
    return config
