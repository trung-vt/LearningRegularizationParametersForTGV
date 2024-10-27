from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--application", dest="application", type=str,
                    choices=["denoising", "mri"],
                    help="The application to run. " +
                    "Currently supporting 'denoising' and 'mri'.")
parser.add_argument("--config", dest="config", type=str,
                    help="[Required] Path to the config file.")
parser.add_argument("--output_dir", dest="output_dir", type=Path,
                    help="The output directory to store the `.pth` " +
                    "state dict file and other logs. " +
                    "If provided, overwrite the config.")
parser.add_argument("--device", dest="device", type=str,
                    help="The device to use for training. " +
                    "If provided, overwrite the config. " +
                    "Recommend 'cuda' if GPU is available.")
parser.add_argument("--uses_wandb", dest="uses_wandb", type=bool,
                    help="Whether to use WandB for logging. " +
                    "Default to False.",
                    default=False)
parser.add_argument("--logs_local", dest="logs_local", type=bool,
                    help="Whether to log locally. " +
                    "If provided without value, save the config and " +
                    "other logs locally. " +
                    "If not provided, still save the config and " +
                    "other logs locally by default. " +
                    "Need to explicitly set to False to disable.",
                    # Save the config and other logs locally helps
                    # make future reference easier.
                    # That's why the default is True.
                    default=True)
parser.add_argument("--savefile", dest="savefile", type=str,
                    help="The file to save the model state dict and config.")
args = parser.parse_args()
print(f"Initial config choice: {args.config}")

if args.application == "denoising":
    from scripts.denoising.trainer import Trainer
elif args.application == "mri":
    from scripts.mri.trainer import Trainer
else:
    raise ValueError(f"Unsupported application: {args.application}")

trainer = Trainer(args=args, tqdm=tqdm)
# trainer.train()
