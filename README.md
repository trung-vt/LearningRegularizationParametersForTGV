# ... (Title of the Paper)

## Quick start

If you have not installed the requirements:
```bash
# Assume you are in the root directory of the repository
# and, preferably, you are in a virtual environment.
pip install -r requirements.txt
```

Start training:
```bash
# Assume you are in the root directory of the repository
# and you have installed the requirements.
python scripts/mri/train_mri.py --config config/example_mri_tgv_config.yaml --device cpu
```

This repository contains the code for the paper "..." (link to the paper) by ...

The paper presents ... In the paper, we use ...


## The Method

The overall approach works as follows...

![equation](...)

which we then (approximately) solve with ...

<img src="figures/....png" width="70%"/>

The weights of the network...

Below, you can see exemplary results for ...

## Code

You will find code for ...

## Citation and Acknowledgement

The paper is ...

If you use the code for your work or if you found the code useful, please cite:

@article{...}
