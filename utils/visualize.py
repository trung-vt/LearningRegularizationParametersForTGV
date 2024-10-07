import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
from pathlib import Path


def make_colorbar(
        min_val: float, max_val: float,
        leq_min: bool, geq_max: bool,
        cmap: str,
        out_path: Optional[Union[str, Path]] = None) -> None:

    val_range = np.linspace(min_val, max_val, num=1024)

    # Make a rectangular array 64 x 1024 of the values
    val_range_2d = np.tile(val_range, (64, 1))
    plt.imshow(
        val_range_2d, cmap=cmap
    )
    plt.yticks([])  # Turn off y-axis

    # Make x-axis ticks go from min_val to max_val
    xticks = [
        f"{x:.3f}" for x in np.linspace(min_val, max_val, 1024 // 256 + 1)]
    latex_is_enabled = plt.rcParams['text.usetex']
    if leq_min:
        leq_sign = r"$\leq$" if latex_is_enabled else "≤"
        xticks[0] = leq_sign + f" {min_val:.3f}"
    if geq_max:
        geq_sign = r"$\geq$" if latex_is_enabled else "≥"
        xticks[-1] = geq_sign + f" {max_val:.3f}"
    plt.xticks(range(0, 1025, 256), xticks)

    if out_path is not None:
        plt.savefig(
            out_path,
            # transparent=True,
            # format="pdf",
            # facecolor="white",
        )
    # plt.show()
