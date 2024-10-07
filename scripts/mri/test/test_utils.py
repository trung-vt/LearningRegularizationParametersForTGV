import torch
import numpy as np
from scipy.io import savemat
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Colormap
from pathlib import Path
from typing import Literal, Optional, Union, Dict, Any, Tuple, Callable

from utils.makepath import makepath as mkp
from config.config_loader import load_config
from data.mri.data_loader import get_dataset
from networks.mri_pdhg_net import MriPdhgNet
from scripts.mri.model_loader import ModelLoader
from encoding_objects.cart_2d_enc_obj import Cart2DEncObj
from utils.metrics import ImageMetricsEvaluator
from utils.visualize import make_colorbar

from sithom.plot import set_dim


def get_prepared_config(
        model_id: Literal["u_tv", "u_tgv"], root_dir: Union[str, Path],
) -> Dict[str, Any]:
    assert model_id in ["u_tv", "u_tgv"], \
        f"Invalid model_id. Expected 'u_tv' or 'u_tgv', got '{model_id}'"

    # if model_id == "u_tv":
    #     model_name = "mri_model_09_14-14_37-good_TV-sigma_to_0_2-R_from_4"
    # elif model_id == "u_tgv":
    #     model_name = "mri_model_09_12-23_02-good_TGV-sigma_to_0_2-R_from_4"

    config_file = mkp(
        root_dir,
        # "tmp", model_name, "config.yaml")
        "scripts", "mri", "pretrained", model_id, "config.yaml")
    config = load_config(config_file, is_training=False, root_dir=root_dir)
    return config


def get_config_and_model(
        model_id: Literal["u_tv", "u_tgv"],
        device: Union[str, torch.device],
        root_dir: Union[str, Path],
):
    if model_id == "u_tv":
        # model_name = "mri_model_09_14-14_37-good_TV-sigma_to_0_2-R_from_4"
        # state_dict_file = "model_state_dict_20.pth"
        state_dict_file = "u_tv-model_state_dict_20-cpu.pth"
    elif model_id == "u_tgv":
        # model_name = "mri_model_09_12-23_02-good_TGV-sigma_to_0_2-R_from_4"
        # state_dict_file = "model_state_dict_30.pth"
        state_dict_file = "u_tgv-model_state_dict_30-cpu.pth"
    else:
        raise ValueError(
            f"Invalid model_id. Expected 'u_tv' or 'u_tgv', got '{model_id}'")

    config = get_prepared_config(model_id=model_id, root_dir=root_dir)

    config["log"]["model_filename"] = state_dict_file
    model_dir = mkp(
        root_dir,
        # "tmp", model_name)
        "scripts", "mri", "pretrained", model_id)
    config["log"]["save_dir"] = model_dir
    config["device"] = device
    model_loader = ModelLoader(
        config_choice=config, is_training=False)
    net = model_loader.load_pretrained_model()
    net.eval()
    print(f"{model_id} model loaded")
    return config, net


def get_data_config_and_models(
        device: Union[str, torch.device], root_dir: Union[str, Path],
) -> Tuple[MriPdhgNet, MriPdhgNet]:
    config_tv, u_tv_net = get_config_and_model(
        model_id="u_tv", root_dir=root_dir, device=device)
    config_tgv, u_tgv_net = get_config_and_model(
        model_id="u_tgv", root_dir=root_dir, device=device)
    data_config = config_tv["data"]
    return data_config, u_tv_net, u_tgv_net


class TestImageSaver:
    def __init__(
            self,
            # sample_idx: int,
            acc_factor_R: Optional[int],
            gaussian_noise_sigma: Optional[float],
            # metrics_evaluator: ImageMetricsEvaluator,
            complex_to_real_conversion: Literal["abs", "view_as_real"],
            root_dir: Union[str, Path],
            out_dir: Union[str, Path],
            # num_rows: int,
            # num_cols: int,
            # dataset: MriPreProcessedDataset,
            # u_tv_net: MriPdhgNet,
            # u_tgv_net: MriPdhgNet,
            # enc_obj: Cart2DEncObj,
            num_iters: int,
            fraction_of_line_width: float,
            height_to_width_ratio: float,
            x_clip_range: Tuple[float, float],
            kdata_clip_range: Tuple[float, float],
            lambda1_v_clip_range: Tuple[float, float],
            lambda0_w_clip_range: Tuple[float, float],
            lambda_ratio_clip_range: Tuple[float, float],
            device: Union[str, torch.device],
            tqdm_progress_bar: Optional[Callable] = None,
    ):
        self.acc_factor_R = acc_factor_R
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.num_iters = num_iters
        self.fraction_of_line_width = fraction_of_line_width
        self.height_to_width_ratio = height_to_width_ratio
        self.x_clip_range = x_clip_range
        self.kdata_clip_range = kdata_clip_range
        self.lambda1_v_clip_range = lambda1_v_clip_range
        self.lambda0_w_clip_range = lambda0_w_clip_range
        self.lambda_ratio_clip_range = lambda_ratio_clip_range
        self.tqdm_progress_bar = tqdm_progress_bar
        self.out_dir = out_dir

        self.data_config, self.u_tv_net, self.u_tgv_net = \
            get_data_config_and_models(device=device, root_dir=root_dir)

        self.dataset = get_dataset(
            action="test",
            data_config=self.data_config,
            acceleration_factor_R=self.acc_factor_R,
            gaussian_noise_standard_deviation_sigma=self.gaussian_noise_sigma,
            device=device)

        self.enc_obj = Cart2DEncObj()
        self.metrics_evaluator = ImageMetricsEvaluator(
            complex_to_real_conversion=complex_to_real_conversion,
            device=device)
        self.num_rows = 3
        self.num_cols = 4

    def save_colorbar(
            self,
            min_val: Union[int, float],
            max_val: Union[int, float],
            cmap: Union[str, Colormap],
            out_path: Union[str, Path],
            # fraction_of_line_width: float,
    ):
        fraction_of_line_width = self.fraction_of_line_width
        fig = plt.figure()
        height_to_width_ratio = self.height_to_width_ratio
        set_dim(
            fig,
            fraction_of_line_width=fraction_of_line_width,   # Adjust font size
            ratio=height_to_width_ratio   # Height/Width ratio
        )
        print(f"Saving colorbar to {str(out_path)}")
        make_colorbar(
            min_val=min_val, max_val=max_val,
            leq_min=True, geq_max=True,
            cmap=cmap, out_path=out_path)

    def save_as_image(
            self,
            sample_idx: int,
            batch_complex: torch.Tensor,
            image_name: Union[Literal[
                "ground_truth",
                "kdata", "kdata_corrupted", "mask",
                "lambda", "lambda0_w", "lambda1_v",
                "zero_filled", "tv", "tgv", "u_tv", "u_tgv"], str],
            # fraction_of_line_width: float,
            # out_dir: Optional[Union[str, Path]] = ".",
            batch_x_true: Optional[torch.Tensor] = None,

            # # NOTE: temporarily int values for R only to keep file naming simple
            # acc_factor_R: Optional[int] = None,

            # gaussian_noise_sigma: Optional[float] = None,
            # num_iters: Optional[int] = None,
            clip_range: Optional[Tuple[float, float]] = None,

            # metrics_evaluator: Optional[ImageMetricsEvaluator] = None,

            # fig=None,
            # num_rows: Optional[int] = None, num_cols: Optional[int] = None,
            subplot_index: Optional[int] = None,
            cmap: Union[str, Colormap] = "gray",
    ) -> None:
        with torch.no_grad():
            torch.cuda.empty_cache()

        assert len(batch_complex.shape) in [3, 4], \
            f"Expected 3 or 4 dimensions, got {len(batch_complex.shape)}"
        arr_complex = batch_complex.squeeze(0)
        if len(arr_complex.shape) == 3:
            if image_name.startswith("lambda"):
                arr_complex = arr_complex.squeeze(-1)
            else:
                arr_complex = arr_complex.squeeze(0)
        arr_complex = arr_complex.detach().cpu()
        arr_complex_np = arr_complex.numpy()
        arr_real = arr_complex.abs()
        arr_real_np = arr_real.numpy()
        if clip_range is None:
            clip_range = (arr_real_np.min(), arr_real_np.max())
        arr_real_np = np.clip(arr_real_np, clip_range[0], clip_range[1])

        # Normalize the data to the range [0, 1] using Normalize
        norm = matplotlib.colors.Normalize(
            vmin=clip_range[0] if clip_range is not None else arr_real_np.min(),
            vmax=clip_range[1] if clip_range is not None else arr_real_np.max())

        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]

        # Apply the colormap to the normalized data
        normalized_data = cmap(norm(arr_real_np))

        # Convert the colormap output (RGBA) to 8-bit integers (0-255)
        image_data = (normalized_data[:, :, :3] * 255).astype(np.uint8)
        # Strip alpha channel if needed

        # Create a PIL image from the NumPy array
        img_PIL = Image.fromarray(image_data)

        print(f"Saving {image_name} image")
        psnr_str, ssim_str = None, None
        filename = f"sample_{sample_idx}-{image_name}"
        acc_factor_R = self.acc_factor_R
        if acc_factor_R is not None:
            filename += f"-R_{acc_factor_R}"
        gaussian_noise_sigma = self.gaussian_noise_sigma
        if gaussian_noise_sigma is not None:
            filename += f"-sigma_{gaussian_noise_sigma:.2f}"
        if image_name in ["zero_filled", "u_tv", "u_tgv"] \
            or image_name.startswith("tv") \
                or image_name.startswith("tgv"):
            metrics_evaluator = self.metrics_evaluator
            complex_to_real_conversion = \
                metrics_evaluator.complex_to_real_conversion
            psnr, ssim = metrics_evaluator.compute_torch_complex(
                x=batch_complex, x_true=batch_x_true)
            psnr_str = f"{psnr:.2f}"
            ssim_str = f"{ssim:.4f}"
            print(
                f"{complex_to_real_conversion} - {image_name} metrics:",
                f"PSNR = {psnr_str}, SSIM = {ssim_str}")
            num_iters = self.num_iters
            if num_iters is not None:
                filename += f"-T_{num_iters}"
            filename += f"-{complex_to_real_conversion}" + \
                f"-PSNR_{psnr_str}-SSIM_{ssim_str}"

        filename = filename.replace(".", "_")
        out_dir = self.out_dir
        img_PIL.save(mkp(out_dir, filename + ".png"))
        np.save(mkp(out_dir, f"{filename}-complex" + ".npy"), arr_complex_np)
        np.save(mkp(out_dir, f"{filename}-abs" + ".npy"), arr_real_np)
        torch.save(arr_complex, mkp(out_dir, f"{filename}-complex" + ".pt"))
        torch.save(arr_real, mkp(out_dir, f"{filename}-abs" + ".pt"))
        savemat(
            mkp(out_dir, f"{filename}-complex" + ".mat"), {"data": arr_complex_np})
        savemat(
            mkp(out_dir, f"{filename}-abs" + ".mat"), {"data": arr_real_np})
        fraction_of_line_width = self.fraction_of_line_width
        self.save_colorbar(
            min_val=clip_range[0], max_val=clip_range[1], cmap=cmap,
            out_path=mkp(
                out_dir,
                f"{filename}-colorbar-{fraction_of_line_width}".replace(".", "_") +
                ".png"),
            # fraction_of_line_width=fraction_of_line_width
        )
        num_rows = self.num_rows
        num_cols = self.num_cols
        if num_rows is not None and num_cols is not None and \
                subplot_index is not None:
            ax = plt.subplot(num_rows, num_cols, subplot_index)
            # plt.hist(batch_np.flatten(), bins=100)

            ax.imshow(img_PIL)
            title = image_name
            if psnr_str is not None and ssim_str is not None:
                title += f"\nPSNR = {psnr_str}\nSSIM = {ssim_str}"
            plt.title(
                title,
                # y=-0.2   # y < 0 to place title to the bottom
            )
            # plt.axis("off")
            # plt.show()

    def test_sample_and_save_images(
            self,
            # saves: bool,
            sample_idx: int,
            # dataset: MriPreProcessedDataset,
            # u_tv_net: MriPdhgNet, u_tgv_net: MriPdhgNet,
            # enc_obj: Cart2DEncObj,
            # num_iters: int,
            # metrics_evaluator: ImageMetricsEvaluator,
            # fraction_of_line_width: float,
            # out_dir: Union[str, Path] = ".",
            # x_clip_range: Tuple[float, float] = (0, 0.5),
            # kdata_clip_range: Tuple[float, float] = (0, 0.2),
            # lambda1_v_clip_range: Tuple[float, float] = (0, 0.05),
            # lambda0_w_clip_range: Tuple[float, float] = (0, 1),
            # lambda_ratio_clip_range: Tuple[float, float] = (0, 10),
            # tqdm_progress_bar: Optional[Callable] = None,
            # num_rows: int = 3, num_cols: int = 4,
    ) -> None:
        dataset = self.dataset
        u_tv_net = self.u_tv_net
        u_tgv_net = self.u_tgv_net
        enc_obj = self.enc_obj
        num_iters = self.num_iters
        metrics_evaluator = self.metrics_evaluator
        fraction_of_line_width = self.fraction_of_line_width
        out_dir = self.out_dir
        x_clip_range = self.x_clip_range
        kdata_clip_range = self.kdata_clip_range
        lambda1_v_clip_range = self.lambda1_v_clip_range
        lambda0_w_clip_range = self.lambda0_w_clip_range
        lambda_ratio_clip_range = self.lambda_ratio_clip_range
        tqdm_progress_bar = self.tqdm_progress_bar
        num_rows = self.num_rows
        num_cols = self.num_cols

        sample = dataset[sample_idx]
        x_corrupted, x_true, \
            kdata_corrupted, undersampling_kmask = sample

        batch_x_corrupted = x_corrupted.unsqueeze(0)
        batch_x_true = x_true.unsqueeze(0)
        batch_kdata_corrupted = kdata_corrupted.unsqueeze(0)
        batch_undersampling_kmask = undersampling_kmask.unsqueeze(0)

        print(f"batch_x_corrupted.shape = {batch_x_corrupted.shape}")
        print(f"batch_x_true.shape = {batch_x_true.shape}")
        print(f"batch_kdata_corrupted.shape = {batch_kdata_corrupted.shape}")
        print(f"batch_undersampling_kmask.shape = {batch_undersampling_kmask.shape}")

        print("Reconstructing with U-TGV ...")
        lambda_reg_tgv = u_tgv_net.get_lambda_cnn(batch_x_corrupted)
        batch_x_reconstructed_u_tgv = u_tgv_net.pdhg_solver(
            num_iters=num_iters, lambda_reg=lambda_reg_tgv,
            kdata=batch_kdata_corrupted, kmask=batch_undersampling_kmask,
            state=batch_x_corrupted.clone(),
            sigma=u_tgv_net.sigma, tau=u_tgv_net.tau, theta=1.0,
            tqdm_progress_bar=tqdm_progress_bar)

        print("Reconstructing with U-TV ...")
        lambda_reg_tv = u_tv_net.get_lambda_cnn(batch_x_corrupted)
        lambda_reg_tv = lambda_reg_tv["lambda1_v"]
        batch_x_reconstructed_u_tv = u_tv_net.pdhg_solver(
            num_iters=num_iters, lambda_reg=lambda_reg_tv,
            kdata=batch_kdata_corrupted, kmask=batch_undersampling_kmask,
            state=batch_x_corrupted.clone(),
            sigma=u_tv_net.sigma, tau=u_tv_net.tau, theta=1.0,
            tqdm_progress_bar=tqdm_progress_bar)

        batch_kdata = enc_obj.apply_A(batch_x_true, csm=None, mask=None)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))

        self.save_as_image(
            sample_idx=sample_idx, batch_x_true=batch_x_true,
            batch_complex=batch_kdata, image_name="kdata",
            clip_range=kdata_clip_range,
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=1)
        self.save_as_image(
            sample_idx=sample_idx,
            batch_complex=batch_x_true, image_name="ground_truth",
            clip_range=x_clip_range,
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=2)

        self.save_as_image(
            sample_idx=sample_idx,
            batch_complex=batch_undersampling_kmask, image_name="mask",
            # acc_factor_R=dataset.acc_factor_R,
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=3)
        self.save_as_image(
            sample_idx=sample_idx,
            batch_complex=batch_kdata_corrupted, image_name="kdata_corrupted",
            clip_range=kdata_clip_range,
            # acc_factor_R=dataset.acc_factor_R,
            # gaussian_noise_sigma=dataset.gaussian_noise_sigma,
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=4)
        self.save_as_image(
            sample_idx=sample_idx, batch_x_true=batch_x_true,
            # acc_factor_R=dataset.acc_factor_R,
            # gaussian_noise_sigma=dataset.gaussian_noise_sigma,
            batch_complex=batch_x_corrupted, image_name="zero_filled",
            # metrics_evaluator=metrics_evaluator,
            clip_range=x_clip_range,
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=5)

        self.save_as_image(
            sample_idx=sample_idx,
            batch_complex=lambda_reg_tv, image_name="lambda",
            clip_range=lambda1_v_clip_range,
            # acc_factor_R=dataset.acc_factor_R,
            # gaussian_noise_sigma=dataset.gaussian_noise_sigma,
            cmap="magma",
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=6)
        self.save_as_image(
            sample_idx=sample_idx, batch_x_true=batch_x_true,
            batch_complex=batch_x_reconstructed_u_tv, image_name="u_tv",
            # metrics_evaluator=metrics_evaluator,
            clip_range=x_clip_range,
            # acc_factor_R=dataset.acc_factor_R,
            # gaussian_noise_sigma=dataset.gaussian_noise_sigma,
            # num_iters=num_iters,
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=7)

        self.save_as_image(
            sample_idx=sample_idx,
            batch_complex=lambda_reg_tgv["lambda0_w"], image_name="lambda0_w",
            clip_range=lambda0_w_clip_range,
            # acc_factor_R=dataset.acc_factor_R,
            # gaussian_noise_sigma=dataset.gaussian_noise_sigma,
            cmap="magma",
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=8)
        self.save_as_image(
            sample_idx=sample_idx,
            batch_complex=lambda_reg_tgv["lambda1_v"], image_name="lambda1_v",
            clip_range=lambda1_v_clip_range,
            # acc_factor_R=dataset.acc_factor_R,
            # gaussian_noise_sigma=dataset.gaussian_noise_sigma,
            cmap="magma",
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=9)
        self.save_as_image(
            sample_idx=sample_idx, batch_x_true=batch_x_true,
            clip_range=x_clip_range,
            batch_complex=batch_x_reconstructed_u_tgv, image_name="u_tgv",
            # metrics_evaluator=metrics_evaluator,
            # acc_factor_R=dataset.acc_factor_R,
            # gaussian_noise_sigma=dataset.gaussian_noise_sigma,
            # num_iters=num_iters,
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=10)

        lambda0_w_over_lambda1_v = \
            lambda_reg_tgv["lambda0_w"] / lambda_reg_tgv["lambda1_v"]
        self.save_as_image(
            sample_idx=sample_idx,
            batch_complex=lambda0_w_over_lambda1_v,
            image_name="lambda0_w_over_lambda1_v",
            clip_range=lambda_ratio_clip_range,
            # acc_factor_R=dataset.acc_factor_R,
            # gaussian_noise_sigma=dataset.gaussian_noise_sigma,
            cmap="magma",
            # out_dir=out_dir,
            # fig=fig,
            # fraction_of_line_width=fraction_of_line_width,
            # num_rows=num_rows, num_cols=num_cols,
            subplot_index=11)

        with torch.no_grad():
            torch.cuda.empty_cache()
