import pandas as pd
from typing import List, Dict, Callable, Any, Optional
import os
import numpy as np
from PIL import Image


class Results:

    def __init__(
        self,
        config: Dict[str, Any],
        data_loader,
        metrics_evaluator,
        tqdm: Optional[Callable] = None
    ):
        """

        Parameters
        ----------

        """
        log_config = config["log"]
        results_config = config["results"]

        self.results_config = results_config
        if "default_results_filename" in results_config:
            default_results_filename = results_config["default_results_filename"]
        else:
            default_results_filename = "metrics"
        self.default_csv_filename = f"{default_results_filename}.csv"
        if "default_denoised_filename" in results_config:
            self.default_denoised_filename = results_config["default_denoised_filename"]
        else:
            self.default_denoised_filename = "denoised"

        checkpoint_num = log_config["checkpoint_num"]

        # Base folder where denoised images are saved.
        # `f"{base_output_folder}/{noisy_img_folder}"`
        #     where `noisy_img_folder = noisy_img_path.replace(f".{extension}", "")`
        self.base_output_folder = os.path.join(model_folder, f"test_results/model_checkpoint_{checkpoint_num}")

        self.data_loader = data_loader
        self.metrics_evaluator = metrics_evaluator
        self.tqdm = tqdm

    def get_denoised_folder(self, noisy_img_path: str) -> str:
        extension = noisy_img_path.split(".")[-1]
        noisy_img_folder = noisy_img_path.replace(f".{extension}", "")
        output_folder = os.path.join(self.base_output_folder, noisy_img_folder)
        return output_folder

    def get_denoised_folders_list(self) -> List[str]:
        denoised_folders_list = []
        # clean_img_paths_list = self.file_paths_dict[0]
        # sigma is the standard deviation of the noise
        # for sigma in self.sigmas:
        for sigma in self.file_paths_dict:
            if sigma == 0:
                continue    # skip clean images
            noisy_img_paths_list = self.file_paths_dict[sigma]
            # for i in tqdm(range(len(clean_img_paths_list))):
            for i in self.tqdm(range(len(noisy_img_paths_list))):
                denoised_folder = self.get_denoised_folder(noisy_img_paths_list[i])
                # denoised_folder = self.create_denoised_folder(clean_img_paths_list[i], sigma)
                denoised_folders_list.append(denoised_folder)
        return denoised_folders_list

    def get_single_example_results(self, denoised_folder:str) -> pd.DataFrame:
        csv_file = os.path.join(denoised_folder, self.default_csv_filename)
        results = pd.read_csv(csv_file)
        # return results

        # TODO: Generalise this for other metrics as well
        results_best = results[results["PSNR"] == results["PSNR"].max()]
        return results_best

    def load_results_dataframe(self) -> pd.DataFrame:
        denoised_folders = self.get_denoised_folders_list()
        df_combined_results = pd.DataFrame()
        iterator = range(len(denoised_folders))
        if self.tqdm is not None:
            iterator = self.tqdm(iterator)
        for i in iterator:
            denoised_folder = denoised_folders[i]
            results = self.get_single_example_results(denoised_folder)
            results["folder"] = denoised_folder
            df_combined_results = pd.concat(
                [df_combined_results, results], ignore_index=True)
        return df_combined_results

    def get_results_for_sigma(
            self, sigma: float, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.load_results_dataframe()
        sigma_str = "_" + str(sigma).replace(".", "_")
        df_sigma = df[df["folder"].str.contains(sigma_str)]
        return df_sigma

    def get_file_with_name(self, folder: str, filename: str) -> str:
        files = os.listdir(folder)
        for file in files:
            if file.startswith(filename):
                return os.path.join(folder, file)
        return None

    def get_full_path(self, path: str) -> str:
        return os.path.join(self.data_config["dataset"], path)

    def get_full_path_clean(self, i: int) -> str:
        return self.get_full_path(self.file_paths_dict[0][i])

    def generate_denoised_images(
        self,
        denoise_and_evaluate: Callable,
        saves_results: bool = True,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        data_paths : List[Tuple[str, str, str]]
            List of tuples with noisy image path, clean image path, and denoised folder path.

        """
        df_combined_results = pd.DataFrame()
        iterator = range(len(self.dataset))
        if self.tqdm is not None:
            iterator = self.tqdm(iterator)
        for i in iterator:
            # noisy_2d, clean_2d = self.dataset[i]
            noisy_2d, clean_2d, mat_file_name, noisy_img_key, sigma = self.dataset[i]

            # print(f"mat_file_name: {mat_file_name}")
            # print(f"noisy_img_key: {noisy_img_key}")

            denoised_2d, df_results = denoise_and_evaluate(noisy_2d, clean_2d)

            # denoised_folder = denoised_folders_list[i]

            sigma_str = str(sigma).replace(".", "_")
            im_size = noisy_2d.shape[0]
            folder_name = f"images_crop_resize_{im_size}_greyscale_noisy_{sigma_str}"

            denoised_folder = f"{self.base_output_folder}/{folder_name}/{mat_file_name}"

            # print(f"denoised_folder: {denoised_folder}")
            if saves_results:
                os.makedirs(denoised_folder, exist_ok=True)

                def save_result(tensor, filename:str, idx:int):
                    np_arr = tensor.detach().cpu().numpy()
                    npy_file = os.path.join(denoised_folder, f"{filename}.npy")
                    np.save(npy_file, np_arr)
                    PIL_file = os.path.join(denoised_folder, f"{filename}.png")
                    img_PIL = Image.fromarray(np.clip(np_arr * 255, 0, 255).astype(np.uint8))
                    img_PIL.save(PIL_file)
                    # plt.subplot(1, 3, idx)
                    # plt.imshow(img_PIL)
                    # plt.title(filename)

                # Save as npy, no clipping, no editing whatsoever
                save_result(noisy_2d, "noisy", 1)
                save_result(denoised_2d, self.default_denoised_filename, 2)
                save_result(clean_2d, "clean", 3)
                # plt.show()
                # print(df_results)

                # Save metrics as csv
                csv_file = os.path.join(denoised_folder, self.default_csv_filename)
                df_results.to_csv(csv_file)

            df_results["folder"] = denoised_folder
            # return df_results
            df_combined_results = pd.concat([df_combined_results, df_results], ignore_index=True)

        # sigmas = self.data_loader.config_loader.get_sigmas_list()
        # num_threads = len(sigmas) # Number of noise levels (sigma values)

        # from multiprocessing.dummy import Pool as ThreadPool
        # pool = ThreadPool(num_threads)
        # results = pool.map(do_iter, range(len(denoised_folders_list)))
        # df_combined_results = pd.concat(results, ignore_index=True)

        return df_combined_results


    def generate_metrics_results_dataframe(
        self,
        # load_img:Callable,
        saves_results:bool = False,
        # num_threads:int = 1
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        data_paths : List[Tuple[str, str, str]]
            List of tuples with noisy image path, clean image path, and denoised folder path.

        """
        df_combined_results = pd.DataFrame()
        denoised_folders_list = self.get_denoised_folders_list()
        # print(f"len(denoised_folders_list): {len(denoised_folders_list)}")
        # print(f"denoised_folders_list: {denoised_folders_list}")

        # def do_iter(i) -> pd.DataFrame:
        for i in tqdm(range(len(denoised_folders_list))):
            denoised_folder = denoised_folders_list[i]
            npy_file = os.path.join(denoised_folder, f"{self.default_denoised_filename}.npy")
            denoised_np = np.load(npy_file)
            # print(f"npy_file: {npy_file}")
            noisy_tensor, clean_tensor = self.dataset[i]
            clean_np = clean_tensor.detach().cpu().numpy()
            df_results = self.metrics_obj.compare_np(denoised_np, clean_np)

            # noisy_np = noisy_tensor.detach().cpu().numpy()
            # plt.subplot(1, 3, 1)
            # plt.imshow(noisy_np, cmap="gray")
            # plt.title("Noisy")
            # plt.subplot(1, 3, 2)
            # plt.imshow(denoised_np, cmap="gray")
            # plt.title("Denoised")
            # plt.subplot(1, 3, 3)
            # plt.imshow(clean_np, cmap="gray")
            # plt.title("Clean")
            # plt.show()
            # print(df_results)


            if saves_results:
                os.makedirs(denoised_folder, exist_ok=True)

                # Save metrics as csv
                csv_file = os.path.join(denoised_folder, self.default_csv_filename)
                df_results.to_csv(csv_file)

            df_results["folder"] = denoised_folder
            # return df_results
            df_combined_results = pd.concat([df_combined_results, df_results], ignore_index=True)

        # sigmas = self.data_loader.config_loader.get_sigmas_list()
        # num_threads = len(sigmas) # Number of noise levels (sigma values)

        # from multiprocessing.dummy import Pool as ThreadPool
        # pool = ThreadPool(num_threads)
        # results = pool.map(do_iter, range(len(denoised_folders_list)))
        # df_combined_results = pd.concat(results, ignore_index=True)

        return df_combined_results


    def generate_results_common(self, saves_results:bool):
        model = self.model_loader.load_model()

        def denoise_and_evaluate_common(noisy_2d, clean_2d):
            noisy_4d = noisy_2d.unsqueeze(0).unsqueeze(0) # Add batch and channel dimensions
            denoised_4d = model(noisy_4d)
            denoised_2d = denoised_4d.squeeze(0).squeeze(0) # Remove batch and channel dimensions
            df_results = self.metrics_obj.compare_torch_2d(denoised_2d, clean_2d)

            return denoised_2d, df_results

        return self.generate_denoised_images(
            denoise_and_evaluate = denoise_and_evaluate_common,
            saves_results = saves_results,
        )

    def generate_results_brute_force(self, scalar_solver, saves_results:bool):
        return self.generate_denoised_images(
            denoise_and_evaluate = scalar_solver,
            saves_results = saves_results,
        )