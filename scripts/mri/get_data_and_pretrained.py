import gdown

from utils.makepath import makepath


if __name__ == "__main__":
    train_data_id = ...
    val_data_id = ...
    test_data_id = ...

    u_tv_id = ...
    u_tgv_id = ...

    names = ["train_data", "val_data", "test_data", "u_tv", "u_tgv"]
    ids = [train_data_id, val_data_id, test_data_id, u_tv_id, u_tgv_id]
    dirs = ["data", "data", "data", "pretrained", "pretrained"]

    root_dir = "."

    for i in range(len(names)):
        name = names[i]
        id = ids[i]
        dir = dirs[i]
        output = makepath(root_dir, dir, f"{name}.pth")
        gdown.download(
            f"https://drive.google.com/uc?id={id}", output, quiet=False)
        print(f"Downloaded {name} model to {output}")
