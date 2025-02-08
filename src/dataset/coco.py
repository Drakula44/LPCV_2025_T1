import os
import shutil
import wget


def download_coco(dataset_dir: str, validation: bool = False, train: bool = True):
    """Download COCO dataset (Train & Val)"""
    urls = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }
    if validation:
        urls["val"] = "http://images.cocodataset.org/zips/val2017.zip"
    if train:
        urls["train"] = "http://images.cocodataset.org/zips/train2017.zip"

    os.makedirs(dataset_dir, exist_ok=True)

    for split, url in urls.items():
        zip_path = os.path.join(dataset_dir, f"{split}.zip")
        if os.path.exists(zip_path):
            print(f"Skipping {split} dataset download, as it is already downloaded.")
            continue
        print(f"Downloading {split} dataset...")
        wget.download(url, zip_path)


def extract_dataset(dataset_dir: str):
    zip_dir = {
        "annotations": "annotations",
        "val": "val2017",
        "train": "train2017",
    }
    for split, dir_name in zip_dir.items():
        unpack_folder = os.path.join(dataset_dir, dir_name)
        if os.path.exists(unpack_folder):
            print(f"Skipping {split} dataset extraction, as it is already extracted.")
            continue
        zip_path = os.path.join(dataset_dir, f"{split}.zip")
        if not os.path.exists(zip_path):
            print(f"Skipping {split} dataset extraction, as it is not downloaded.")
            continue
        print(f"Extracting {split} dataset...")
        shutil.unpack_archive(zip_path, dataset_dir)


if __name__ == "__main__":
    dataset_dir = "datasets/coco"
    download_coco(dataset_dir, validation=True, train=False)
    extract_dataset(dataset_dir)