import subprocess
import os
import json
import shutil
from tqdm import tqdm
import torchvision
from utils import GLOBAL_CLASSES


def download_coco(dataset_dir: str, validation: bool = False, train: bool = True):
    """Download COCO dataset (Train & Val)"""
    print("🚀 Downloading COCO dataset...")

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
        subprocess.run(["wget", "-O", zip_path, url])

        subprocess.run(["unzip", "-q", "-d", dataset_dir, zip_path])
        os.remove(zip_path)

    print("✅ COCO dataset downloaded and extracted!")


def move_images_to_class_folders(dataset_dir, split, classes):
    """Move COCO images into class folders based on labels."""
    annotations_file = os.path.join(
        dataset_dir, "annotations", f"instances_{split}2017.json"
    )

    # Load COCO annotations
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    classes = [cls.lower() for cls in classes]
    # Map category ID to category name
    category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    print(category_map)

    # Keep only the required classes
    category_map = {k: v for k, v in category_map.items() if v in classes}

    print(category_map)

    filtered_annotations = [
        ann for ann in coco_data["annotations"] if ann["category_id"] in category_map
    ]

    if not filtered_annotations:
        print(f"⚠️ No images found for {split}! Check annotation file.")

    for ann in tqdm(filtered_annotations, desc=f"Moving {split} images"):
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        category_name = category_map[category_id]
        file_name = next(
            (img["file_name"] for img in coco_data["images"] if img["id"] == image_id),
            None,
        )

        if file_name is None:
            print(f"⚠️ Warning: No file found for image ID {image_id}")
            continue

        src_path = os.path.join(dataset_dir, f"{split}2017", file_name)
        dest_folder = os.path.join(dataset_dir, split, category_name)
        dest_path = os.path.join(dest_folder, file_name)

        # Ensure class folder exists
        os.makedirs(dest_folder, exist_ok=True)

        # Move the file only if it exists
        if os.path.exists(src_path):
            os.rename(src_path, dest_path)
            print(f"✅ Moved: {src_path} → {dest_path}")
        else:
            print(f"⚠️ File missing: {src_path}")

    shutil.rmtree(os.path.join(dataset_dir, f"{split}2017"))


def coco_dataset():
    dataset = torchvision.datasets.ImageFolder(
        root="data/coco/val",
        transform=torchvision.transforms.ToTensor(),
    )
    return dataset


def get_coco(dataset_dir, validation: bool = False, train: bool = True):
    coco_dir = os.path.join(dataset_dir, "coco")
    if not os.path.exists(coco_dir):
        download_coco(coco_dir, validation=validation, train=train)
        if validation:
            move_images_to_class_folders(
                dataset_dir=dataset_dir,
                split="val",
                classes=GLOBAL_CLASSES,
            )
        if train:
            move_images_to_class_folders(
                dataset_dir=dataset_dir,
                split="train",
                classes=GLOBAL_CLASSES,
            )
    return coco_dataset()
