import os
import shutil
import wget
from torchvision import transforms as T
import torch
import json
from PIL import Image

from .utils import GLOBAL_CLASSES


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


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, target_classes, transform=None, condition=None):
        """
        Args:
            annotation_file (str): Path to the COCO JSON annotation file.
            image_dir (str): Path to the directory containing images.
            target_classes (list): List of category names to include.
            transform (callable, optional): Optional transform to apply to cropped images.
            Condition (callable, optional): Optional condition determine if the image should be included.
        """
        
        self.image_dir = image_dir
        self.transform = transform
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        

        # Zeljene kategorije
        self.target_cat_names = set(target_classes)

        # Mapiranje imena kategorija u id JSON fila-a
        self.name_to_annotation_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}

        # Mapiranje id kategorija u JSON fajlu u ime kategorija
        self.annotation_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        # Set IDja u JSON fajlu koji odgovaraju zeljenim kategorijama
        self.target_cat_ids_annotations = {self.name_to_annotation_id[name] for name in target_classes} 

        # Mapiranje imena kategorija u lokalne IDjeve (lokalni ID je redni broj kategorije u listi target_classes)
        self.name_to_local_id = {name: idx for idx, name in enumerate(target_classes)}

        # Mapiranje lokalnih IDjeva u imena kategorija
        self.local_id_to_name = {idx: name for idx, name in enumerate(target_classes)}

        # Filter annotations to include only those of target classes
        self.annotations = [ann for ann in self.coco_data['annotations'] if ann['category_id'] in self.target_cat_ids_annotations]
        
        # Create a mapping from image ID to image file
        self.image_id_to_file = {img['id']: img['file_name'] for img in self.coco_data['images']}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Returns a cropped object image and its label."""
        ann = self.annotations[idx]
        image_id = ann['image_id']
        category_id_annotation = ann['category_id']
        category_name = self.annotation_id_to_name[category_id_annotation]
        local_id = self.name_to_local_id[category_name]
        bbox = ann['bbox']  # Format: [x, y, width, height]
        
        # Load image
        image_path = os.path.join(self.image_dir, self.image_id_to_file[image_id])
        image = Image.open(image_path).convert("RGB")
        
        # Crop object
        x, y, w, h = map(int, bbox)
        cropped_img = image.crop((x, y, x + w, y + h))
        
        # Apply transformations if provided
        if self.transform:
            cropped_img = self.transform(cropped_img)
        
        return cropped_img, local_id

if __name__ == "__main__":
    dataset_dir = "data/coco"
    # download_coco(dataset_dir, validation=True, train=False)
    # extract_dataset(dataset_dir)
    split = "val"

    annotation_file = f'{dataset_dir}/annotations/instances_{split}2017.json'
    image_dir = f'{dataset_dir}/{split}2017'

    dataset = COCODataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        target_classes=[s.lower() for s in GLOBAL_CLASSES],
        transform=T.Compose([T.Resize((224, 224)), T.ToTensor()])
    )

    print(f"Number of images: {len(dataset)}")
