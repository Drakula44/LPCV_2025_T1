import torch
import torchvision.transforms as T
import json
import os
from PIL import Image
import src.dataset.utils as utils

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
        
        self.target_cat_names = set(target_classes)

        self.name_to_annotation_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}

        self.annotation_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        self.target_cat_ids_annotations = {self.name_to_annotation_id[name] for name in target_classes} 

        self.name_to_local_id = {name: idx for idx, name in enumerate(target_classes)}

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

if (__name__ == '__main__'):

    dataset = COCODataset(
    annotation_file='../data/annotations/annotations/instances_val2017.json',
    image_dir= '../data/val2017/val2017',
    target_classes=[s.lower() for s in utils.GLOBAL_CLASSES],
    transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]))

    print(dataset.target_cat_ids_annotations)
    print(dataset.name_to_annotation_id)
    print(dataset.target_cat_ids_annotations)

    print("Dataset made")

    print(len(dataset))

    classes = [0]*64

    for i in range(0, len(dataset)):
        image, label = dataset[i]
        classes[label] += 1

    for i in range(0, 64):
        print(classes[i])


    # print(label)

    # print(type(image))

    # import matplotlib.pyplot as plt
    # print("Image shape:", image.shape)
    # plt.imshow(image.permute(1, 2, 0))
    # plt.show()