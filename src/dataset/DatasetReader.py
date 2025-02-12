import torch
import torchvision.transforms as T
from torchvision import datasets
import json
import os
from PIL import Image
import random
# import utils as utils

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

# if (__name__ == '__main__'):
#     import utils as utils
#     from torchvision import transforms
#     from torch.utils.data import DataLoader
#     import skimage as ski
#     import numpy as np
#     # %matplotlib pyqt
#     # import matplotlib
#     # matplotlib.use('Qt5Agg')
#     import matplotlib.pyplot as plt
#     import utils as utils

#     dataset = COCODataset(
#     annotation_file='../data/annotations/annotations/instances_val2017.json',
#     image_dir= '../data/val2017/val2017',
#     target_classes=[s.lower() for s in utils.GLOBAL_CLASSES],
#     transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]))

#     print(dataset.target_cat_ids_annotations)
#     print(dataset.name_to_annotation_id)
#     print(dataset.target_cat_ids_annotations)

#     class_names = [s.lower().replace(' ', '_') for s in utils.GLOBAL_CLASSES]

#     # Path to the root folder containing subfolders (each representing a category)
#     root_folder = '/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet/coco_80'

#     # Define your transforms (you can add any preprocessing steps here)
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),  # Example of resizing all images
#         transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
#     ])

#     # Create the custom dataset
#     dataset = CustomImageFolder(root_dir=root_folder, class_names=class_names, transform=transform)

#     print("Dataset made")

#     print(len(dataset))

#     classes = [0]*64

#     for i in range(0, len(dataset)):
#         image, label = dataset[i]
#         classes[label] += 1

#     for i in range(0, 64):
#         print(classes[i])


class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, where each class is in a separate subfolder.
            class_names (list): List of class names, where the index of each class corresponds to the label.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        # Create the ImageFolder dataset
        self.image_folder = datasets.ImageFolder(root=root_dir, transform=transform, loader=Image.open)
        
        # Map folder names to their indices (labels)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        
        self.name_to_local_id = {name: idx for idx, name in enumerate(class_names)}
        self.imagenet_id_to_local_id = {}
        for cl in self.image_folder.classes:
            id_imgnet = self.image_folder.class_to_idx[cl]

            id_local = self.name_to_local_id[cl]
            self.imagenet_id_to_local_id[id_imgnet] = id_local

        
        # Update the labels in the image_folder dataset based on the provided class_names
        self.image_folder.samples = [
            (path, self.imagenet_id_to_local_id[class_id]) 
            for path, class_id in self.image_folder.samples
        ]
        
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        # Get an image and label from the ImageFolder dataset
        image, label = self.image_folder[idx]
        
        # if self.transform:
        #     image = self.transform(image)
        
        return image, label

# if __name__ == "__main__":
#     from torchvision import transforms
#     from torch.utils.data import DataLoader
#     import skimage as ski
#     import numpy as np
#     # %matplotlib pyqt
#     # import matplotlib
#     # matplotlib.use('Qt5Agg')
#     import matplotlib.pyplot as plt
#     import utils as utils

#     # Example class names you want to pass
#     # class_names = utils.GLOBAL_CLASSES
#     class_names = [s.lower().replace(' ', '_') for s in utils.GLOBAL_CLASSES]

#     # Path to the root folder containing subfolders (each representing a category)
#     root_folder = '/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet/coco_80'

#     # Define your transforms (you can add any preprocessing steps here)
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),  # Example of resizing all images
#         transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
#     ])

#     # Create the custom dataset
#     dataset = CustomImageFolder(root_dir=root_folder, class_names=class_names, transform=transform)

#     # Create a DataLoader for batching and shuffling
#     # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # Example usage: Iterating through the dataloader
#     # for images, labels in dataloader:
#         # print(images.shape, labels.shape)  # For example, (batch_size, 3, 128, 128) for images
#     # print(type(dataset[0][0]))

#     print(dataset[0][0].shape)
#     print(dataset[0][1])

#     plt.figure() 
#     plt.imshow(np.transpose(dataset[0][0], (1, 2, 0)))
#     plt.show()

if (__name__ == '__main__'):
    import utils as utils
    from torchvision import transforms
    from torch.utils.data import DataLoader, ConcatDataset
    import skimage as ski
    import numpy as np
    # %matplotlib pyqt
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import utils as utils


    datasetCOCO = COCODataset(
    annotation_file = "/home/centar15-desktop1/LPCV_2025_T1/datasets/coco/annotations/instances_train2017.json", 
    image_dir= '/home/centar15-desktop1/LPCV_2025_T1/datasets/coco/train2017',
    target_classes=[s.lower() for s in utils.GLOBAL_CLASSES],
    transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]))

    class_names = [s.lower().replace(' ', '_') for s in utils.GLOBAL_CLASSES]

    # Path to the root folder containing subfolders (each representing a category)
    root_folder = '/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet/coco_80'

    # Define your transforms (you can add any preprocessing steps here)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Example of resizing all images
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    ])

    # Create the custom dataset
    datasetImageNet = CustomImageFolder(root_dir=root_folder, class_names=class_names, transform=transform)

    dataset = ConcatDataset([datasetCOCO, datasetImageNet])


    print(len(dataset))

    classes = [0]*64

    for i in range(0, len(dataset)):
        image, label = dataset[i]
        classes[label] += 1
        if(i%20000 == 0):
            print(i)
            

    for i in range(0, 64):
        print(classes[i])
