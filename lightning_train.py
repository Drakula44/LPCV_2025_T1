import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Import your dataset modules
from src.dataset.coco import get_coco
import src.dataset.DatasetReader as DatasetReader
import src.dataset.utils as utils

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p = 0.3),
    transforms.ColorJitter(brightness = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define dataset paths
root_folder = r"datasets/imagenet/coco_80"
annotation_file = os.path.join(r"datasets", "coco", "annotations", "instances_train2017.json")
image_dir = os.path.join(r"datasets", "coco", "train2017")

# Load datasets
dataset_coco = DatasetReader.COCODataset(
    annotation_file=annotation_file,
    image_dir=image_dir,
    target_classes=[s.lower() for s in utils.GLOBAL_CLASSES],
    transform=transform
)

class_names = [s.lower().replace(' ', '_') for s in utils.GLOBAL_CLASSES]
dataset_imagenet = DatasetReader.CustomImageFolder(root_dir=root_folder, class_names=class_names, transform=transform)

dataset = torch.utils.data.ConcatDataset([dataset_coco, dataset_imagenet])

# Define DataLoader
batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=15, persistent_workers=True, pin_memory=True)

# Define Lightning Model
class LightningModel(pl.LightningModule):
    def __init__(self, num_classes=64, learning_rate=2e-5):
        super(LightningModel, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Define model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="models/",
    filename="mobilenet_v3-{epoch:02d}-{train_loss:.2f}",
    save_top_k=3,
    monitor="train_loss",
    mode="min"
)

from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("logs/", name = "mobilenet_v3", default_hp_metric=False)

# Initialize and train model using Lightning Trainer
model = LightningModel()
trainer = Trainer(
    max_epochs=50,
    devices=1,  # Use only one GPU or CPU
    accelerator="auto",
    strategy="auto",  # Allows later multi-GPU setup without changing code
    precision = "16-mixed",
    callbacks=[checkpoint_callback],
    logger = logger
)

trainer.fit(model, dataloader)