from src.dataset.coco import get_coco, coco_dataset
import torch

dataset = get_coco("../data", validation=True, train=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

print(dataset.class_to_idx)

import skimage as ski
import numpy as np

def show_image(image):
    image = image.cpu().numpy()
    ski.io.imshow(np.transpose(image, (1, 2, 0)))
    ski.io.show()

show_image(next(iter(dataloader))[0][0])