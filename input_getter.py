# Apstraktna klasa za dobijanje ulaznih slika, za sada samo koristim random i onu njihovu sliku, ideja je da se kada se napravi
# dataset da se nekako izvlaci iz njega

import numpy as np
from PIL import Image
import requests
import torch

from abc import ABC, abstractmethod

class input_getter(ABC):
    @abstractmethod
    def get_input(self):
        pass

class random_input_getter(input_getter):
    def get_input(self):
        return torch.rand(1, 3, 224, 224)

class mug_image_getter(input_getter):
    def get_input(self):
        sample_image_url = (
        "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
        )
        response = requests.get(sample_image_url, stream=True)
        response.raw.decode_content = True
        image = Image.open(response.raw).resize((224, 224))
        input_array = np.expand_dims(
            np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
        )
        return input_array
    
def marko():
    return "ja"