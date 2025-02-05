import torch
import pydantic

GLOBAL_CLASSES = [
    "Bicycle",
    "Car",
    "Motorcycle",
    "Airplane",
    "Bus",
    "Train",
    "Truck",
    "Boat",
    "Traffic Light",
    "Stop Sign",
    "Parking Meter",
    "Bench",
    "Bird",
    "Cat",
    "Dog",
    "Horse",
    "Sheep",
    "Cow",
    "Elephant",
    "Bear",
    "Zebra",
    "Backpack",
    "Umbrella",
    "Handbag",
    "Tie",
    "Skis",
    "Sports Ball",
    "Kite",
    "Tennis Racket",
    "Bottle",
    "Wine Glass",
    "Cup",
    "Knife",
    "Spoon",
    "Bowl",
    "Banana",
    "Apple",
    "Orange",
    "Broccoli",
    "Hot Dog",
    "Pizza",
    "Donut",
    "Chair",
    "Couch",
    "Potted Plant",
    "Bed",
    "Dining Table",
    "Toilet",
    "TV",
    "Laptop",
    "Mouse",
    "Remote",
    "Keyboard",
    "Cell Phone",
    "Microwave",
    "Oven",
    "Toaster",
    "Sink",
    "Refrigerator",
    "Book",
    "Clock",
    "Vase",
    "Teddy Bear",
    "Hair Drier",
]


class DataloaderConfig(pydantic.BaseModel):
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False


def get_dataloader(dataset, config: DataloaderConfig = None):
    if config is None:
        config = DataloaderConfig()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
