import torch
import torchvision.models as models
import torcheval
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
import pydantic
from pprint import pprint
from tqdm import tqdm

from dataset.coco import get_coco
from dataset.utils import DataloaderConfig, get_dataloader, GLOBAL_CLASSES
from dataset.imagenet import get_class1k


class MobileNet64(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        classes = [cls.lower() for cls in GLOBAL_CLASSES]
        self.mobile_net = models.mobilenet_v2(*args, pretrained=True, **kwargs)
        image_net_classes = get_class1k()
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(image_net_classes)
        }
        self.idx_to_class = {
            idx: class_name for idx, class_name in enumerate(image_net_classes)
        }
        pprint(self.class_to_idx)
        self.filter_indices = [
            self.class_to_idx.get(class_name, 0) for class_name in classes
        ]

    def forward(self, x):
        tmp = self.mobile_net(x)
        return tmp[..., self.filter_indices]


class EvalConfig(pydantic.BaseModel):
    dataset: str
    batch_size: int
    metrics: list[str]
    data_loader: DataloaderConfig


def validation_loop(model, dataloader, metrics):
    for batch in tqdm(dataloader):
        images, targets = batch
        outputs = model(images)
        for metric in metrics:
            metric.update(outputs, targets)

    results = {metric: metric.compute() for metric in metrics}
    return results


def evaluate(model, config: EvalConfig):
    if config.dataset == "COCO":
        dataset = get_coco("data", validation=True, train=False)
    else:
        assert False, f"Unknown dataset: {config.dataset}"

    dataloader = get_dataloader(dataset, config.data_loader)

    metrics = [
        MulticlassAccuracy(num_classes=64),
        MulticlassConfusionMatrix(num_classes=64),
    ]

    results = validation_loop(model, dataloader, metrics)
    print(results)


if __name__ == "__main__":
    import torchvision.models as models

    model = MobileNet64()

    config = EvalConfig(
        dataset="COCO",
        batch_size=4,
        metrics=["MulticlassAccuracy", "MulticlassConfusionMatrix"],
        data_loader=DataloaderConfig(batch_size=4, shuffle=True),
    )

    evaluate(model, config)
