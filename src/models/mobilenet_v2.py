import torch
import pathlib
import torchvision
import lightning as L
import pydantic

class MobileNetV2Config(pydantic.BaseModel):
    pretrained_weights: str = None
    num_classes: int = 64
    lr: float = 1e-3


def get_model(pretrained_weights=None, num_classes=64):
    mobilenet_v2 = torchvision.models.mobilenet_v2(num_classes=num_classes)

    # Load pretrained weights from .pth file
    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights, weights_only=True)
        mobilenet_v2.load_state_dict(state_dict)

    return mobilenet_v2


class L_MobileNetV2(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = get_model(config.pretrained_weights, config.num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":
    config = MobileNetV2Config(pretrained_weights="./src/models/lpcv_mobilenet_v2.pth")
    model = L_MobileNetV2(config)
    print(model)