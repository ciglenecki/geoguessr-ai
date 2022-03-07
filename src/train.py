import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from data_module import GeoguesserDataModule
from utils_env import DEFAULT_IMAGE_SIZE, DEFAULT_LR
from utils_paths import PATH_DATA_RAW


class Model(pl.LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
    1. Model and computations (init).
    2. Train Loop (training_step)
    3. Validation Loop (validation_step)
    4. Test Loop (test_step)
    5. Prediction Loop (predict_step)
    6. Optimizers and LR Schedulers (configure_optimizers)

    LightningModule can itself be used as a model object. This is because `forward` function is exposed and can be used. Then, instead of using self.model(x) we can write self(x). See: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#starter-example
    """

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet34(pretrained=True, progress=True)

    def training_step(self, batch, batch_idx):
        """
        TODO: work in progress
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training-loop
        """
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        TODO: work in progress
        """
        batch_images, batch_latitude, batch_longitude = batch
        y_hat = self.model(batch_images[0])
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=DEFAULT_LR)


image_transform = transforms.Compose(
    [
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
geoguesser_datamodule = GeoguesserDataModule(data_dir=PATH_DATA_RAW, image_transform=image_transform)

model = Model()
trainer = pl.Trainer()
trainer.fit(model, geoguesser_datamodule)
