import pytorch_lightning as pl
import torchmetrics as tm
import torchvision
import pandas as pd
#import cv2
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.data_module import GeoguesserDataModule
from src.utils_env import DEFAULT_IMAGE_SIZE
from src.utils_paths import PATH_DATA_RAW

PATH = "../input/cassava-leaf-disease-classification/train_images/"
CLASSES = 24


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
        # image_size = 64
        super().__init__()
        self.dic = {4615: 0, 4616: 1, 4617: 2, 4513: 4, 4514: 5, 4515: 6, 4516: 7, 4517: 8, 4518: 9, 4519: 10, 4413: 11, 4414: 12, 4415: 13, 4416: 14, 4418: 15, 4419: 16, 4315: 17, 4316: 18, 4317: 19, 4215: 20, 4216: 21, 4217: 22, 4218: 23}
        self.cnv = nn.Conv2d(3, 128, kernel_size=5, stride=1)
        self.rel = nn.ReLU()
        self.bn = nn.BatchNorm2d(128)
        self.mxpool = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, CLASSES)
        self.softmax = nn.Softmax()
        self.accuracy = tm.Accuracy()

    def forward(self, x):
        out = self.bn(self.rel(self.cnv(x)))
        out = self.flat(self.mxpool(x))
        out = self.rel(self.fc1(out))
        out = self.rel(self.fc2(out))
        out = self.fc3(out)
        return out

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, CLASSES), target)

    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        image, latitude, longitude = batch
        label = torch.LongTensor([self.dic[int(str(int(lat)) + str(int(long)))] for lat, long in zip(latitude, longitude)])
        out = self(image)
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, latitude, longitude = batch
        images, latitude, longitude = batch
        label = torch.LongTensor([self.dic[int(str(int(lat)) + str(int(long)))] for lat, long in zip(latitude, longitude)])
        out = self(image)
        loss = self.loss_fn(out, label)
        out = nn.Softmax(-1)(out)
        logits = torch.argmax(out, dim=1)
        accu = self.accuracy(logits, label)
        self.log('valid_loss', loss)
        self.log('train_acc_step', accu)
        return loss, accu


image_transform = transforms.Compose(
    [
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
geoguesser_datamodule = GeoguesserDataModule(data_dir=PATH_DATA_RAW, image_transform=image_transform)

mod = Model()
trainer = pl.Trainer(max_epochs=6)
trainer.fit(model=mod, datamodule=geoguesser_datamodule)
