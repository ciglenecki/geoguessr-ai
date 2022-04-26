from __future__ import annotations, division, print_function

import math
from typing import Any, List
import random

import geopandas
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pyproj import Transformer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics.pairwise import haversine_distances
from torch import nn
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.efficientnet import model_urls as efficientnet_model_urls
from torchvision.models.resnet import model_urls as resnet_model_urls

from data_module_geoguesser import GeoguesserDataModule
from defaults import (
    DEFAULT_EARLY_STOPPING_EPOCH_FREQ,
    DEFAULT_TORCHVISION_VERSION,
    DEFAULT_GLOBAL_CRS,
    DEFAULT_CROATIA_CRS,
)
from preprocess_sample_coords import reproject_dataframe
from utils_functions import timeit
from utils_model import lat_lng_weighted_mean, model_remove_fc
from utils_train import multi_acc

allowed_models = list(resnet_model_urls.keys()) + list(efficientnet_model_urls.keys())


class OnTrainEpochStartLogCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module: LitModel):
        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        data_dict = {
            "trainable_params_num": float(pl_module.get_num_of_trainable_params()),
            "current_lr": float(current_lr),
            "step": trainer.current_epoch,
        }
        pl_module.log_dict(data_dict)

    def on_train_start(self, trainer, pl_module: LitModel):
        self.on_train_epoch_start(trainer, pl_module)


class LitModel(pl.LightningModule):
    """
    A LightningModule organizes PyTorch code into 6 sections:
    1. Model and computations (init).
    2. Train Loop (training_step)
    3. Validation Loop (validation_step)
    4. Test Loop (test_step)
    5. Prediction Loop (predict_step)
    6. Optimizers and LR Schedulers (configure_optimizers)

    LightningModule can itself be used as a model object. This is because `forward` function is exposed and can be used. Then, instead of using self.backbone(x) we can write self(x). See: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#starter-example
    """

    loggers: List[TensorBoardLogger]

    # send class to centroid map instead of data module!, and cl,one it before
    def __init__(
        self,
        data_module: GeoguesserDataModule,
        num_classes: int,
        model_name,
        pretrained,
        learning_rate,
        weight_decay,
        batch_size,
        image_size,
    ):
        super(LitModel, self).__init__()
        self.register_buffer(
            "class_to_centroid_map", data_module.class_to_centroid_map.clone().detach()
        )  # set self.class_to_centroid_map on gpu

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.data_module = data_module

        backbone = torch.hub.load(DEFAULT_TORCHVISION_VERSION, model_name, pretrained=pretrained)
        self.backbone = model_remove_fc(backbone)
        self.fc = nn.Linear(self._get_last_fc_in_channels(), num_classes)

        self._set_example_input_array()
        self.save_hyperparameters(ignore=["data_module"])

    def _set_example_input_array(self):
        num_channels = 3
        num_of_image_sides = 4
        list_of_images = [
            torch.rand(self.batch_size, num_channels, self.image_size, self.image_size)
        ] * num_of_image_sides
        self.example_input_array = torch.stack(list_of_images)

    def _get_last_fc_in_channels(self) -> Any:
        """
        Returns:
            number of input channels for the last fc layer (number of variables of the second dimension of the flatten layer). Fake image is created, passed through the backbone and flattened (while perseving batches).
        """
        num_channels = 3
        num_image_sides = 4
        with torch.no_grad():
            image_batch_list = [
                torch.rand(self.batch_size, num_channels, self.image_size, self.image_size)
            ] * num_image_sides
            outs_backbone = [self.backbone(image) for image in image_batch_list]
            out_backbone_cat = torch.cat(outs_backbone, dim=1)
            flattened_output = torch.flatten(out_backbone_cat, 1)  # shape (batch_size x some_number)
        return flattened_output.shape[1]

    def get_num_of_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, image_list) -> Any:
        outs_backbone = [self.backbone(image) for image in image_list]
        out_backbone_cat = torch.cat(outs_backbone, dim=1)
        out_flatten = torch.flatten(out_backbone_cat, 1)
        out = self.fc(out_flatten)
        return out

    def training_step(self, batch, batch_idx):
        image_list, y, _ = batch
        y_pred = self(image_list)  # same as self.forward

        loss = F.cross_entropy(y_pred, y)
        acc = multi_acc(y_pred, y)

        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "train/loss": loss,
            "train/acc": acc,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def training_epoch_end(self, outs):
        loss = sum(map(lambda x: x["train/loss"], outs)) / len(outs)
        acc = sum(map(lambda x: x["train/acc"], outs)) / len(outs)
        log_dict = {
            "train/loss_epoch": loss,
            "train/acc_epoch": acc,
            "step": self.current_epoch,  # explicitly set the x axis
        }

        self.log_dict(log_dict)

    def validation_step(self, batch, batch_idx):
        image_list, y_true, image_true_coords = batch
        y_pred = self(image_list)
        coord_pred = lat_lng_weighted_mean(y_pred, self.class_to_centroid_map, top_k=5)
        coord_pred_changed, image_true_coords_changed = self.crs_to_lat_long(coord_pred, image_true_coords)
        haver_dist = np.mean(
            haversine_distances(torch.deg2rad(coord_pred_changed), torch.deg2rad(image_true_coords_changed))
        )

        loss = F.cross_entropy(y_pred, y_true)
        acc = multi_acc(y_pred, y_true)
        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "val/loss": loss,
            "val/acc": acc,
            "val/haversine_distance": haver_dist,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def validation_epoch_end(self, outs):
        loss = sum(map(lambda x: x["val/loss"], outs)) / len(outs)
        acc = sum(map(lambda x: x["val/acc"], outs)) / len(outs)
        log_dict = {
            "val/loss_epoch": loss,
            "val/acc_epoch": acc,
            "step": self.current_epoch,
        }
        self.log_dict(log_dict)

    def test_step(self, batch, batch_idx):
        image_list, y_true, image_true_coords = batch
        y_pred = self(image_list)
        coord_pred = lat_lng_weighted_mean(y_pred, self.class_to_centroid_map, top_k=5)
        coord_pred_changed, image_true_coords_changed = self.crs_to_lat_long(coord_pred, image_true_coords)
        haver_dist = np.mean(
            haversine_distances(torch.deg2rad(coord_pred_changed), torch.deg2rad(image_true_coords_changed))
        )

        loss = F.cross_entropy(y_pred, y_true)
        acc = multi_acc(y_pred, y_true)
        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "test/loss": loss,
            "test/acc": acc,
            "test/haversine_distance": haver_dist,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def test_epoch_end(self, outs):
        loss = sum(map(lambda x: x["test/loss"], outs)) / len(outs)
        acc = sum(map(lambda x: x["test/acc"], outs)) / len(outs)
        log_dict = {
            "test/loss_epoch": loss,
            "test/acc_epoch": acc,
            "step": self.current_epoch,
        }
        self.log_dict(log_dict)

    def cart_to_lat_long(self, y, images):

        y[:, 0] = y[:, 0] * self.data_module.x_max + self.data_module.x_min
        y[:, 1] = y[:, 1] * self.data_module.y_max + self.data_module.y_min
        y[:, 2] = y[:, 2] * self.data_module.z_max + self.data_module.z_min

        tmp_tensor = torch.zeros(y.size(0), y.size(1))
        tmp_tensor[:, :2] = y[:, [0, 1]] ** 2
        latitude_y = torch.atan2(y[:, 2], torch.sum(tmp_tensor[:, :2], dim=-1).sqrt())
        longitude_y = torch.atan2(y[:, 1], y[:, 0])

        images[:, 0] = images[:, 0] * self.data_module.x_max + self.data_module.x_min
        images[:, 1] = images[:, 1] * self.data_module.y_max + self.data_module.y_min
        images[:, 2] = images[:, 2] * self.data_module.z_max + self.data_module.z_min

        tmp_tensor = torch.zeros(images.size(0), images.size(1))

        tmp_tensor[:, :2] = images[:, [0, 1]] ** 2
        latitude_images = torch.atan2(images[:, 2], torch.sum(tmp_tensor[:, :2], dim=-1).sqrt())
        longitude_images = torch.atan2(images[:, 1], images[:, 0])

        return torch.stack((latitude_y, longitude_y)), torch.stack((latitude_images, longitude_images))

    def crs_to_lat_long(self, y, images):

        """
        Returns: torch tensor of lat and long in radians.
        """

        transformer = Transformer.from_crs("epsg:3766", "epsg:4326")

        # y[:, 1] = y[:, 1] * self.data_module.lat_max + self.data_module.lat_min
        # y[:, 0] = y[:, 0] * self.data_module.lng_max + self.data_module.lng_min

        images[:, 0] = images[:, 0] * self.data_module.lat_max + self.data_module.lat_min
        images[:, 1] = images[:, 1] * self.data_module.lng_max + self.data_module.lng_min
        images1, images2 = transformer.transform(images[:, 1].cpu(), images[:, 0].cpu())
        y1, y2 = transformer.transform(y[:, 1].cpu(), y[:, 0].cpu())

        return torch.tensor(np.dstack([y1, y2])[0]), torch.tensor(np.dstack([images1, images2])[0])

    def configure_optimizers(self):
        optimizer = (
            torch.optim.RMSprop(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.2,
            )
            if type(self.backbone) is EfficientNet
            else torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=int(DEFAULT_EARLY_STOPPING_EPOCH_FREQ // 2) - 1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # The unit of the scheduler's step size, could also be 'step', 'epoch' updates the scheduler on epoch end whereas 'step', updates it after a optimizer update
                "interval": "epoch",
                "monitor": "val/loss_epoch",
                # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
                # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
                "name": None,
            },
        }


class LitModelRegression(pl.LightningModule):
    loggers: List[TensorBoardLogger]
    num_of_outputs = 3

    def __init__(
        self,
        data_module,
        num_classes: int,
        model_name: str,
        pretrained: bool,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        image_size: int,
    ):
        super(LitModelRegression, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_module = data_module

        backbone = torch.hub.load(DEFAULT_TORCHVISION_VERSION, model_name, pretrained=pretrained)
        self.backbone = model_remove_fc(backbone)
        self.fc = nn.Linear(self._get_last_fc_in_channels(), LitModelRegression.num_of_outputs)

        self._set_example_input_array()
        self.save_hyperparameters()

    def _set_example_input_array(self):
        num_channels = 3
        num_of_image_sides = 4
        list_of_images = [
            torch.rand(self.batch_size, num_channels, self.image_size, self.image_size)
        ] * num_of_image_sides
        self.example_input_array = torch.stack(list_of_images)

    def _get_last_fc_in_channels(self) -> Any:
        """
        Returns: number of input channels for the last fc layer (number of variables of the second dimension of the
        flatten layer). Fake image is created, passed through the backbone and flattened (while preserving batches).
        """
        num_channels = 3
        num_image_sides = 4
        with torch.no_grad():
            image_batch_list = [
                torch.rand(self.batch_size, num_channels, self.image_size, self.image_size)
            ] * num_image_sides
            outs_backbone = [self.backbone(image) for image in image_batch_list]
            out_backbone_cat = torch.cat(outs_backbone, dim=1)
            flattened_output = torch.flatten(out_backbone_cat, 1)  # shape (batch_size x some_number)
        return flattened_output.shape[1]

    def get_num_of_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, image_list, *args, **kwargs) -> Any:
        outs_backbone = [self.backbone(image) for image in image_list]
        out_backbone_cat = torch.cat(outs_backbone, dim=1)
        out_flatten = torch.flatten(out_backbone_cat, 1)
        out = self.fc(out_flatten)
        return out

    def training_step(self, batch, batch_idx):
        image_list, _, image_true_coords = batch
        y_pred = self(image_list)

        loss = F.mse_loss(y_pred, image_true_coords)

        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "train/loss": loss,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def training_epoch_end(self, outs):
        loss = sum(map(lambda x: x["train/loss"], outs)) / len(outs)
        log_dict = {
            "train/loss_epoch": loss,
            "step": self.current_epoch,  # explicitly set the x axis
        }
        self.log_dict(log_dict)

    def validation_step(self, batch, batch_idx):
        image_list, _, image_true_coords = batch
        y_pred = self(image_list)

        y_pred_changed, image_true_coords_transformed = self.normalize_output(y_pred, image_true_coords)
        print(y_pred_changed, image_true_coords_transformed)

        haver_dist = np.mean(haversine_distances(y_pred_changed.cpu(), image_true_coords_transformed.cpu()))

        loss = F.mse_loss(y_pred, image_true_coords)
        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "val/loss": loss,
            "val/haversine_distance": haver_dist,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def validation_epoch_end(self, outs):
        loss = sum(map(lambda x: x["val/loss"], outs)) / len(outs)
        log_dict = {
            "val/loss_epoch": loss,
            "step": self.current_epoch,
        }
        self.log_dict(log_dict)

    def test_step(self, batch, batch_idx):
        image_list, _, image_true_coords = batch
        y_pred = self(image_list)

        y_pred_changed, image_true_coords_transformed = self.normalize_output(y_pred, image_true_coords)

        haver_dist = np.mean(haversine_distances(y_pred_changed.cpu(), image_true_coords_transformed.cpu()))

        loss = F.mse_loss(y_pred, image_true_coords)
        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "test/loss": loss,
            "test/haversine_distance": haver_dist,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def test_epoch_end(self, outs):
        loss = sum(map(lambda x: x["test/loss"], outs)) / len(outs)
        log_dict = {
            "test/loss_epoch": loss,
            "step": self.current_epoch,
        }
        self.log_dict(log_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if type(self.backbone) is EfficientNet:
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.2,
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=int(DEFAULT_EARLY_STOPPING_EPOCH_FREQ // 2) - 1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # The unit of the scheduler's step size, could also be 'step', 'epoch' updates the scheduler on epoch end whereas 'step', updates it after a optimizer update
                "interval": "epoch",
                "monitor": "val/loss_epoch",
                # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
                # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
                "name": None,
            },
        }

    def normalize_output(self, y, coords):
        y[:, 0] = y[:, 0] * self.data_module.lat_max_sin + self.data_module.lat_min_sin
        y[:, 1] = y[:, 1] * self.data_module.lng_max_sin + self.data_module.lng_min_sin

        coords[:, 0] = coords[:, 0] * self.data_module.lat_max_sin + self.data_module.lat_min_sin
        coords[:, 1] = coords[:, 1] * self.data_module.lng_max_sin + self.data_module.lng_min_sin

        return y, coords


class LitSingleModel(LitModel):
    def __init__(self, *args: Any, **kwargs: Any):
        print("LitSingleModel init")
        super(LitSingleModel, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self._get_last_fc_in_channels(), self.num_classes)

    def _get_last_fc_in_channels(self) -> Any:
        """
        Returns:
            number of input channels for the last fc layer (number of variables of the second dimension of the flatten layer). Fake image is created, passed through the backbone and flattened (while perseving batches).
        """
        num_channels = 3
        with torch.no_grad():
            image = torch.rand(self.batch_size, num_channels, self.image_size, self.image_size)
            out_backbone = self.backbone(image)
            flattened_output = torch.flatten(out_backbone, 1)  # shape (batch_size x some_number)
        return flattened_output.shape[1]

    def forward(self, image_list, *args, **kwargs) -> Any:
        image = random.choice(image_list)
        outs_backbone = self.backbone(image)
        out_flatten = torch.flatten(outs_backbone, 1)
        out = self.fc(out_flatten)
        return out


if __name__ == "__main__":
    pass
