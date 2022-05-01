from __future__ import annotations, division, print_function

import random
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.efficientnet import model_urls as efficientnet_model_urls
from torchvision.models.resnet import model_urls as resnet_model_urls

from datamodule_geoguesser import GeoguesserDataModule
from defaults import DEFAULT_EARLY_STOPPING_EPOCH_FREQ, DEFAULT_TORCHVISION_VERSION
from utils_geo import crs_coords_to_degree, haversine_from_degs
from utils_model import crs_coords_weighed_mean, model_remove_fc
from utils_train import SchedulerType, multi_acc

allowed_models = list(resnet_model_urls.keys()) + list(efficientnet_model_urls.keys())


def get_haversine_from_predictions(
    crs_scaler: MinMaxScaler, pred_crs_coord: torch.Tensor, image_true_crs_coords: torch.Tensor
):
    pred_crs_coord = pred_crs_coord.cpu()
    image_true_crs_coords = image_true_crs_coords.cpu()

    pred_crs_coord_transformed = crs_scaler.inverse_transform(pred_crs_coord)
    true_crs_coord_transformed = crs_scaler.inverse_transform(image_true_crs_coords)

    pred_degree_coords = crs_coords_to_degree(pred_crs_coord_transformed)
    true_degree_coords = crs_coords_to_degree(true_crs_coord_transformed)
    return haversine_from_degs(pred_degree_coords, true_degree_coords)


class LitModelClassification(pl.LightningModule):
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

    def __init__(
        self,
        datamodule: GeoguesserDataModule,
        num_classes: int,
        model_name: str,
        pretrained: bool,
        learning_rate: torch.Tensor,
        weight_decay: float,
        batch_size: int,
        image_size: int,
        scheduler_type: str,
        epochs: int,
    ):
        super().__init__()
        self.register_buffer(
            "class_to_crs_centroid_map", datamodule.class_to_crs_centroid_map.clone().detach()
        )  # set self.class_to_crs_centroid_map on gpu

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.datamodule = datamodule
        self.scheduler_type = scheduler_type
        self.epochs = epochs

        backbone = torch.hub.load(DEFAULT_TORCHVISION_VERSION, model_name, pretrained=pretrained)
        self.backbone = model_remove_fc(backbone)
        self.fc = nn.Linear(self._get_last_fc_in_channels(), num_classes)

        self._set_example_input_array()
        self.save_hyperparameters(ignore=["datamodule"])

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

    def validation_step(self, batch, batch_idx):
        image_list, y_true, image_true_crs_coords = batch
        y_pred = self(image_list)

        pred_crs_coord = crs_coords_weighed_mean(y_pred, self.class_to_crs_centroid_map, top_k=5)
        haver_dist = get_haversine_from_predictions(self.datamodule.crs_scaler, pred_crs_coord, image_true_crs_coords)

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

    def test_step(self, batch, batch_idx):
        image_list, y_true, image_true_crs_coords = batch
        y_pred = self(image_list)

        pred_crs_coord = crs_coords_weighed_mean(y_pred, self.class_to_crs_centroid_map, top_k=5)
        haver_dist = get_haversine_from_predictions(self.datamodule.crs_scaler, pred_crs_coord, image_true_crs_coords)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.learning_rate), weight_decay=self.weight_decay
        )
        if self.scheduler_type is SchedulerType.AUTO_LR.value:
            """SchedulerType.AUTO_LR sets it's own scheduler. Only the optimizer has to be returned"""
            return optimizer

        config_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "monitor": "val/loss",
                # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
                # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
                "name": self.scheduler_type,
            },
        }
        if self.scheduler_type == SchedulerType.ONECYCLE.value:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.01, steps_per_epoch=len(self.datamodule.train_dataloader()), epochs=self.epochs
            )
            interval = "step"
        else:  # SchedulerType.PLATEAU
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=int(DEFAULT_EARLY_STOPPING_EPOCH_FREQ // 2) - 1
            )
            interval = "epoch"

        config_dict["lr_scheduler"].update({"scheduler": scheduler, "interval": interval})
        return config_dict


class LitModelRegression(pl.LightningModule):
    loggers: List[TensorBoardLogger]
    num_of_outputs = 2

    def __init__(
        self,
        datamodule: GeoguesserDataModule,
        num_classes: int,
        model_name: str,
        pretrained: bool,
        learning_rate: torch.Tensor,
        weight_decay: float,
        batch_size: int,
        image_size: int,
        scheduler_type: str,
        epochs: int,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.image_size = image_size
        self.datamodule = datamodule
        self.scheduler_type = scheduler_type
        self.epochs = epochs

        backbone = torch.hub.load(DEFAULT_TORCHVISION_VERSION, model_name, pretrained=pretrained)
        self.backbone = model_remove_fc(backbone)
        self.fc = nn.Linear(self._get_last_fc_in_channels(), LitModelRegression.num_of_outputs)

        self._set_example_input_array()
        self.save_hyperparameters(ignore=["datamodule"])

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

    def forward(self, image_list) -> Any:
        outs_backbone = [self.backbone(image) for image in image_list]
        out_backbone_cat = torch.cat(outs_backbone, dim=1)
        out_flatten = torch.flatten(out_backbone_cat, 1)
        out = self.fc(out_flatten)
        return out

    def training_step(self, batch, batch_idx):
        image_list, _, image_true_crs_coords = batch
        y_pred = self(image_list)

        loss = F.mse_loss(y_pred, image_true_crs_coords)

        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "train/loss": loss,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def validation_step(self, batch, batch_idx):
        image_list, _, image_true_crs_coords = batch
        pred_crs_coord = self(image_list)

        haver_dist = get_haversine_from_predictions(self.datamodule.crs_scaler, pred_crs_coord, image_true_crs_coords)

        loss = F.mse_loss(pred_crs_coord, image_true_crs_coords)
        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "val/loss": loss,
            "val/haversine_distance": haver_dist,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def test_step(self, batch, batch_idx):
        image_list, _, image_true_crs_coords = batch
        pred_crs_coord = self(image_list)

        haver_dist = get_haversine_from_predictions(self.datamodule.crs_scaler, pred_crs_coord, image_true_crs_coords)

        loss = F.mse_loss(pred_crs_coord, image_true_crs_coords)
        data_dict = {
            "loss": loss,  # the 'loss' key needs to be present
            "test/loss": loss,
            "test/haversine_distance": haver_dist,
        }
        log_dict = data_dict.copy()
        log_dict.pop("loss", None)
        self.log_dict(log_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.learning_rate), weight_decay=self.weight_decay
        )

        if self.scheduler_type is SchedulerType.AUTO_LR.value:
            """SchedulerType.AUTO_LR sets it's own scheduler. Only the optimizer has to be returned"""
            return optimizer

        config_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "monitor": "val/loss",
                # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
                # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
                "name": self.scheduler_type,
            },
        }
        if self.scheduler_type == SchedulerType.ONECYCLE.value:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.01, steps_per_epoch=len(self.datamodule.train_dataloader()), epochs=self.epochs
            )
            interval = "step"
        else:  # SchedulerType.PLATEAU
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=int(DEFAULT_EARLY_STOPPING_EPOCH_FREQ // 2) - 1
            )
            interval = "epoch"

        config_dict["lr_scheduler"].update({"scheduler": scheduler, "interval": interval})
        return config_dict


class LitSingleModel(LitModelClassification):
    def __init__(self, *args: Any, **kwargs: Any):
        print("LitSingleModel init")
        super().__init__(*args, **kwargs)
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
