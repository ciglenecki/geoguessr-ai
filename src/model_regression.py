from __future__ import annotations, division, print_function

import random
from time import time
from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from defaults import DEFAULT_EARLY_STOPPING_EPOCH_FREQ, DEFAULT_TORCHVISION_VERSION
from utils_geo import crs_coords_to_degree
from utils_model import crs_coords_weighed_mean, model_remove_fc
from utils_train import OptimizerType, SchedulerType


class LitModelRegression(pl.LightningModule):
    loggers: List[TensorBoardLogger]
    num_of_outputs = 2

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        pretrained: bool,
        learning_rate: float,
        lr_finetune: float,
        weight_decay: float,
        batch_size: int,
        image_size: int,
        scheduler_type: str,
        epochs: int,
        class_to_crs_centroid_map: torch.Tensor,
        crs_scaler: MinMaxScaler,
        train_dataloader_size: int,
        optimizer_type: str,
        unfreeze_at_epoch: int,
    ):
        super().__init__()

        self.learning_rate = torch.tensor(learning_rate).float()
        self.lr_finetune = lr_finetune
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.crs_scaler: MinMaxScaler = crs_scaler  # type: ignore
        self.scheduler_type = scheduler_type
        self.epochs = epochs
        self.train_dataloader_size = train_dataloader_size
        self.class_to_crs_centroid_map = class_to_crs_centroid_map
        self.optimizer_type = optimizer_type
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.epoch = 0

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

        haver_dist = get_haversine_from_predictions(self.crs_scaler, pred_crs_coord, image_true_crs_coords)

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

        haver_dist = get_haversine_from_predictions(self.crs_scaler, pred_crs_coord, image_true_crs_coords)

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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        image_list, uuid = batch
        y_pred = self(image_list)

        pred_crs_coord = crs_coords_weighed_mean(y_pred, self.class_to_crs_centroid_map, top_k=5)
        pred_crs_coord = pred_crs_coord.cpu()
        pred_crs_coord_transformed = self.crs_scaler.inverse_transform(pred_crs_coord)
        pred_degree_coords = crs_coords_to_degree(pred_crs_coord_transformed)

        data_dict = {"latitude": pred_degree_coords[:, 0], "longitude": pred_degree_coords[:, 1], "uuid": uuid}
        return data_dict

    def configure_optimizers(self):
        if not self.trainer:
            return

        if self.optimizer_type == OptimizerType.ADAMW.value:
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.learning_rate), weight_decay=5e-3)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=float(self.learning_rate), weight_decay=self.weight_decay
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
            best_onecycle_min_lr = 0.00025
            best_onecycle_initial_lr = 0.132
            scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
                optimizer,
                max_lr=best_onecycle_initial_lr,  # TOOD:self.learning_rate,
                final_div_factor=best_onecycle_initial_lr / best_onecycle_min_lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            interval = "step"
            reduce_on_plateau = False
        else:  # SchedulerType.PLATEAU
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.5, patience=int(DEFAULT_EARLY_STOPPING_EPOCH_FREQ // 2) - 1
            )
            interval = "epoch"
            reduce_on_plateau = True

        config_dict["lr_scheduler"].update(
            {
                "scheduler": scheduler,
                "interval": interval,
                "reduce_on_plateau": reduce_on_plateau,
            }
        )
        return config_dict

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        print("self.current_epoch < self.unfreeze_at_epoc", self.current_epoch, self.unfreeze_at_epoch)
        if self.current_epoch < self.unfreeze_at_epoch:
            return
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx: int,
        optimizer_closure,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:

        num_of_steps_to_skip = int(self.train_dataloader_size * self.unfreeze_at_epoch)

        if self.trainer.global_step < num_of_steps_to_skip:
            # if self.trainer.current_epoch < self.unfreeze_at_epoch:
            upper_limit = min(self.lr_finetune, 1 - (float(self.trainer.global_step + 1) / num_of_steps_to_skip))
            lr_scale = max(1e-6, upper_limit)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale
        elif (
            self.trainer.current_epoch == self.unfreeze_at_epoch and self.scheduler_type == SchedulerType.PLATEAU.value
        ):
            for pg in optimizer.param_groups:
                pg["lr"] = self.learning_rate

        optimizer.step(closure=optimizer_closure)


if __name__ == "__main__":
    pass
