from __future__ import annotations, division, print_function

from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from torch import nn
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.efficientnet import model_urls as efficientnet_model_urls
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import model_urls as resnet_model_urls
from data_module_geoguesser import GeoguesserDataModule
from sklearn.metrics.pairwise import haversine_distances
from utils_model import Identity, model_remove_fc
from utils_env import DEFAULT_EARLY_STOPPING_EPOCH_FREQ
from utils_train import multi_acc
import numpy as np

allowed_models = list(resnet_model_urls.keys()) + list(efficientnet_model_urls.keys())

hyperparameter_metrics = [
    "train_loss_epoch",
    "train_acc_epoch",
    "val_loss_epoch",
    "val_acc_epoch",
    "test_loss_epoch",
    "test_acc_epoch",
]


class OnTrainEpochStartLogCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module: LitModel):
        data_dict = {
            "trainable_params_num": pl_module.get_num_of_trainable_params(),
        }
        pl_module.log_dict(data_dict)


class LitModel(pl.LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
    1. Model and computations (init).
    2. Train Loop (training_step)
    3. Validation Loop (validation_step)
    4. Test Loop (test_step)
    5. Prediction Loop (predict_step)
    6. Optimizers and LR Schedulers (configure_optimizers)

    LightningModule can itself be used as a model object. This is because `forward` function is exposed and can be used. Then, instead of using self.backbone(x) we can write self(x). See: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#starter-example
    """

    loggers: List[TensorBoardLogger]

    def __init__(self, data_module: GeoguesserDataModule, num_classes: int, model_name, pretrained, learning_rate, weight_decay, batch_size, image_size, context_dict={}, **kwargs: Any):
        super().__init__()
        self.data_module = data_module
        self.df_csv = data_module.dataset.df_csv
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.image_size = image_size

        backbone = torch.hub.load("pytorch/vision:v0.12.0", model_name, pretrained=pretrained)
        self.backbone = model_remove_fc(backbone)
        self.fc = nn.Linear(self._get_last_fc_in_channels(), num_classes)

        self.save_hyperparameters()
        self.save_hyperparameters(
            {
                **context_dict,
                **{"model_name": self.backbone.__class__.__name__},
            }
        )

    def _get_last_fc_in_channels(self) -> Any:
        """
        Returns:
            number of input channels for the last fc layer (number of variables of the second dimension of the flatten layer). Fake image is created, passed through the backbone and flattened (while perseving batches).
        """
        num_channels = 3
        num_image_sides = 4
        with torch.no_grad():
            image_batch_list = [torch.rand(self.batch_size, num_channels, self.image_size, self.image_size)] * num_image_sides
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

    def on_train_start(self) -> None:
        if self.logger:
            for logger in self.loggers:
                zeros_dict = {metric: 0 for metric in hyperparameter_metrics}
                logger.log_hyperparams(self.hparams, zeros_dict)

    def training_step(self, batch, batch_idx):
        image_list, y, _, _ = batch
        y_pred = self(image_list)

        loss = F.cross_entropy(y_pred, y)
        acc = multi_acc(y_pred, y)

        data_dict = {
            "train_loss": loss.detach(),
            "train_acc": acc,
            "loss": loss,
            "h_distance": 0,
        }
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log_dict(data_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def training_epoch_end(self, outs):
        loss = sum(map(lambda x: x["train_loss"], outs)) / len(outs)
        acc = sum(map(lambda x: x["train_acc"], outs)) / len(outs)
        data_dict = {
            "train_loss_epoch": loss,
            "train_acc_epoch": acc,
            "trainable_params_num": self.get_num_of_trainable_params(),
        }
        self.log_dict(data_dict)
        pass

    def validation_step(self, batch, batch_idx):
        image_list, y_true, centroid_lat, centroid_lng = batch
        y_pred = self(image_list)

        # TODO: caculation for haversine distances stuff should be cached. How? Create a dict where keys are y_index and values are whatever we need. That might be precaculated np.stack([lat, lng], axis=1). Anything that will speed up the caculations
        y_true_idx = torch.argmax(y_true, dim=1).detach().numpy()
        y_pred_idx = torch.argmax(y_pred, dim=1).detach().numpy()

        row_true = self.df_csv.iloc[y_true_idx, :] #hash
        true_lat, true_lng = row_true["latitude"].to_numpy(), row_true["longitude"].to_numpy() #hash

        row_pred = self.df_csv.iloc[y_pred_idx, :]
        pred_lat, pred_lng = row_pred["latitude"].to_numpy(), row_pred["longitude"].to_numpy()

        haver_x = np.stack([true_lat, true_lng], axis=1) #hash
        haver_y = np.stack([pred_lat, pred_lng], axis=1)
        haver_dist = np.mean(haversine_distances(haver_x, haver_y))

        loss = F.cross_entropy(y_pred, y_true)
        acc = multi_acc(y_pred, y_true)
        data_dict = {
            "val_loss": loss.detach(),
            "val_acc": acc,
            "loss": loss,
            "haver_dist": haver_dist,
        }
        # self.log("val_loss", loss.detach(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log_dict(data_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def validation_epoch_end(self, outs):
        loss = sum(map(lambda x: x["val_loss"], outs)) / len(outs)
        acc = sum(map(lambda x: x["val_acc"], outs)) / len(outs)
        data_dict = {"val_loss_epoch": loss, "val_acc_epoch": acc}
        self.log_dict(data_dict)
        pass

    def test_step(self, batch, batch_idx):
        image_list, y, _, _ = batch
        y_pred = self(image_list)
        loss = F.cross_entropy(y_pred, y)
        acc = multi_acc(y_pred, y)
        data_dict = {
            "test_loss": loss.detach(),
            "test_acc": acc,
            "loss": loss,
        }
        self.log_dict(data_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def test_epoch_end(self, outs):
        loss = sum(map(lambda x: x["test_loss"], outs)) / len(outs)
        acc = sum(map(lambda x: x["test_acc"], outs)) / len(outs)
        data_dict = {"test_loss_epoch": loss, "test_acc_epoch": acc}
        self.log_dict(data_dict)
        pass

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if type(self.backbone) is EfficientNet:
            optimizer = torch.optim.RMSprop(
                self.backbone.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.2,
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=int(DEFAULT_EARLY_STOPPING_EPOCH_FREQ / 2) - 1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # The unit of the scheduler's step size, could also be 'step', 'epoch' updates the scheduler on epoch end whereas 'step', updates it after a optimizer update
                "interval": "epoch",
                "monitor": "val_loss_epoch",
                # How many epochs/steps should pass between calls to `scheduler.step()`.1 corresponds to updating the learning  rate after every epoch/step.
                # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the learning rate progress, this keyword can be used to specify a custom logged name
                "name": None,
            },
        }


if __name__ == "__main__":
    pass
