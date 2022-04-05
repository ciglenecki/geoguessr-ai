from __future__ import annotations, division, print_function

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import LoggerCollection
from torch import nn
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.efficientnet import model_urls as efficientnet_model_urls
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import model_urls as resnet_model_urls

from utils_model import Identity
from utils_env import DEFAULT_EARLY_STOPPING_EPOCH_FREQ
from utils_train import multi_acc

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

    logger: LoggerCollection

    def __init__(self, num_classes: int, model_name, pretrained, learning_rate, leave_last_n, weight_decay, batch_size, image_size, context_dict={}, **kwargs: Any):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.backbone = torch.hub.load("pytorch/vision:v0.12.0", model_name, pretrained=pretrained)
        self.batch_size = batch_size
        self.image_size = image_size

        self.fc = nn.Linear(self.get_pre_fc_in_channels(), num_classes)

        if type(self.backbone) is ResNet:
            self.backbone.fc = Identity()
            print(self.backbone.fc)
            # self.backbone.modules = self.backbone.modules[:-1]
            # original_in_features = self.backbone.fc.in_features
            # self.backbone.fc = nn.Linear(original_in_features, num_classes)

        if type(self.backbone) is EfficientNet:
            # self.last_layer = list(self.backbone.children())
            # self.last_linear = self.last_layer[-1]
            last_layer = list(self.backbone.classifier)
            last_layer_without_linear = last_layer[:-1]  # dropout 0.4
            last_linear: nn.Linear = last_layer[-1]  # type: ignore

            self.backbone.classifier = nn.Sequential(
                *last_layer_without_linear,
                nn.Linear(last_linear.in_features, num_classes),
            )
        self.save_hyperparameters()

        self.save_hyperparameters(
            {
                **context_dict,
                **{"model_name": self.backbone.__class__.__name__},
            }
        )

    def get_pre_fc_in_channels(self, *args, **kwargs) -> Any:
        """
        Forward is called whenever we type self(image) or similar
        """
        # ([8, 3, 28, 28])
        images = [torch.rand(self.batch_size, 3,  self.image_size, self.image_size)] * 4
        image_list_of_tensors = images
        output0 = self.backbone(image_list_of_tensors[0])
        output1 = self.backbone(image_list_of_tensors[1])
        output2 = self.backbone(image_list_of_tensors[2])
        output3 = self.backbone(image_list_of_tensors[3])
        concatenated_output = torch.cat([output0, output1, output2, output3])
        flattened_output = torch.flatten(concatenated_output)
        return flattened_output.shape[0]

    def forward(self, images, *args, **kwargs) -> Any:
        """
        Forward is called whenever we type self(image) or similar
        """
        image_list_of_tensors = images
        output0 = self.backbone(image_list_of_tensors[0])
        output1 = self.backbone(image_list_of_tensors[1])
        output2 = self.backbone(image_list_of_tensors[2])
        output3 = self.backbone(image_list_of_tensors[3])
        concatenated_output = torch.cat([output0, output1, output2, output3])
        flattened_output = torch.flatten(concatenated_output)
        self.fc(flattened_output)

    def get_num_of_trainable_params(self):
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def on_train_start(self) -> None:
        if self.logger:
            for logger in self.loggers:
                zeros_dict = {metric: 0 for metric in hyperparameter_metrics}
                logger.log_hyperparams(self.hparams, zeros_dict)  # TODO: make sure to

    def training_step(self, batch, batch_idx):
        images, y = batch
        y_pred = self(images)
        loss = F.cross_entropy(y_pred, y)
        acc = multi_acc(y_pred, y)
        data_dict = {
            "train_loss": loss.detach(),
            "train_acc": acc,
            "loss": loss,
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
        images, y = batch
        y_pred = self(images)
        loss = F.cross_entropy(y_pred, y)
        acc = multi_acc(y_pred, y)
        data_dict = {
            "val_loss": loss.detach(),
            "val_acc": acc,
            "loss": loss,
        }
        self.log_dict(data_dict, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return data_dict

    def validation_epoch_end(self, outs):
        loss = sum(map(lambda x: x["val_loss"], outs)) / len(outs)
        acc = sum(map(lambda x: x["val_acc"], outs)) / len(outs)
        data_dict = {"val_loss_epoch": loss, "val_acc_epoch": acc}
        self.log_dict(data_dict)
        pass

    def test_step(self, batch, batch_idx):
        images, y = batch
        y_pred = self(images)
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
        print(outs)
        loss = sum(map(lambda x: x["test_loss"], outs)) / len(outs)
        acc = sum(map(lambda x: x["test_acc"], outs)) / len(outs)
        data_dict = {"test_loss_epoch": loss, "test_acc_epoch": acc}
        self.log_dict(data_dict)
        pass

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if type(self.backbone) is EfficientNet:
            optimizer = torch.optim.RMSprop(
                self.backbone.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.2,
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",
                                                               patience=int(DEFAULT_EARLY_STOPPING_EPOCH_FREQ / 2) - 1)
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
