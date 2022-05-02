from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning, Callback, BaseFinetuning
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from pabloppp_optim.delayer_scheduler import DelayerScheduler

from utils_model import get_last_layer, get_model_blocks
import logging
from typing import Any, Callable, Iterable, List, Optional, Union

import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from utils_train import get_trainer_steps_in_epoch


class InvalidArgument(Exception):
    pass


min_value = float(0)
max_value = float(1e9)
hyperparameter_metrics_init = {
    "train/loss_epoch": max_value,
    "train/acc_epoch": min_value,
    "val/loss_epoch": max_value,
    "val/acc_epoch": min_value,
    "val/haversine_distance_epoch": max_value,
    "test/loss_epoch": max_value,
    "test/acc_epoch": min_value,
    "test/haversine_distance_epoch": max_value,
}


class ConfigureOptimizers(pl.Callback):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_train_epoch_end(trainer, pl_module)


class LogMetricsAsHyperparams(pl.Callback):
    """
    Registeres metrics in the "hparams" tab in TensorBoard
    This isn't done automatically by tensorboard so it has to be done manually.
    For this callback to work, default_hp_metric has to be set to false when creating TensorBoardLogger
    """

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.loggers:
            for logger in pl_module.loggers:
                logger.log_hyperparams(pl_module.hparams, hyperparameter_metrics_init)  # type: ignore


class OnTrainEpochStartLogCallback(pl.Callback):
    """Logs metrics. pl_module has to implement get_num_of_trainable_params function"""

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        data_dict = {
            "current_lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
            "epoch": float(trainer.current_epoch),
            "trainable_params_num": float(pl_module.get_num_of_trainable_params()),  # type: ignore
        }
        pl_module.log_dict(data_dict, on_step=True, on_epoch=True, logger=True)
        pl_module.save_hyperparameters(data_dict)

    # def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     data_dict = {
    #         "current_lr/epoch": float(trainer.optimizers[0].param_groups[0]["lr"]),
    #         "epoch/epoch": float(trainer.current_epoch),
    #         "trainable_params_num/epoch": float(pl_module.get_num_of_trainable_params()),  # type: ignore
    #     }
    #     pl_module.log_dict(data_dict)
    #     pl_module.log("step", trainer.current_epoch)

    def on_train_start(self, trainer, pl_module: pl.LightningModule):
        log_dict_fix = {
            "val/loss_epoch": max_value,
            "val/haversine_distance_epoch": max_value,
            "train/loss": max_value,
            "train/acc": min_value,
            "val/loss": max_value,
            "val/acc": min_value,
            "val/haversine_distance": max_value,
            "test/loss": max_value,
            "test/acc": min_value,
            "test/haversine_distance": max_value,
        }
        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        data_dict = {
            "trainable_params_num": float(pl_module.get_num_of_trainable_params()),  # type: ignore
            "current_lr": float(current_lr),
            "epoch": float(trainer.current_epoch),
            **log_dict_fix,
        }
        pl_module.save_hyperparameters(data_dict)
        pl_module.log_dict(data_dict, logger=False)


class OverrideEpochMetricCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_training_epoch_end(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_training_epoch_start(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)


# class BackboneFinetuning2(BaseFinetuning):
#     r"""Finetune a backbone model based on a learning rate user-defined scheduling.

#     When the backbone learning rate reaches the current model learning rate
#     and ``should_align`` is set to True, it will align with it for the rest of the training.

#     Args:
#         unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.
#         lambda_func: Scheduling function for increasing backbone learning rate.
#         backbone_initial_ratio_lr:
#             Used to scale down the backbone learning rate compared to rest of model
#         backbone_initial_lr: Optional, Initial learning rate for the backbone.
#             By default, we will use ``current_learning /  backbone_initial_ratio_lr``
#         should_align: Whether to align with current learning rate when backbone learning
#             reaches it.
#         initial_denom_lr: When unfreezing the backbone, the initial learning rate will
#             ``current_learning_rate /  initial_denom_lr``.
#         train_bn: Whether to make Batch Normalization trainable.
#         verbose: Display current learning rate for model and backbone
#         rounding: Precision for displaying learning rate

#     Example::

#         >>> from pytorch_lightning import Trainer
#         >>> from pytorch_lightning.callbacks import BackboneFinetuning
#         >>> multiplicative = lambda epoch: 1.5
#         >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
#         >>> trainer = Trainer(callbacks=[backbone_finetuning])

#     """

#     def __init__(
#         self,
#         unfreeze_blocks_num: Union[int, str],
#         # train_dataloader_size: int,
#         unfreeze_backbone_at_epoch: int = 10,
#         backbone_initial_lr: Optional[float] = 1e-3,
#         should_align: bool = True,
#         train_bn: bool = True,
#         verbose: bool = False,
#         rounding: int = 12,
#     ) -> None:
#         super().__init__()

#         self.unfreeze_blocks_num = unfreeze_blocks_num
#         self.unfreeze_backbone_at_epoch: int = unfreeze_backbone_at_epoch
#         self.backbone_initial_lr: Optional[float] = backbone_initial_lr
#         self.should_align: bool = should_align
#         self.train_bn: bool = train_bn
#         self.verbose: bool = verbose
#         self.rounding: int = rounding
#         self.previous_backbone_lr: Optional[float] = None
#         # self.train_dataloader_size = train_dataloader_size

#     def unfreeze_and_add_param_group(
#         self,
#         modules: Union[Module, Iterable[Union[Module, Iterable]]],
#         optimizer: Optimizer,
#         lr: Optional[float] = None,
#         initial_denom_lr: float = 10,
#         train_bn: bool = True,
#     ) -> None:
#         # TODO: DONT DO THIS PLS add support for multiple modules, current version suports only one module
#         blocks = get_model_blocks(modules)

#         trainable_blocks: List[Module] = blocks
#         if type(self.unfreeze_blocks_num) is int:
#             trainable_blocks = blocks[len(blocks) - self.unfreeze_blocks_num :]
#         elif self.unfreeze_blocks_num != "all":
#             raise InvalidArgument("unfreeze_blocks_num argument should be [0, inf> or 'all'")

#         last_layer = get_last_layer(modules)
#         if last_layer:
#             trainable_blocks.append(last_layer)

#         return super().unfreeze_and_add_param_group(trainable_blocks, optimizer, lr, initial_denom_lr, train_bn)

#     def state_dict(self) -> Dict[str, Any]:
#         return {
#             "internal_optimizer_metadata": self._internal_optimizer_metadata,
#             "previous_backbone_lr": self.previous_backbone_lr,
#         }

#     def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
#         self.previous_backbone_lr = state_dict["previous_backbone_lr"]
#         super().load_state_dict(state_dict)

#     def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         """
#         Raises:
#             MisconfigurationException:
#                 If LightningModule has no nn.Module `backbone` attribute.
#         """
#         if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module):
#             self.trainer = trainer
#             return super().on_fit_start(trainer, pl_module)
#         raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

#     def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
#         self.freeze(pl_module.backbone)

#     def finetune_function(
#         self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
#     ) -> None:
#         """Called when the epoch begins."""
#         # TODO: ako ovo ne radi samo zamrzni a handleaj nekako optimizere unutar modula?
#         self.previous_backbone_lr = optimizer.param_groups[0]["lr"]
#         if epoch == 0:
#             train_dataloader_size = get_trainer_steps_in_epoch(self.trainer)
#             for i in range(len(self.trainer.lr_scheduler_configs)):
#                 existing_scheduler = self.trainer.lr_scheduler_configs[i].scheduler
#                 self.trainer.lr_scheduler_configs[i].scheduler = DelayerScheduler(
#                     self.unfreeze_backbone_at_epoch,
#                     existing_scheduler,
#                     existing_scheduler.optimizer,
#                     train_dataloader_size=train_dataloader_size,
#                 )
#         if epoch == self.unfreeze_backbone_at_epoch:
#             self.unfreeze_and_add_param_group(
#                 modules=pl_module.backbone,
#                 optimizer=optimizer,
#                 lr=self.backbone_initial_lr,
#                 train_bn=self.train_bn,
#             )


class BackboneFreezing(Callback):
    def __init__(self, unfreeze_blocks_num: Union[int, str], unfreeze_backbone_at_epoch: int):
        self.unfreeze_blocks_num = unfreeze_blocks_num
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        super().__init__()

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        BackboneFinetuning.freeze(pl_module.backbone)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        from pytorch_lightning.loops.utilities import _get_active_optimizers

        for opt_idx, optimizer in _get_active_optimizers(trainer.optimizers, trainer.optimizer_frequencies):
            self.unfreeze_if_needed(pl_module, trainer.current_epoch, optimizer, opt_idx)

    def unfreeze_if_needed(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        if epoch == self.unfreeze_backbone_at_epoch:
            self.unfreeze_and_add_param_group(pl_module.backbone, optimizer)  # type: ignore

    def unfreeze_and_add_param_group(
        self,
        modules: Module,
        optimizer: Optimizer,
        train_bn: bool = True,
    ) -> None:

        blocks = get_model_blocks(modules)
        trainable_blocks: List[Module] = blocks
        if type(self.unfreeze_blocks_num) is int:
            trainable_blocks = blocks[len(blocks) - self.unfreeze_blocks_num :]
        elif self.unfreeze_blocks_num != "all":
            raise InvalidArgument("unfreeze_blocks_num argument should be [0, inf> or 'all'")

        last_layer = get_last_layer(modules)
        if last_layer:
            trainable_blocks.append(last_layer)

        BaseFinetuning.make_trainable(trainable_blocks)
        params = BaseFinetuning.filter_params(trainable_blocks, train_bn=train_bn, requires_grad=True)
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        # if params:
        #     optimizer.add_param_group({"params": params})


# class BackboneFinetuningLastLayers(BackboneFinetuning):
#     """
#     BackboneFinetuning (base class) callback performs finetuning procedure where only the last layer is trainable. After unfreeze_backbone_at_epoch number of epochs it makes the whole model trainable.
#     BackboneFinetuningLastLayers does the same thing but it makes only last N blocks trainable instead of the whole model (all blocks).
#     https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.BackboneFinetuning.html

#     Args:
#         unfreeze_blocks_num - number of blocks that will be trainable after the finetuning procedure. Argument 'all' will reduce this class to BackboneFinetuning because the whole model will become trainable.
#     """

#     def __init__(self, unfreeze_blocks_num: Union[int, str], *args, **kwargs):
#         self.unfreeze_blocks_num = unfreeze_blocks_num
#         super().__init__(*args, **kwargs)

#     def unfreeze_and_add_param_group(
#         self,
#         modules: Union[Module, Iterable[Union[Module, Iterable]]],
#         optimizer: Optimizer,
#         lr: Optional[float] = None,
#         initial_denom_lr: float = 10,
#         train_bn: bool = True,
#     ) -> None:
#         # TODO: DONT DO THIS PLS add support for multiple modules, current version suports only one module
#         blocks = get_model_blocks(modules)

#         trainable_blocks: List[Module] = blocks
#         if type(self.unfreeze_blocks_num) is int:
#             trainable_blocks = blocks[len(blocks) - self.unfreeze_blocks_num :]
#         elif self.unfreeze_blocks_num != "all":
#             raise InvalidArgument("unfreeze_blocks_num argument should be [0, inf> or 'all'")

#         last_layer = get_last_layer(modules)
#         if last_layer:
#             trainable_blocks.append(last_layer)

#         return super().unfreeze_and_add_param_group(trainable_blocks, optimizer, lr, initial_denom_lr, train_bn)
