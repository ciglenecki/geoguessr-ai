from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import BackboneFinetuning, BaseFinetuning, Callback
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from utils_model import get_last_layer, get_model_blocks


class InvalidArgument(Exception):
    pass


class LogMetricsAsHyperparams(pl.Callback):
    """
    Registeres metrics in the "hparams" tab in TensorBoard
    This isn't done automatically by tensorboard so it has to be done manually.
    For this callback to work, default_hp_metric has to be set to false when creating TensorBoardLogger
    """

    def __init__(self) -> None:
        super().__init__()
        min_value = float(0)
        max_value = float(1e5)
        self.hyperparameter_metrics_init = {
            "train/loss_epoch": max_value,
            "train/acc_epoch": min_value,
            "val/loss_epoch": max_value,
            "val/acc_epoch": min_value,
            "val/haversine_distance_epoch": max_value,
            "test/loss_epoch": max_value,
            "test/acc_epoch": min_value,
            "test/haversine_distance_epoch": max_value,
            "epoch": float(0),
            "epoch_true": float(0),
        }

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        if pl_module.loggers:
            for logger in pl_module.loggers:  # type: ignore
                logger: pl_loggers.TensorBoardLogger
                logger.log_hyperparams(pl_module.hparams, self.hyperparameter_metrics_init)  # type: ignore


class OnTrainEpochStartLogCallback(pl.Callback):
    """Logs metrics. pl_module has to implement get_num_of_trainable_params function"""

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        data_dict = {
            "trainable_params_num/epoch": float(pl_module.get_num_of_trainable_params()),  # type: ignore
            "current_lr/epoch": trainer.optimizers[0].param_groups[0]["lr"],
            "epoch_true": trainer.current_epoch,
            "step": trainer.current_epoch,
        }
        pl_module.log_dict(data_dict)
        # # TODO: maybe, put this in val_start ? and start with --val check = 1 !!!remove this when loading from legacy
        # pl_module.log("val/loss_epoch", 100000)
        # pl_module.log("val/haversine_distance_epoch", 100000)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        data_dict = {
            "current_lr/step": trainer.optimizers[0].param_groups[0]["lr"],
        }
        pl_module.log_dict(data_dict)


class OverrideEpochMetricCallback(Callback):
    """Override the X axis in Tensorboard for all "epoch" events. X axis will be epoch index instead of step index"""

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
        pl_module.log("step", float(trainer.current_epoch))


class BackboneFreezing(Callback):
    def __init__(self, unfreeze_blocks_num: Union[int, str], unfreeze_at_epoch: int):
        self.unfreeze_blocks_num = unfreeze_blocks_num
        self.unfreeze_at_epoch = unfreeze_at_epoch
        super().__init__()

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        BackboneFinetuning.freeze(pl_module.backbone)

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException(
            "The LightningModule should have a nn.Module `backbone` attribute"
        )

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        from pytorch_lightning.loops.utilities import _get_active_optimizers

        for opt_idx, optimizer in _get_active_optimizers(
            trainer.optimizers, trainer.optimizer_frequencies
        ):
            self.unfreeze_if_needed(
                pl_module, trainer.current_epoch, optimizer, opt_idx
            )

    def unfreeze_if_needed(
        self,
        pl_module: "pl.LightningModule",
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        if epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(pl_module.backbone, optimizer)  # type: ignore

    def unfreeze_and_add_param_group(
        self,
        modules: Module,
        optimizer: Optimizer,
        train_bn: bool = True,
    ) -> None:

        blocks = get_model_blocks(modules)
        trainable_blocks: list[Module] = blocks
        if type(self.unfreeze_blocks_num) is int:
            trainable_blocks = blocks[len(blocks) - self.unfreeze_blocks_num :]
        elif self.unfreeze_blocks_num != "all":
            raise InvalidArgument(
                "unfreeze_blocks_num argument should be [0, inf> or 'all'"
            )

        last_layer = get_last_layer(modules)
        if last_layer:
            trainable_blocks.append(last_layer)

        BaseFinetuning.make_trainable(trainable_blocks)
        params = BaseFinetuning.filter_params(
            trainable_blocks, train_bn=train_bn, requires_grad=True
        )
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        if params:
            optimizer.add_param_group({"params": params})
