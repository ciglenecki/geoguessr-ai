from typing import Iterable, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning, Callback
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from utils_model import get_last_layer, get_model_blocks


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

    def on_train_start(self, trainer, pl_module: pl.LightningModule):
        log_dict_fix = {
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
            "step": trainer.current_epoch,
            **log_dict_fix,
        }
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

    def _log_step_as_current_epoch(self, trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)


class BackboneFinetuningLastLayers(BackboneFinetuning):
    """
    BackboneFinetuning (base class) callback performs finetuning procedure where only the last layer is trainable. After unfreeze_backbone_at_epoch number of epochs it makes the whole model trainable.
    BackboneFinetuningLastLayers does the same thing but it makes only last N blocks trainable instead of the whole model (all blocks).
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.BackboneFinetuning.html

    Args:
        unfreeze_blocks_num - number of blocks that will be trainable after the finetuning procedure. Argument 'all' will reduce this class to BackboneFinetuning because the whole model will become trainable.
    """

    def __init__(self, unfreeze_blocks_num: Union[int, str], *args, **kwargs):
        self.unfreeze_blocks_num = unfreeze_blocks_num
        super().__init__(*args, **kwargs)

    def unfreeze_and_add_param_group(
        self,
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        initial_denom_lr: float = 10,
        train_bn: bool = True,
    ) -> None:
        # TODO: DONT DO THIS PLS add support for multiple modules, current version suports only one module
        blocks = get_model_blocks(modules)

        trainable_blocks: List[Module] = blocks
        if type(self.unfreeze_blocks_num) is int:
            trainable_blocks = blocks[len(blocks) - self.unfreeze_blocks_num :]
        elif self.unfreeze_blocks_num != "all":
            raise InvalidArgument("unfreeze_blocks_num argument should be [0, inf> or 'all'")

        last_layer = get_last_layer(modules)
        if last_layer:
            trainable_blocks.append(last_layer)

        return super().unfreeze_and_add_param_group(trainable_blocks, optimizer, lr, initial_denom_lr, train_bn)
