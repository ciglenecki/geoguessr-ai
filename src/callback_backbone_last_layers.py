from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pytorch_lightning.callbacks import BackboneFinetuning, BaseFinetuning
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl
from utils_model import get_last_layer, get_model_blocks


class InvalidArgument(Exception):
    pass


class BackboneFinetuningLastLayers(BackboneFinetuning):
    r"""Finetune a backbone model based on a learning rate user-defined scheduling.

    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:
        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.
        lambda_func: Scheduling function for increasing backbone learning rate.
        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model
        backbone_initial_lr: Optional, Initial learning rate for the backbone.
            By default, we will use ``current_learning /  backbone_initial_ratio_lr``
        should_align: Whether to align with current learning rate when backbone learning
            reaches it.
        initial_denom_lr: When unfreezing the backbone, the initial learning rate will
            ``current_learning_rate /  initial_denom_lr``.
        train_bn: Whether to make Batch Normalization trainable.
        verbose: Display current learning rate for model and backbone
        rounding: Precision for displaying learning rate

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])

    """
    unfreeze_backbone_at_epoch: int
    backbone_initial_lr: float
    verbose: bool
    rounding: float
    initial_denom_lr: float
    backbone_initial_ratio_lr: float
    train_bn: bool

    def __init__(
        self,
        unfreeze_blocks_num: Union[int, str],
        unfreeze_at_epoch: int,
        lr_finetuning_range: Tuple[float, float],
        lr_after_finetune: float,
        train_dataloader_size: int,
        verbose: bool = True,
    ):
        print("BackboneFinetuningLastLayers")
        self.unfreeze_blocks_num = unfreeze_blocks_num
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.lr_finetuning_min, self.lr_finetuning_max = lr_finetuning_range
        self.lr_total_diff = self.lr_finetuning_max - self.lr_finetuning_min
        self.lr_after_finetune = lr_after_finetune

        """ Useless arguments but we send them anyway"""
        lambda_func = lambda x: x
        backbone_initial_ratio_lr = 1
        initial_denom_lr = 1
        backbone_initial_lr = lr_after_finetune

        super().__init__(
            unfreeze_backbone_at_epoch=unfreeze_at_epoch,
            lambda_func=lambda_func,
            backbone_initial_lr=backbone_initial_lr,
            backbone_initial_ratio_lr=backbone_initial_ratio_lr,
            initial_denom_lr=initial_denom_lr,
            verbose=verbose,
        )

    def unfreeze_and_add_param_group(
        self,
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        initial_denom_lr: float = 1,
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

        BaseFinetuning.make_trainable(trainable_blocks)
        params = BaseFinetuning.filter_params(trainable_blocks, train_bn=train_bn, requires_grad=True)
        params = BaseFinetuning.filter_on_optimizer(optimizer, params)
        if params:
            optimizer.add_param_group({"params": params, "lr": lr})

    def finetune_function(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        """Called when the epoch begins."""

        if epoch < self.unfreeze_backbone_at_epoch:
            trainer = pl_module.trainer
            num_of_steps_to_skip = int(trainer.num_training_batches * self.unfreeze_at_epoch)
            progress_percentage = float(trainer.global_step) / num_of_steps_to_skip
            lr_delta = self.lr_total_diff * progress_percentage
            new_lr = self.lr_finetuning_min + lr_delta
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

            self.previous_backbone_lr = new_lr
            if self.verbose:
                print("\nFinetuning LR:", new_lr)

        elif epoch == self.unfreeze_backbone_at_epoch:
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr_after_finetune

            self.previous_backbone_lr = self.lr_after_finetune

            self.unfreeze_and_add_param_group(
                modules=pl_module.backbone,
                optimizer=optimizer,
                lr=self.lr_after_finetune,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            if self.verbose:
                print("\nSetting learning rate back to:", self.lr_after_finetune)


if __name__ == "__main__":
    pass
