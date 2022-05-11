from typing import Iterable, List, Optional, Tuple, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning, BaseFinetuning
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from utils_model import get_last_layer, get_model_blocks


class InvalidArgument(Exception):
    pass


class BackboneFinetuningLastLayers(BackboneFinetuning):
    """
    BackboneFinetuning (base class) callback performs finetuning procedure where only the last layer is trainable. After unfreeze_backbone_at_epoch number of epochs it makes the whole model trainable.
    BackboneFinetuningLastLayers does the same thing but it makes only last N blocks trainable instead of the whole model (all blocks).
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.BackboneFinetuning.html

    Args:
        unfreeze_blocks_num - number of blocks that will be trainable after the finetuning procedure. Argument 'all' will reduce this class to BackboneFinetuning because the whole model will become trainable.

        unfreeze_at_epoch - at which epoch to unfreeze the rest of model
        lr_finetuning_range - two ranges that define the starting and the ending learning rate during the finetuning phase
        lr_after_finetune - sets learning rate to this value after finetuning is finished
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
        verbose: bool = True,
    ):
        self.unfreeze_blocks_num = unfreeze_blocks_num
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.lr_finetuning_min, self.lr_finetuning_max = lr_finetuning_range
        self.lr_total_diff = self.lr_finetuning_max - self.lr_finetuning_min
        self.lr_after_finetune = lr_after_finetune

        self.is_finetuning = True

        """ Useless arguments but we send them anyway as a sanity check"""
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

    def _return_trainable_modules(
        self,
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        unfreeze_blocks_num: Union[int, str],
    ):

        if type(unfreeze_blocks_num) is int:
            blocks = get_model_blocks(modules)
            return blocks[len(blocks) - unfreeze_blocks_num :]

        elif type(unfreeze_blocks_num) is str and "layer" in unfreeze_blocks_num:
            """Add all modules that continue after unfreeze_blocks_num module (e.g. layer3.2)"""

            found_layer = False
            modules_list = []

            for name, module in modules.named_modules():
                if found_layer:
                    if self.verbose:
                        print("Adding layer", name)
                    modules_list.append(module)
                if unfreeze_blocks_num in name:
                    found_layer = True

            if not found_layer:
                raise InvalidArgument(
                    "unfreeze_blocks_num {} should be a a named module (e.g. layer3.2)".format(unfreeze_blocks_num)
                )
            return modules_list

        elif unfreeze_blocks_num != "all":
            raise InvalidArgument("unfreeze_blocks_num argument should be [0, inf> or 'all'")

        return get_model_blocks(modules)

    def unfreeze_and_add_param_group(
        self,
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        lr: Optional[float] = None,
        train_bn: bool = True,
    ) -> None:
        # TODO: current version suports only one module

        trainable_blocks: List[Module] = self._return_trainable_modules(modules, self.unfreeze_blocks_num)

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
        if epoch >= self.unfreeze_backbone_at_epoch and self.is_finetuning:
            self.is_finetuning = False
            self.unfreeze_and_add_param_group(
                modules=pl_module.backbone,
                optimizer=optimizer,
                lr=self.lr_after_finetune,
                train_bn=self.train_bn,
            )
            if self.verbose:
                print("\nBackboneFinetuningLastLayers is setting the learning rate to:", self.lr_after_finetune)


if __name__ == "__main__":
    pass
