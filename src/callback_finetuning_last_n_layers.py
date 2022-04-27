from typing import Iterable, List, Optional, Union

from pytorch_lightning.callbacks import BackboneFinetuning
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
    """

    def __init__(self, unfreeze_blocks_num: Union[int, str], *args, **kwargs):
        self.unfreeze_blocks_num = unfreeze_blocks_num
        super(BackboneFinetuningLastLayers, self).__init__(*args, **kwargs)

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


if __name__ == "__main__":
    pass
