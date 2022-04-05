from __future__ import annotations, division, print_function

import torch.nn as nn
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.resnet import ResNet


def get_model_blocks(model):
    """
    returns list of model blocks without last layer (classifier/fc)
    """
    if type(model) is EfficientNet:
        return list(model.features)  # model.children())[-3]
    if type(model) is ResNet:
        return list(model.children())  # model.children())[-3]
    return []


def get_last_layer(model: nn.Module) -> nn.Module | None:
    """
    returns last layer (classifier/fc)
    """
    if type(model) is EfficientNet:
        return model.classifier  # model.children())[-3]
    if type(model) is ResNet:
        return model.fc  # model.children())[-3]


def freeze_but_last_n_blocks(model, leave_last_n):
    """
    Freeze all layers until last n layers
    """
    model_name = model.__class__.__name__
    model_blocks = get_model_blocks(model)

    for model_block in model_blocks[:-leave_last_n]:
        for param in model_block.parameters():
            param.requires_grad = False
    print("{} freezing ({}/{}) layers".format(model_name, len(model_blocks) - leave_last_n, len(model_blocks)))
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

if __name__ == "__main__":
    pass
