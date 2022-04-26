from __future__ import annotations, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.resnet import ResNet


def model_remove_fc(model: ResNet):
    """Replaces fully connected layer with identity function"""
    if type(model) is ResNet:
        model.fc = Identity()
    if type(model) is EfficientNet:
        model.classifier = Identity()
    return model


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
    if type(model) is ResNet:
        return model.fc  # model.children())[-3]
    if type(model) is EfficientNet:
        return model.classifier  # model.children())[-3]


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


def lat_lng_weighted_mean(y_pred, class_map, top_k):
    preds, indices = torch.topk(F.softmax(y_pred), k=top_k)
    preds = preds / torch.sum(preds, dim=1, keepdim=True)  # sum to 1 again
    preds = preds.unsqueeze(dim=-1)  # [[0.2, 0.2, 0.6], [0.4, 0.5, 0.1]]
    ones = [1] * len(preds.shape)
    preds = preds.repeat(*ones, 3)  # [[[0.2, 0.2], [0.2, 0.2] [0.6, 0.6]], [[0.4, 0.4]...

    picked_coords = class_map[
        indices
    ]  # mask with indices, new column is added where data is concated. Pick only the first row [0] and drop the rest with squeeze
    scaled_coords = picked_coords * preds
    weighted_sum = torch.sum(scaled_coords, dim=-2).squeeze()
    return weighted_sum


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    pass
