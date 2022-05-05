from __future__ import annotations, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.resnet import ResNet
from sklearn.preprocessing import MinMaxScaler
from utils_functions import tensor_sum_of_elements_to_one
from utils_geo import crs_coords_to_degree, haversine_from_degs


def get_haversine_from_predictions(
    crs_scaler: MinMaxScaler, pred_crs_coord: torch.Tensor, image_true_crs_coords: torch.Tensor
):
    pred_crs_coord = pred_crs_coord.cpu()
    image_true_crs_coords = image_true_crs_coords.cpu()

    pred_crs_coord_transformed = crs_scaler.inverse_transform(pred_crs_coord)
    true_crs_coord_transformed = crs_scaler.inverse_transform(image_true_crs_coords)

    pred_degree_coords = crs_coords_to_degree(pred_crs_coord_transformed)
    true_degree_coords = crs_coords_to_degree(true_crs_coord_transformed)
    return haversine_from_degs(pred_degree_coords, true_degree_coords)


def model_remove_fc(model: ResNet):
    """Replaces fully connected layer with identity function"""
    if type(model) is ResNet:
        model.fc = Identity()
    if type(model) is EfficientNet:
        model.classifier = Identity()
    return model


def get_model_blocks(model: nn.Module):
    """
    returns list of model blocks without last layer (classifier/fc)
    """
    if type(model) is EfficientNet:
        return list(model.features)  # model.children())[-3]
    if type(model) is ResNet:
        return list(model.children())  # model.children())[-3]
    return list(model.children())


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


def crs_coords_weighed_mean(y_pred: torch.Tensor, class_map, top_k: int) -> torch.Tensor:
    """
    Args:
        y_pred: tensor of shape (N, C)
        class_map: tensor of shape (N). Hold crs for each class
    Returns
    """
    preds, indices = torch.topk(F.softmax(y_pred, dim=-1), k=top_k)
    preds = tensor_sum_of_elements_to_one(preds, dim=1)
    preds = preds.unsqueeze(dim=-1)  # [[0.2, 0.2, 0.6], [0.4, 0.5, 0.1]]
    ones = [1] * len(preds.shape)

    # repeat every axis once (change nothing), but repeat the last axis twice because of lat,lng [[[0.2, 0.2], [0.2, 0.2] [0.6, 0.6]], [[0.4, 0.4]...
    preds = preds.repeat(*ones, 2)

    # index class_map with indices, new column is added where data is concated. Pick only the first row [0] and drop the rest with squeeze
    picked_coords = class_map[indices]

    scaled_coords = picked_coords * preds
    weighted_sum = torch.sum(scaled_coords, dim=-2).squeeze(dim=0)
    return weighted_sum


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    pass
