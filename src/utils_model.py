from __future__ import annotations, division, print_function

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
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


def plot_filters_single_channel_big(t):
    # setting the rows and columns
    nrows = t.shape[0] * t.shape[2]
    ncols = t.shape[1] * t.shape[3]

    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap="gray", ax=ax, cbar=False)

    plt.tight_layout()
    plt.show()


def plot_filters_single_channel(t):
    # kernels depth * number of kernels
    nplots = t.shape[0] * t.shape[1]
    ncols = 12

    nrows = 1 + nplots // ncols
    # convert tensor to numpy image
    # npimg = np.array(t.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + "," + str(j))
            ax1.axis("off")
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()


def plot_filters_multi_channel(t):
    # get the number of kernals
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis("off")
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.savefig("myimage.png", dpi=100)
    plt.tight_layout()
    plt.show()


def plot_weights(model, layer_num, single_channel=True, collated=False):
    # extracting the model features at the particular layer number
    layer = model.backbone.layer4._modules["2"].conv3

    # checking whether the layer is convolution layer or not
    if isinstance(layer, nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = model.backbone.conv1.weights.data

        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor)
            else:
                plot_filters_single_channel(weight_tensor)

        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor)
            else:
                print("Can only plot weights with three channels with single channel = False")

    else:
        print("Can only visualize layers which are convolutional")


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
