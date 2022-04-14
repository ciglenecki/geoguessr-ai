import random

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=10_000)

SEED = 20


def log(*args, **kwags):
    print(*args, **kwags, sep="\n", end="\n\n")


def get_model_grads_sample(model):
    return model.conv1.weight.grad[32:34, 0:2, 3:4, 3:6]


def get_model_weights_sample(model):
    return model.conv1.weight[30:33, 0:2, 2:3, 2:3]


def get_setup(batch_size, consecutive_forwads_num, channels, image_size, model_name="resnet18", freeze_batchnorm_layers=False):

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(SEED)

    model: ResNet = torch.hub.load("pytorch/vision:v0.12.0", model_name, pretrained=True)
    model.fc = Identity()

    if freeze_batchnorm_layers:
        freeze_batchnorm(model)

    torch.manual_seed(SEED)
    criterion = nn.NLLLoss()

    torch.manual_seed(SEED)
    optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.2, weight_decay=0)

    torch.manual_seed(SEED)
    labels = torch.randint(0, 3, size=(batch_size * consecutive_forwads_num, 1)).squeeze()

    torch.manual_seed(SEED)
    image_batch_list = [torch.rand(batch_size, channels, image_size, image_size)] * consecutive_forwads_num

    torch.manual_seed(SEED)

    return model, criterion, optimizer, labels, image_batch_list


def forward_cat(model, image_batch_list):

    image_batch_cat = torch.cat(image_batch_list, dim=0)
    output = model(image_batch_cat)

    log("Model weights", get_model_weights_sample(model))
    log("Image batch shape:", image_batch_cat.shape)
    log("Output image shape:", output.shape)
    log("Output image example:", output[0:3, 0:5])
    return output


def forward_consecutive(model, image_batch_list):
    single_image_batch = image_batch_list[0]
    output_list = [model(image) for image in image_batch_list]
    output_stack = torch.stack(output_list, dim=0)
    output_cat = torch.cat(output_list, dim=0).float()

    log("Model weights", get_model_weights_sample(model))
    log("Image batch shape:", single_image_batch.shape)
    log("Output image shape:", output_list[0].shape)
    log("Output (cat) shape", output_cat.shape)
    log("Output (cat)", output_cat[0:3, 0:5])
    log("Output (stacked) shape", output_stack.shape)
    log("Output (stacked)", output_stack[0:3, 0:3, 0:5])

    return output_cat


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def mini_tensor():
    model: ResNet = torch.hub.load("pytorch/vision:v0.12.0", "resnext101_32x8d", pretrained=True)
    model.fc = Identity()
    model.eval()

    image1 = torch.rand(1, 3, 2, 2)
    image2 = torch.rand(1, 3, 2, 2)
    batch = torch.cat([image1, image2])
    log(batch.shape)

    out1 = model.forward(image1)
    out2 = model.forward(image2)
    out_batch = model.forward(batch)

    log("Out 1", out1[0:5, 0:5])
    log("Out 2", out2[0:5, 0:5])
    log("Out batch", out_batch[0:5, 0:5])


def freeze_batchnorm(model: ResNet):
    for (param_name, param) in model.named_modules():
        if "bn" in param_name or "downsample.1" in param_name:  # downsample.1 is batchnorm in resnet18
            log("Freezing:", param_name)
            param.requires_grad_(False)
            param.affine = False


def model_cat_vs_multi_forward():

    channels = 3
    image_size = 16
    batch_size = 8
    consecutive_forwads_num = 4

    log("CONSECUTIVE FORWARD ======================")

    model, criterion, optimizer, labels, image_batch_list = get_setup(batch_size, consecutive_forwads_num, channels, image_size, freeze_batchnorm_layers=True)
    output = forward_consecutive(model, image_batch_list)
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    grads_consecutive_forward = get_model_grads_sample(model)
    optimizer.step()

    log("CAT FORWARD ======================")
    log("Instead of 8 batches that go through forward 4 times, this time we will stack 4 images on each batch => 8 x 4 = 32 batches")

    model, criterion, optimizer, labels, image_batch_list = get_setup(batch_size, consecutive_forwads_num, channels, image_size, freeze_batchnorm_layers=True)
    output = forward_cat(model, image_batch_list)
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    grads_cat_forward = get_model_grads_sample(model)
    optimizer.step()

    log("RANDOM IMAGES - CONSECUTIVE FORWARD ======================")
    """ Used as a sanity check to confirm that the model is getting completely different gradients for completely different batch of images"""

    model, criterion, optimizer, labels, _ = get_setup(batch_size, consecutive_forwads_num, channels, image_size)
    torch.manual_seed(99)  # use different seed to ensure images are different
    image_batch_list_different = [torch.rand(batch_size, channels, image_size, image_size)] * consecutive_forwads_num
    output = forward_cat(model, image_batch_list_different)
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    grads_c = get_model_grads_sample(model)
    optimizer.step()

    log("COMPARTING GRADIENTS ======================")
    log("Gradient CONSECUTIVE FORWARD:", grads_consecutive_forward)
    log("Gradient CAT FORWARD:", grads_cat_forward)
    log("Gradient different batch -- CAT FORWARD:", grads_c)


# def test():
#     m = nn.LogSoftmax(dim=1)
#     loss = nn.NLLLoss()
#     # input is of size N x C = 3 x 5
#     input = torch.randn(3, 5, requires_grad=True)
#     # each element in target has to have 0 <= value < C
#     target = torch.tensor([1, 0, 4])
#     out = m(input)
#     log(out.shape, target.shape)
#     output = loss(out, target)
#     output.backward()


if __name__ == "__main__":
    # test()
    # mini_tensor()
    model_cat_vs_multi_forward()
