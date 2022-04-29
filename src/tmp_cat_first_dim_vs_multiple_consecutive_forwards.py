"""
pip install numpy torch torchvision
"""

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet
import random
import numpy as np

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(threshold=10_000)

SEED = 20
USE_ANOTHER_BATCH_OF_RANDOM_IMAGES_AS_SANITY_CHECK = False
FREEZE_BATCHNORM_LAYERS = False
channels = 3
image_size = 16
batch_size = 8
consecutive_forwads_num = 4
# MODEL_NAME = "resnet50"
MODEL_NAME = "resnext101_32x8d"

"""
note:
    - for resnet50 the gradients are the same, whether we use FREEZE_BATCHNORM_LAYERS True or False
    
    - for resnext101_32x8d the gradients are different in both cases, why? Also worth noting; increasing the consecutive_forwads_num parameter doesn't scale the difference between cat_forward gradients and consecutive_forwards gradients.
"""


def log(*args, **kwags):
    print(*args, **kwags, sep="\n", end="\n\n")


def get_model_grads_sample(model):
    return model.conv1.weight.grad[30:32, 0:2, 3:4, 3:6]


def get_model_weights_sample(model):
    return model.conv1.weight[30:32, 0:2, 3:4, 3:6]


def get_setup(
    batch_size,
    consecutive_forwads_num,
    channels,
    image_size,
    model_name,
    freeze_batchnorm_layers=False,
):
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(SEED)

    model: ResNet = torch.hub.load("pytorch/vision:v0.12.0", model_name, pretrained=True)
    # model.fc = Identity()

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
        super().__init__()

    def forward(self, x):
        return x


def freeze_batchnorm(model: ResNet):
    for (param_name, param) in model.named_modules():
        if "bn" in param_name or "downsample.1" in param_name:  # downsample.1 is batchnorm in resnet18
            param.requires_grad_(False)
            param.affine = False


def main():
    log("CONSECUTIVE FORWARD ======================")

    model, criterion, optimizer, labels, image_batch_list = get_setup(
        batch_size,
        consecutive_forwads_num,
        channels,
        image_size,
        MODEL_NAME,
        freeze_batchnorm_layers=FREEZE_BATCHNORM_LAYERS,
    )
    output = forward_consecutive(model, image_batch_list)
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    grads_consecutive_forward = get_model_grads_sample(model)
    optimizer.step()

    log("CAT FORWARD ======================")
    log(
        "Instead of 8 batches that go through forward 4 times, this time we will concat 4 images on the first dimension. This will increase the batch size => 8 x 4 = 32 batches"
    )

    model, criterion, optimizer, labels, image_batch_list = get_setup(
        batch_size,
        consecutive_forwads_num,
        channels,
        image_size,
        MODEL_NAME,
        freeze_batchnorm_layers=FREEZE_BATCHNORM_LAYERS,
    )
    output = forward_cat(model, image_batch_list)
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    grads_cat_forward = get_model_grads_sample(model)
    optimizer.step()

    if USE_ANOTHER_BATCH_OF_RANDOM_IMAGES_AS_SANITY_CHECK:
        log("RANDOM IMAGES - CONSECUTIVE FORWARD ======================")
        """ Used as a sanity check to confirm that the model is getting completely different gradients for completely different batch of images"""

        model, criterion, optimizer, labels, _ = get_setup(
            batch_size,
            consecutive_forwads_num,
            channels,
            image_size,
            MODEL_NAME,
            freeze_batchnorm_layers=FREEZE_BATCHNORM_LAYERS,
        )
        # use different seed to ensure images are different
        torch.manual_seed(99)
        image_batch_list_different = [
            torch.rand(batch_size, channels, image_size, image_size)
        ] * consecutive_forwads_num
        output = forward_cat(model, image_batch_list_different)
        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        grads_c = get_model_grads_sample(model)
        optimizer.step()

    log("COMPARTING GRADIENTS ======================")
    log("Gradient CONSECUTIVE FORWARD:", grads_consecutive_forward)
    log("Gradient CAT FORWARD:", grads_cat_forward)

    if USE_ANOTHER_BATCH_OF_RANDOM_IMAGES_AS_SANITY_CHECK:
        log("Gradient different batch -- CAT FORWARD:", grads_c)


if __name__ == "__main__":
    main()
