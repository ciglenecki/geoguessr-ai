from utils_paths import WORK_DIR
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet
import random
import numpy as np

torch.autograd.set_detect_anomaly(True)

torch.set_printoptions(threshold=10_000)


def get_model_grads_sample(model):
    print(model.conv1.weight.grad.shape)
    return model.conv1.weight.grad[32:34, 0:2, 3:4, 3:6]


def get_model_weights_sample(model):
    return model.conv1.weight[30:33, 0:2, 2:3, 2:3]


def get_setup(batch_size, num_images, channels, image_size):

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(5)
    model: ResNet = torch.hub.load("pytorch/vision:v0.12.0", "resnet18", pretrained=True)
    model.fc = Identity()

    torch.manual_seed(5)
    criterion = nn.NLLLoss()

    torch.manual_seed(5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.2, weight_decay=0)

    torch.manual_seed(5)
    labels = torch.randint(0, 3, size=(batch_size * num_images, 1)).squeeze()

    torch.manual_seed(5)
    image_batch_list = [torch.rand(batch_size, channels, image_size, image_size)] * num_images
    torch.manual_seed(5)

    return model, criterion, optimizer, labels, image_batch_list


def forward_cat(model, image_batch_list):

    image_batch_cat = torch.cat(image_batch_list, dim=0)  # 15 x 3 x 2 x 2
    output = model(image_batch_cat)  # 15 x 2048

    print("Model weights", get_model_weights_sample(model))
    print("Image batch shape:", image_batch_cat.shape, sep="\n", end="\n\n")
    print("Image batch with 1 channel:", image_batch_cat[:, 0], sep="\n", end="\n\n")
    print("Output image shape:", output.shape, sep="\n", end="\n\n")
    print("Output image example:", output[0:3, 0:5], sep="\n", end="\n\n")


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
    print(batch.shape)

    out1 = model.forward(image1)
    out2 = model.forward(image2)
    out_batch = model.forward(batch)

    print("Out 1", out1[0:5, 0:5])
    print("Out 2", out2[0:5, 0:5])
    print("Out batch", out_batch[0:5, 0:5])


def model_cat_vs_multi_forward():

    """PURE x4 FORWARD"""

    channels = 3
    image_size = 3
    batch_size = 3
    num_images = 100

    model, criterion, optimizer, labels, image_batch_list = get_setup(batch_size, num_images, channels, image_size)

    model_output_list = [model(image) for image in image_batch_list]
    model_output_cat = torch.cat(model_output_list, dim=0).float()
    model_output_stack = torch.stack(model_output_list, dim=0)
    image_batch = image_batch_list[0]

    print("Model weights", get_model_weights_sample(model))
    print("Image batch shape:", image_batch.shape, sep="\n", end="\n\n")
    print("Image batch with 1 channel:", image_batch[:, 0], sep="\n", end="\n\n")  # all images in a batch but 1 channel
    print("Output image shape:", model_output_list[0].shape, sep="\n", end="\n\n")
    print("Output image example:", model_output_list[0][0:3, 0:5], sep="\n", end="\n\n")
    print("Output (cat) shape", model_output_cat.shape, sep="\n", end="\n\n")  # 15 x 2048
    print("Output (cat)", model_output_cat[0:3, 0:5], sep="\n", end="\n\n")
    print("Output (stacked) shape", model_output_stack.shape, sep="\n", end="\n\n")  # 5 x 3 x 2048
    print("Output (stacked)", model_output_stack[0:3, 0:3, 0:5], sep="\n", end="\n\n")

    optimizer.zero_grad()
    loss = criterion(model_output_cat, labels)
    loss.backward()
    grads_a = get_model_grads_sample(model)
    optimizer.step()

    """ CAT FORWARD """

    print("\n\n============B===========\n\n\n")
    print("Instead of 8 batches that go through forward 4 times, this time we will stack 4 images on each batch => 8 x 4 = 32 batches")
    model, criterion, optimizer, labels, image_batch_list = get_setup(batch_size, num_images, channels, image_size)

    image_batch_cat = torch.cat(image_batch_list, dim=0)  # 15 x 3 x 2 x 2
    output = model(image_batch_cat)  # 15 x 2048

    print("Model weights", get_model_weights_sample(model))
    print("Image batch shape:", image_batch_cat.shape, sep="\n", end="\n\n")
    print("Image batch with 1 channel:", image_batch_cat[:, 0], sep="\n", end="\n\n")
    print("Output image shape:", output.shape, sep="\n", end="\n\n")
    print("Output image example:", output[0:3, 0:5], sep="\n", end="\n\n")

    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    grads_b = get_model_grads_sample(model)
    optimizer.step()

    print("\n\n============C===========\n\n\n")

    model, criterion, optimizer, labels, _ = get_setup(batch_size, num_images, channels, image_size)
    torch.manual_seed(23)  # use different seed to ensure the image is not the same
    image_batch_list_different = [torch.rand(batch_size, channels, image_size, image_size)] * num_images
    image_batch_cat_different = torch.cat(image_batch_list_different, dim=0)  # 15 x 3 x 2 x 2
    output = model(image_batch_cat_different)  # 15 x 2048
    optimizer.zero_grad()
    loss = criterion(output, labels)
    loss.backward()
    grads_c = get_model_grads_sample(model)
    optimizer.step()

    print("Comparing gradients")
    print("Gradient A:\n", grads_a)
    print("Gradient B:\n", grads_b)
    print("Gradient C:\n", grads_c)


def test():
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss()
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    out = m(input)
    print(out.shape, target.shape)
    output = loss(out, target)
    output.backward()


if __name__ == "__main__":
    # test()
    # mini_tensor()
    model_cat_vs_multi_forward()
