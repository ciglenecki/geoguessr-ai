from utils_paths import WORK_DIR
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet

torch.set_printoptions(threshold=10_000)


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

    model: ResNet = torch.hub.load("pytorch/vision:v0.12.0", "resnext101_32x8d", pretrained=True)
    model.fc = Identity()
    model.eval()

    channels = 3
    image_size = 2
    batch_size = 3
    num_images = 5

    torch.manual_seed(0)
    image_batch_list = [torch.rand(batch_size, channels, image_size, image_size)] * num_images
    model_output_list = [model(image) for image in image_batch_list]
    model_output_cat = torch.cat(model_output_list, dim=0)
    model_output_stack = torch.stack(model_output_list, dim=0)
    image_batch = image_batch_list[0]

    print("Image batch shape:", image_batch.shape, sep="\n", end="\n\n")
    print("Image batch with 1 channel:", image_batch[:, 0], sep="\n", end="\n\n")  # all images in a batch but 1 channel
    print("Output image shape:", model_output_list[0].shape, sep="\n", end="\n\n")
    print("Output image example:", model_output_list[0][0:2, 0:3], sep="\n", end="\n\n")
    print("Output (cat) shape", model_output_cat.shape, sep="\n", end="\n\n")  # 15 x 2048
    print("Output (cat)", model_output_cat[0:3, 0:5], sep="\n", end="\n\n")
    print("Output (stacked) shape", model_output_stack.shape, sep="\n", end="\n\n")  # 5 x 3 x 2048
    print("Output (stacked)", model_output_stack[:, :, 0:5], sep="\n", end="\n\n")

    print("\n\n============B===========\n\n\n")
    print("Instead of 8 batches that go through forward 4 times, this time we will stack 4 images on each batch => 8 x 4 = 32 batches")

    torch.manual_seed(0)
    image_batch_cat = torch.cat(image_batch_list, dim=0)  # 15 x 3 x 2 x 2
    output = model(image_batch_cat)  # 15 x 2048

    # we are decomposing 15 to 5 and 3, *(output.shape[1:]) represents the rest of the output shape (2048)
    output_restacked = output.reshape(num_images, batch_size, *(output.shape[1:]))

    print("Image batch shape:", image_batch.shape, sep="\n", end="\n\n")
    print("Image batch with 1 channel:", image_batch[:, 0], sep="\n", end="\n\n")

    print("Output image shape:", output.shape, sep="\n", end="\n\n")
    print("Output image example:", output[0:3, 0:5], sep="\n", end="\n\n")

    print("Ouput (stacked) shape", output_restacked.shape)  # 5 x 3 x 2048
    print("Ouput (stacked) example:", output_restacked[:, :, 0:5])
    exit(1)


if __name__ == "__main__":
    # mini_tensor()
    model_cat_vs_multi_forward()
