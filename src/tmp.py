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
    backbone: ResNet = torch.hub.load("pytorch/vision:v0.12.0", "resnext101_32x8d", pretrained=True)
    backbone.fc = Identity()
    backbone.eval()

    image1 = torch.rand(1, 3, 2, 2)
    image2 = torch.rand(1, 3, 2, 2)
    batch = torch.cat([image1, image2])
    print(batch.shape)

    out1 = backbone.forward(image1)
    out2 = backbone.forward(image2)
    out_batch = backbone.forward(batch)

    print("Out 1", out1[0:5, 0:5])
    print("Out 2", out2[0:5, 0:5])
    print("Out batch", out_batch[0:5, 0:5])


def model_tesnor_test():
    def get_sample(tensor):
        return tensor[0:5, 0:5, 0:3]

    num_of_classes = 10
    criterion = nn.CrossEntropyLoss()

    labels = torch.rand(8) * num_of_classes // 1

    backbone: ResNet = torch.hub.load("pytorch/vision:v0.12.0", "resnext101_32x8d", pretrained=True)
    backbone.fc = Identity()
    backbone.eval()
    fc = nn.Linear(8192, num_of_classes)

    image_size = 2
    batch_size = 3
    num_images = 5

    # VERSION A ver 1 (8, (4) 3, 24, 24)
    torch.manual_seed(0)
    images_list = [torch.rand(batch_size, 3, image_size, image_size)] * num_images

    single_image_batch = images_list[0]

    backbone_output_list = [backbone(image) for image in images_list]
    backbone_output_tensor = torch.cat(backbone_output_list, dim=0)
    backbone_output_stacked = torch.stack(backbone_output_list, dim=0)

    print("Single image batch", single_image_batch.shape)
    print("Image 0", single_image_batch[:, 0])  # all batches 1 channel
    print("Output SINGLE shape", backbone_output_list[0].shape)
    print("Output", backbone_output_list[0][0:2, 0:3])
    print("Output cat shape", backbone_output_tensor.shape)  # 15 x 2048
    print("Output cat", backbone_output_tensor[0:3, 0:5])
    print("Output stacked shape", backbone_output_stacked.shape)  # 5 x 3 x 2048
    print("Output stacked", backbone_output_stacked[0:2, 0:2, 0:5])

    # ver2
    torch.manual_seed(0)
    input_four = torch.cat(images_list, dim=0)
    input_four_output = backbone(input_four)
    print("\n\n\n\n")

    print("Input shape", input_four.shape)
    print("Image B", input_four[0:batch_size, 0])
    print("B Model output shape", input_four_output.shape)
    print("B Model output shape", input_four_output[0:3, 0:5])
    input_four_re = input_four_output.reshape(num_images, batch_size, *(input_four_output.shape[1:]))
    print("input_four_re", input_four_re.shape)
    print("Output B", input_four_re)
    exit(1)
    flattened = torch.flatten(input_four_re, 1)
    print(flattened[0:2])
    out = fc(flattened)

    backbone.zero_grad()
    loss = criterion(out, labels.long())
    loss.backward()
    b_grads = backbone.conv1.weight.grad
    print("===========B============")
    print(b_grads)


if __name__ == "__main__":
    # mini_tensor()
    model_tesnor_test()
