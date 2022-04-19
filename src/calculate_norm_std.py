import torch
from PIL import Image
from torchvision.transforms import transforms


def calculate_norm_std(image_cache):
    transform = transforms.ToTensor()
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for images in image_cache.values():
        images = [transform(Image.open(image_path)) for image_path in images]
        for image in images:
            channels_sum += torch.mean(image, dim=[1, 2])
            channels_squared_sum += torch.mean(image ** 2, dim=[1, 2])
            num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print("Dataset mean: " + str(mean))
    print("Dataset std: " + str(std))

    return mean, std

