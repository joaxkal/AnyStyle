#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def img_denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to("cuda")
    std = torch.tensor([0.229, 0.224, 0.225]).to("cuda")
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = image * std + mean
    return image


def img_normalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).to("cuda")
    std = torch.tensor([0.229, 0.224, 0.225]).to("cuda")
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize(image):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to("cuda")
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to("cuda")
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize_batch(images):
    # images: (B, C, H, W)
    images = F.interpolate(images, size=224, mode='bicubic')

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)

    return (images - mean) / std

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


def load_image(img_path, target_size=256, augment=False):

    image = Image.open(img_path).convert("RGB")
    image = TF.to_tensor(image)[:3]  # -> (3, H, W)

    _, H, W = image.shape

    if H < W:
        ratio = W / H
        new_H = target_size
        new_W = int(ratio * new_H)
    else:
        ratio = H / W
        new_W = target_size
        new_H = int(ratio * new_W)

    image = TF.resize(image, (new_H, new_W))
    if augment:
        image = transforms.RandomCrop(target_size)(image)
    else:
        image = transforms.CenterCrop(target_size)(image)

    return image.unsqueeze(0)
