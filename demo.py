import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import json
import glob
import tqdm
import cv2
import shutil
from PIL import Image

from parsing.im2tensor import im2tensor

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def sample_mask(parsing_anno, save_path=None):
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    mask = np.full((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]), 0)# + 255

    num_of_class = vis_parsing_anno.max()+1
    for pi in range(1, num_of_class):
        mask[vis_parsing_anno == pi] = pi*40

    mask = mask.astype(np.uint8)
    cv2.imwrite(save_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def sample_images(img_path, generator, transform, to_tensor, channels=3):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = im2tensor(img, to_tensor, transform)

    real_A = Variable(img.type(Tensor))
    fake_A, fake_B, fake_mask, feature = generator(real_A)

    # images of D and M
    name_mask = img_path.replace(opt.rgb.split('/')[-2], 'M_out')
    name_depth = img_path.replace(opt.rgb.split('/')[-2], 'D_out')

    # save D M images
    depth = tensor2im(fake_B)
    cv2.imwrite(name_depth, depth)

    # parsing = fake_mask.squeeze(0).cpu().numpy().argmax(0)
    # sample_mask(parsing, save_path=name_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, default=None, required=True, help="path of testing images")
    parser.add_argument("--type", type=str, default='jpg', help="jpg or png")
    parser.add_argument("--channels", type=int, default=3)
    opt = parser.parse_args()

    # ----------
    #  saving path
    # ----------
    # images of D and M
    out_mask = opt.rgb.replace(opt.rgb.split('/')[-2], 'M_out')
    out_depth = opt.rgb.replace(opt.rgb.split('/')[-2], 'D_out')

    # images of D and M
    if out_mask[-1] != '/':
        out_mask = out_mask + '/'
    elif out_depth[-1] != '/':
        out_depth = out_depth + '/'

    print('Save mask folder: {:}, Save depth folder: {:}'.format(out_mask, out_depth))


    # images of D and M
    try:
        os.stat(out_mask)
    except:
        os.makedirs(out_mask, exist_ok=True)
    try:
        os.stat(out_depth)
    except:
        os.makedirs(out_depth, exist_ok=True)


    # ----------
    # Load DMNet
    # ----------
    generator = DM_Generator(in_channels=opt.channels, out_channels=3)
    generator = generator.cuda()
    generator.load_state_dict(torch.load('ckpt/generator_5.pth'))
    generator.eval()


    # ----------
    # Configure dataloaders
    # ----------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    # ----------
    # evaluation
    # ----------
    imgs = sorted(glob.glob(opt.rgb + '*/*.' + opt.type))
    print('Number of images: ', len(imgs))

    with torch.no_grad():               
        for image_path in tqdm.tqdm(imgs):
            if '.jpg' not in image_path and '.png' not in image_path:
                continue

            # images of D and M
            fold_mask = image_path.replace(opt.rgb, out_mask)
            fold_mask = fold_mask.replace(fold_mask.split('/')[-1], '')
            if not os.path.exists(fold_mask):
                os.makedirs(fold_mask)
                
            fold_depth = image_path.replace(opt.rgb, out_depth)
            fold_depth = fold_depth.replace(fold_depth.split('/')[-1], '')
            if not os.path.exists(fold_depth):
                os.makedirs(fold_depth)

            sample_images(image_path, generator, transform, to_tensor, opt.channels)