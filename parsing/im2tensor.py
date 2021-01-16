from parsing.logger import setup_logger
from parsing.model import BiSeNet

import torch

import glob
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import tqdm
import shutil

def im2tensor(img, to_tensor, transform):
    net = BiSeNet(n_classes=19)
    net.cuda()
    net.load_state_dict(torch.load('ckpt/new_256.pth'))
    net.eval()

    img1 = to_tensor(img)
    img2 = transform(img)

    img = torch.unsqueeze(img1, 0)
    img = img.cuda()
    if img.shape[1] != 3:
        img = img[:, :3, :, :]
    parsing = net(img)[0]
    parsing = parsing.argmax(1,keepdim=True).to(torch.uint8)
    if parsing.shape[1] != 3:
        parsing = parsing.repeat(1,3,1,1)

    img = torch.unsqueeze(img2, 0)
    img = img.cuda()
    if img.shape[1] != 3:
        img = img.repeat(1,3,1,1)
    
    pars = (parsing==1) | (parsing==2) | (parsing==3) |(parsing==4) | (parsing==5) |(parsing==6) | (parsing==10) | (parsing==11) | (parsing==12) | (parsing==13)
    img[pars==0] = -1
    return img