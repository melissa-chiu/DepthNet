import glob
import random
import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset_chmask(Dataset):
    def __init__(self, root, img_size=(256, 256), data_type='bu', mode="train"):
        transforms_rgbd = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transforms_mask = [
            transforms.ToTensor(),
        ]
        self.transform_rgbd = transforms.Compose(transforms_rgbd)
        self.transform_mask = transforms.Compose(transforms_mask)
        self.img_size = img_size
        self.mode = mode

        if data_type == 'bu':
            print('Train on BU3DFE.', root)
                
            self.files_A = sorted(glob.glob(os.path.join(root, "D_gt") + "/*/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, "RGB") + "/*/*.*"))
        else:
            raise 'Not valid training dataset.'

        print('Number of images A: %d B: %d' % (len(self.files_A), len(self.files_B)))


    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])
        
        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        
        item_A = self.transform_rgbd(image_A)
        item_B = self.transform_rgbd(image_B)

        # 7 classes ['background, 'skin', 'brows', 'eyes', 'eye_g', 'nose', 'mouth']
        name = self.files_B[index % len(self.files_B)].replace('RGB', 'M_gt')
        mask_label = np.array(Image.open(name).convert('L'))
        
        mask = np.zeros(self.img_size)
        mask[mask_label == 0] = 255
        mask = mask.astype(np.uint8)
        item_mask = self.transform_mask(Image.fromarray(mask))
        
        for i in range(1,7):
            mask = np.zeros(self.img_size)
            mask[mask_label == i*40] = 255
            mask = mask.astype(np.uint8)
            mask_ = self.transform_mask(Image.fromarray(mask))
            item_mask = torch.cat((item_mask, mask_), dim=0)
        
        # if self.mode != 'train':
        #     print(self.files_B[index % len(self.files_B)])
        #     print(self.files_A[index % len(self.files_A)])
        #     print()

        return {"A": item_A, "B": item_B, "mask": item_mask}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDataset(Dataset):
    def __init__(self, root, img_size=(256, 256), data_type='bu', mode="train"):
        transforms_rgbd = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transform_rgbd = transforms.Compose(transforms_rgbd)
        self.img_size = img_size
        self.mode = mode

        if data_type == 'bu':
            print('Train on BU3DFE.', root)
            self.files_A = sorted(glob.glob(os.path.join(root, "D_gt") + "/*/*.*"))
            self.files_B = sorted(glob.glob(os.path.join(root, "RGB") + "/*/*.*"))
        else:
            raise 'Not valid training dataset.'

        print('Number of images A: %d B: %d' % (len(self.files_A), len(self.files_B)))


    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])
        
        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        
        item_A = self.transform_rgbd(image_A)
        item_B = self.transform_rgbd(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



if __name__ == "__main__":
    data = ImageDataset_chmask(root='/home/timmy/Datasets/bu_align/train/', img_size=(256, 256), data_type='bu')
    batch = data[0]
    real_mask = batch['mask']
    