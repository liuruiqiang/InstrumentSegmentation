import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.pytorch.transforms import img_to_tensor
from random import randint
from PIL import Image
from  time import time
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)



class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]

        image,mask=self.imgs2batch(img_file_name,self.mode)
        image=torch.cat(image,dim=0)
        mask=torch.cat(mask,dim=0)

        if self.mode == 'train':
            return image,mask
        else:
            return image, [str(i) for i in img_file_name]#str(img_file_name)
    def readfile(self,img_file_name,mode,rand_h,rand_v):

        image = load_image(img_file_name)

        mask = load_mask(img_file_name, self.problem_type)
        if rand_h: # image flip
            image=cv2.flip(image,1)
            mask=cv2.flip(mask,1)
        if rand_v: # image flip
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        data = {"image": image, "mask": mask}


        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]


        return img_to_tensor(image).unsqueeze(dim=0), torch.from_numpy(mask).long().unsqueeze(dim=0)
        #return img_to_tensor(image), torch.from_numpy(mask).long()
    def imgs2batch(self,filelist,mode):
        imgs=[]
        masks=[]
        rand_h=randint(0,1)
        rand_v = randint(0, 1)
        if mode!='train':
            rand_v=0
            rand_h=0
        for i in filelist:
            img,mask=self.readfile(i,mode,rand_h,rand_v)
            imgs.append(img)
            masks.append(mask)
        return imgs,masks

def load_image(path):
    img = cv2.imread(str(path))
    # img = cv2.copyMakeBorder(img, 0, 256, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) #padding image from 1280*1024 to 1280*1280
    img = cv2.resize(img, (320,256), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):

    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = prepare_data.binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = prepare_data.parts_factor
    elif problem_type == 'instruments':
        factor = prepare_data.instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)
    # mask = cv2.copyMakeBorder(mask, 0, 256, 0, 0, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.resize(mask, (320,256), interpolation=cv2.INTER_NEAREST)
    return (mask / factor).astype(np.uint8)



