import json
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter
import os
import random
import numpy as np
from time import time
import argparse
import json
from pathlib import Path
from validation import validation_binary, validation_multi
import utils
import sys
from torch import nn
from models import  LinkNet34, UNet, UNet16, AlbuNet
import matplotlib.pyplot as plt
from loss import LossBinary, LossMulti,FocalLoss
from dataset import RoboticsDataset
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from prepare_train_val import get_split
# import torch.backends.cudnn
import torch
from TransModels import MultiFrameFusionUNet11,UNet11,MultiFrameFusionUNet11_ViTViT
from RAUNet import RAUNet,TeRAUNet,MultiFrameFusionRAUNet11,MultiFrameFusionRAUNet_v2,\
    MultiFrameFusionRAUNet11_2trans
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop
)

moddel_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34,
               'RAUNet':RAUNet,
               'MultiFrameFusionUNet11_ViTViT':MultiFrameFusionUNet11_ViTViT,
               'MultiFrameFusionUNet11':MultiFrameFusionUNet11,
               'MultiFrameFusionRAUNet11':MultiFrameFusionRAUNet11,
               'MultiFrameFusionRAUNet_v2':MultiFrameFusionRAUNet_v2,
               'MultiFrameFusionRAUNet11_2trans':MultiFrameFusionRAUNet11_2trans,
               'TeRAUNet':TeRAUNet}

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.3, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--model_type',type=str,default='transformer_fusion',help='whether use transformer to fuse mutiframes feature')
    arg('--fold', type=int, help='fold', default=3)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--heads', type=int, default=4)
    arg('--n-epochs', type=int, default=300)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg('--train_crop_height', type=int, default=960)
    arg('--train_crop_width', type=int, default=960)
    arg('--val_crop_height', type=int, default=960)
    arg('--val_crop_width', type=int, default=960)
    arg('--type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--model', type=str, default='MultiFrameFusionUNet11_ViTViT', choices=moddel_list.keys())

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    if not utils.check_crop_size(args.train_crop_height, args.train_crop_width):
        print('Input image sizes should be divisible by 32, but train '
              'crop sizes ({train_crop_height} and {train_crop_width}) '
              'are not.'.format(train_crop_height=args.train_crop_height, train_crop_width=args.train_crop_width))
        sys.exit(0)

    if not utils.check_crop_size(args.val_crop_height, args.val_crop_width):
        print('Input image sizes should be divisible by 32, but validation '
              'crop sizes ({val_crop_height} and {val_crop_width}) '
              'are not.'.format(val_crop_height=args.val_crop_height, val_crop_width=args.val_crop_width))
        sys.exit(0)

    if args.type == 'parts':
        num_classes = 4
    elif args.type == 'instruments':
        num_classes = 8
    else:
        num_classes = 1

    if args.model_type == 'transformer_fusion':
        # print('......')
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True,
                                       h=320,w=256,numheads=args.heads)
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True)

    if args.model == 'TeRAUNet':
        model = TeRAUNet(num_classes=num_classes, pretrained=True,numheads=args.heads)
    elif args.model == 'MultiFrameFusionRAUNet11':
        # print(args.model)
        model = MultiFrameFusionRAUNet11(num_classes=num_classes, pretrained=True,
                                       h=320,w=256,numheads=args.heads)
    elif args.model == 'MultiFrameFusionRAUNet_v2':
        # print(args.model)
        model = MultiFrameFusionRAUNet_v2(num_classes=num_classes, pretrained=True,
                                       h=320,w=256)


    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    cudnn.benchmark = True

    def cuda(x):
        return x.cuda() if torch.cuda.is_available() else x

    def make_loader(file_names, shuffle=True, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(3, gap=10)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            # PadIfNeeded(min_height=args.train_crop_height, min_width=args.train_crop_width, p=1),
            # RandomCrop(height=args.train_crop_height, width=args.train_crop_width, p=1),
            # VerticalFlip(p=0.5),
            # HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            # PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
            # CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=args.type,
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type,
                               batch_size=len(device_ids))

    root = Path(args.root)
    model_path = root / 'model_{model}_{fold}_10%data_head{heads}_raw.pt'.format(model=args.model, fold=3, heads=args.heads) if args.model == 'TeRAUNet' or \
                                                                                                                                   args.model_type == 'transformer_fusion' else root / 'model_{model}_{fold}_10%data_raw.pt'.format(
        model=args.model, fold=3)

    if model_path.exists():
        print(model_path)
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))

    tl = train_loader
    count = 0


    for i, (inputs, targets) in enumerate(tl):
        if count >2:
            break
        t1 = time()
        print('shape:', inputs.shape)
        input_show = inputs.numpy()
        inputs = cuda(inputs)  # .unsqueeze(dim=0)

        t2 = time()

        torch.cuda.empty_cache()
        with torch.no_grad():
            targets = cuda(targets)

            outputs,trans_input,trans_output = model(inputs,return_map=True)
        print(trans_input.shape,trans_output.shape)
        trans_output,trans_input = trans_output.cpu().detach().numpy(),trans_input.cpu().detach().numpy()
        for i in range(512):
            print(i)
            plt.subplot(1,2,1)
            plt.imshow(trans_input[0,i],cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(trans_output[0,i],cmap='gray')
            plt.show()
        count += 1

if __name__ == "__main__":
    main()




