import argparse
import json
from pathlib import Path
from validation import validation_binary, validation_multi
from unet_con import SupConUnet
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import  LinkNet34, UNet, UNet16, AlbuNet
from loss import LossBinary, LossMulti,FocalLoss
from dataset import RoboticsDataset
import utils
import sys
from prepare_train_val import get_split
from TransModels import MultiFrameFusionUNet11,UNet11
from RAUNet import RAUNet,TeRAUNet,MultiFrameFusionRAUNet11,MultiFrameFusionRAUNet_v2,\
    MultiFrameFusionRAUNet11_2trans,TransRAUNet,FocalTransRAUNet11_2Trans,FocalTransRAUNet11,\
MultiFrameFusionRAUNet_v3
from TransModels import TeRAUNet_ViT,TeRAUNet_ViTFusion,MultiFrameFusionUNet11_ViTViT,\
    MultiFrameFusionUNet11_ViTViTv2,MultiFrameFusionUNet11_ViTViTv3

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
               'MultiFrameFusionRAUNet_v3':MultiFrameFusionRAUNet_v3,
               'TransRAUNet':TransRAUNet,
               'MultiFrameFusionUNet11_ViTViT':MultiFrameFusionUNet11_ViTViT,
               'MultiFrameFusionUNet11':MultiFrameFusionUNet11,
               'MultiFrameFusionRAUNet11':MultiFrameFusionRAUNet11,
               'MultiFrameFusionRAUNet_v2':MultiFrameFusionRAUNet_v2,
               'MultiFrameFusionRAUNet11_2trans':MultiFrameFusionRAUNet11_2trans,
               'FocalTransRAUNet11_2Trans':FocalTransRAUNet11_2Trans,
               'FocalTransRAUNet11':FocalTransRAUNet11,
               'TeRAUNet_ViT':TeRAUNet_ViT,
               'MultiFrameFusionUNet11_ViTViTv3':MultiFrameFusionUNet11_ViTViTv3,
               'MultiFrameFusionUNet11_ViTViTv2':MultiFrameFusionUNet11_ViTViTv2,
               'TeRAUNet_ViTFusion':TeRAUNet_ViTFusion,
               'TeRAUNet':TeRAUNet}


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.3, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--model_type',type=str,default='normal',help='whether use transformer to fuse mutiframes feature')
    arg('--fold', type=int, help='fold', default=3)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--heads', type=int, default=4)
    arg('--n-epochs', type=int, default=300)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg('--depth',type=int, default=4)
    arg('--clip', type=int, default=3)
    arg('--train_crop_height', type=int, default=960)
    arg('--train_crop_width', type=int, default=960)
    arg('--val_crop_height', type=int, default=960)
    arg('--val_crop_width', type=int, default=960)
    arg('--type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--model', type=str, default='RAUNet', choices=moddel_list.keys())

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
                                       h=320,w=256,numheads=args.heads,time_length=args.clip)

    elif args.model_type == 'multidepth_transformer_fusion':
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True,
                           h=320, w=256, numheads=args.heads,depth=args.depth)
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True)

    if args.model == 'UNet':
        # model = UNet(num_classes=num_classes)
        model=SupConUnet(num_classes=num_classes)
    elif args.model == 'TeRAUNet':
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

    if args.type == 'binary':
        loss = LossBinary(jaccard_weight=args.jaccard_weight)
    else:
        # print('type:')
        loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)
        # loss = FocalLoss()
        if args.model == 'RAUNet' or args.model == 'MultiFrameFusionRAUNet11'or \
                args.model == 'MultiFrameFusionRAUNet_v2' or args.model_type == 'transformer_fusion':
            print('using focal loss.......')
            loss=FocalLoss()
            # loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)
    cudnn.benchmark = True
    loss = FocalLoss()

    def make_loader(file_names, shuffle=True, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold,gap=50,clip=25)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            #PadIfNeeded(min_height=args.train_crop_height, min_width=args.train_crop_width, p=1),
            #RandomCrop(height=args.train_crop_height, width=args.train_crop_width, p=1),
            #VerticalFlip(p=0.5),
            #HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            #PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
            #CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=args.type,
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type,
                               batch_size=len(device_ids))

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    if args.type == 'binary':
        valid = validation_binary
    else:
        valid = validation_multi
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()
