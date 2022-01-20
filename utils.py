import json
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter
import os
import random
import numpy as np
from torch.optim import lr_scheduler


import torch
import tqdm
from time import time

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()

# _LRScheduler = lr_scheduler._LRScheduler


class WarmUpLR(lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{model}_{fold}_true10%data_head{heads}_clip{clip}raw.pt'.format(model=args.model,fold=fold,heads=args.heads,clip=args.clip) if args.model == 'TeRAUNet' or\
    args.model_type == 'transformer_fusion' else root / 'model_{model}_{fold}_true10%data_clip{clip}raw.pt'.format(model=args.model,fold=fold,clip=args.clip)
    if args.model_type == 'multidepth_transformer_fusion':
        model_path = root / 'model_{model}_{fold}_30%data_head{heads}_depth{depth}raw.pt'.format(model=args.model,
                                        fold=fold, heads=args.heads,depth=args.depth)
    if model_path.exists():
        print(model_path)
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, (str(model_path)))
    save_last = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, (str(model_path)+'last'))
    report_each = 10
    log = root.joinpath('train_{model}_true10%data_clip{clip}_{fold}.log'.format(model=args.model,clip=args.clip,fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    torch.cuda.empty_cache()
    losslist=[]
    dicelist=[]
    logs_dir = 'Logs/T{}_{}_heads_{}true10%data_clip{}/'.format(datetime.now().strftime('%Y%m%d_%H%M%S'),args.model,args.heads,args.clip) if args.model_type == 'transformer_fusion' \
    else 'Logs/T{}_{}_true10%data_clip{}/'.format(datetime.now().strftime('%Y%m%d_%H%M%S'),args.model,args.clip)
    if args.model_type == 'multidepth_transformer_fusion':
        logs_dir = 'Logs/T{}_{}_heads{}_depth{}30%data_/'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), args.model, args.heads,args.depth)

    os.mkdir(logs_dir)
    writer = SummaryWriter(logs_dir)
    iter_num = 0
    LR_STEP_SIZE = 40
    LR_GAMMA = 0.1
    # scheduler = lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)
    EPOCHES = n_epochs + 1 - epoch
    warmup_epoch = 10

    iter_per_epoch = len(train_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch,last_epoch=1)
    for epoch in range(epoch, n_epochs + 1):

        model.train()

        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
        losses = []

        tl = train_loader

        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                iter_num += 1
                # if epoch < 5:
                #     warmup_scheduler.step()
                #     warm_lr = warmup_scheduler.get_lr()
                    # print("warm_lr:%s" % warm_lr)

                t1 = time()

                inputs = cuda(inputs)#.unsqueeze(dim=0)

                t2=time()

                torch.cuda.empty_cache()

                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                # print(inputs.shape, targets.shape,outputs.shape)
                t3 = time()

                torch.cuda.empty_cache()

                loss = criterion(outputs, targets)
                t4=time()
                optimizer.zero_grad()

                batch_size = inputs.size(0)
                loss.backward()

                optimizer.step()
                # if epoch >= warmup_epoch:
                #     scheduler.step(epoch)
                #     learn_rate = scheduler.get_lr()[0]
                    # print("Learn_rate:%s" % learn_rate)
                    # scheduler.step(epoch)
                # lr_ = lr * (1.0 - iter_num / (EPOCHES * len(tl))) ** 0.9
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses)
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                torch.cuda.empty_cache()

                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            if epoch%10==0:
                # print('valid')
                valid_metrics = validation(model, criterion, valid_loader, num_classes)
                write_event(log, step, **valid_metrics)
                valid_loss = valid_metrics['valid_loss']
                valid_iou=valid_metrics['iou']
                real_iou=valid_metrics['iou1']
                losslist.append([mean_loss,valid_loss])
                dicelist.append([valid_iou,real_iou])
                valid_losses.append(valid_loss)
                file=open(logs_dir+'LossList.txt','w')
                for ip in losslist:
                    file.write(str(ip[0]))
                    file.write(' ')
                    file.write(str(ip[1]))
                    file.write('\n')
                file.close()
                file=open(logs_dir+'DiceList.txt','w')
                for ip in dicelist:
                    file.write(str(ip[0]))
                    file.write(' ')
                    file.write(str(ip[1]))
                    file.write('\n')
                file.close()
            save_last(epoch)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
