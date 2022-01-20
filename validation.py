import numpy as np
import utils
from torch import nn
import torch


def validation_binary(model, criterion, valid_loader, num_classes=None):
    with torch.no_grad():
        model.eval()
        losses = []

        jaccard = []

        for inputs, targets in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            jaccard += get_jaccard(targets, (outputs > 0).float())

        valid_loss = np.mean(losses)  # type: float

        valid_jaccard = np.mean(jaccard).astype(np.float64)

        print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
        metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
        return metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())


def validation_multi(model: nn.Module, criterion, valid_loader, num_classes):
    with torch.no_grad():
        model.eval()
        losses = []
        rious=[]
        rdices=[]
        confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.uint32)
        for inputs, targets in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            # if outputs.shape[0] == 1:
            #     outputs = outputs.squeeze(dim=0)
            #     targets = targets.squeeze(dim=0)
            # # print(outputs.shape,targets.shape)
            # else:
            #     outputs = torch.unbind(outputs, dim=0)
            #     outputs = torch.cat(outputs, dim=0)
            #     targets = torch.unbind(targets, dim=0)
            #     targets = torch.cat(targets, dim=0)
            #print(outputs.shape,targets.shape)
            s1=outputs.size()
            s2=targets.size()
            #outputs = outputs.view([s1[0]*s1[1], s1[2], s1[3], s1[4]])
            targets = targets.view([s2[0] * s2[1], s2[2], s2[3]])
            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            target_classes = targets.data.cpu().numpy()
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, num_classes)
            iou=general_jaccard(target_classes,output_classes)
            rious.append(iou)
            dicee=general_dice(target_classes,output_classes)
            rdices.append(dicee)
        confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
        valid_loss = np.mean(losses)  # type: float
        ious = {'iou_{}'.format(cls + 1): iou
                for cls, iou in enumerate(calculate_iou(confusion_matrix))}

        dices = {'dice_{}'.format(cls + 1): dice
                 for cls, dice in enumerate(calculate_dice(confusion_matrix))}

        average_iou = np.mean(list(ious.values()))
        average_dices = np.mean(list(dices.values()))
        r_dice=np.mean(rdices)
        r_iou=np.mean(rious)
        print(
            'Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}, real IoU: {:.4f}, real Dice: {:.4f}'.format(valid_loss,
                                                                                   average_iou,
                                                                                   average_dices,
                                                                                    r_iou,r_dice))
        metrics = {'valid_loss': valid_loss, 'iou': average_iou, 'iou1':r_iou}
        metrics.update(ious)
        metrics.update(dices)
        return metrics


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    #print('6666',prediction.shape,ground_truth.shape)
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)