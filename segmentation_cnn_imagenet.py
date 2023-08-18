import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image, ImageFilter

import methods
from segmentation_utils.metrices import batch_intersection_union
from segmentation_utils.metrices import batch_pix_accuracy
from segmentation_utils.metrices import get_ap_scores
from segmentation_utils.metrices import get_f1_scores


class Imagenet_Segmentation(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        # self.h5py = h5py.File(path, 'r+')
        self.h5py = None
        print(path)
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return self.data_length


def apply_threshold(map):
    meanval = map.flatten().mean()
    new = np.where(map > meanval, 255, 0).astype(np.uint8)
    return new


def eval_batch(Res, labels):
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=0) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("outputs", outputs.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(outputs, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


ITERATION = 'iia'
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from tqdm import tqdm

    device = 'cuda'

    # Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    ds = Imagenet_Segmentation('gtsegs_ijcv.mat',
                               transform=test_img_trans, target_transform=test_lbl_trans)

    operations = ['iia']

    segmentation_results = {}
    for operation in operations:
        segmentation_results[f'{operation}_IoU'] = 0
        segmentation_results[f'{operation}_mAP'] = 0
        segmentation_results[f'{operation}_pixAcc'] = 0
    results = []
    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = [], []
    for i, (img, tgt) in enumerate(tqdm(ds)):
        segmentation_results = {}

        models = ['densnet', 'convnext', 'resnet101']
        layer_options = [12, 8]
        model_name = models[-1]
        FEATURE_LAYER_NUMBER = layer_options[-1]

        heatmaps = methods.generate_heatmap(models[-1], layer_options[-1], operations, img, device=device)
        op_idx = 0
        for operation in operations:
            map = heatmaps[op_idx]
            correct, labeled, inter, union, ap, f1, pred, target = eval_batch(
                torch.tensor(map).unsqueeze(0).unsqueeze(0),
                torch.tensor(tgt).unsqueeze(0))

            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            total_ap += [ap]
            total_f1 += [f1]
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            mAp = np.mean(total_ap)
            mF1 = np.mean(total_f1)
            segmentation_results[f'{operation}_IoU'] = mIoU
            segmentation_results[f'{operation}_mAP'] = mAp
            segmentation_results[f'{operation}_pixAcc'] = pixAcc
            segmentation_results[f'{operation}_mF1'] = mF1
            op_idx += 1
