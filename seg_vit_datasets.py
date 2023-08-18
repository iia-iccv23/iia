import os
import torch
import torch.utils.data as data
import numpy as np
import cv2
import pandas as pd
from torchvision.datasets import ImageNet

from PIL import Image, ImageFilter
import h5py
from glob import glob
import seg_methods_vit as methods
from vit_model import ViTmodel
from voc_dataset import VOCSegmentation
from coco_dataset import Coco_Segmentation
from segmentation_utils.metrices import get_iou
from segmentation_utils.metrices import get_ap_scores
from segmentation_utils.metrices import pixel_accuracy
from segmentation_utils.metrices import batch_pix_accuracy
from segmentation_utils.metrices import batch_intersection_union
from segmentation_utils.metrices import get_f1_scores


class ImageNet_blur(ImageNet):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        gauss_blur = ImageFilter.GaussianBlur(11)
        median_blur = ImageFilter.MedianFilter(11)

        blurred_img1 = sample.filter(gauss_blur)
        blurred_img2 = sample.filter(median_blur)
        blurred_img = Image.blend(blurred_img1, blurred_img2, 0.5)

        if self.transform is not None:
            sample = self.transform(sample)
            blurred_img = self.transform(blurred_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, blurred_img), target


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


class Imagenet_Segmentation_Blur(data.Dataset):
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

        gauss_blur = ImageFilter.GaussianBlur(11)
        median_blur = ImageFilter.MedianFilter(11)

        blurred_img1 = img.filter(gauss_blur)
        blurred_img2 = img.filter(median_blur)
        blurred_img = Image.blend(blurred_img1, blurred_img2, 0.5)

        # blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
        # blurred_img2 = np.float32(cv2.medianBlur(img, 11))
        # blurred_img = (blurred_img1 + blurred_img2) / 2

        if self.transform is not None:
            img = self.transform(img)
            blurred_img = self.transform(blurred_img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return (img, blurred_img), target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return self.data_length


class Imagenet_Segmentation_eval_dir(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 eval_path,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.h5py = h5py.File(path, 'r+')

        # 500 each file
        self.results = glob(os.path.join(eval_path, '*.npy'))

    def __getitem__(self, index):

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))
        res = np.load(self.results[index])

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        return len(self.h5py['/value/img'])


def apply_threshold(map):
    meanval = map.flatten().mean()
    new = np.where(map > meanval, 255, 0).astype(np.uint8)
    return new


def init_get_normalize_and_trns():
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    test_img_trans_only_resize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    return test_img_trans, test_img_trans_only_resize, test_lbl_trans


import matplotlib.pyplot as plt


def image_show(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


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


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from tqdm import tqdm

    model_name = 'vit-base'
    datasets_list = ['imagenet', 'coco', 'voc']
    operations = ['iia']
    chosen_dataset = datasets_list[1]
    device = 'cuda'

    # Data
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    if model_name == 'vit-base':
        model = ViTmodel.vit_base_patch16_224(pretrained=True).to(device)
    else:
        model = ViTmodel.vit_small_patch16_224(pretrained=True).to(device)

    if chosen_dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
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
        total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
        total_ap, total_f1 = [], []
        for i, (img, tgt) in enumerate(tqdm(ds)):
            segmentation_results = {}

            heatmaps = methods.generate_heatmap(model, operations, img, device=device)
            op_idx = 0
            for operation in operations:
                map = heatmaps[op_idx]
                model.zero_grad()
                correct, labeled, inter, union, ap, f1, pred, target = eval_batch(
                    torch.tensor(map).unsqueeze(0).unsqueeze(0),
                    tgt.unsqueeze(0))

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

    elif chosen_dataset == 'coco':
        test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_trns()
        CWD = os.getcwd()
        COCO_SEG_PATH = f'{CWD}/data/COCO/2017'

        ds = Coco_Segmentation(COCO_SEG_PATH,
                               transform=test_img_trans,
                               transform_resize=test_img_trans_only_resize, target_transform=test_lbl_trans)

        test_lbl_trans = transforms.Compose([
            transforms.Resize((224, 224), Image.NEAREST),
        ])

        segmentation_results = {}
        for operation in operations:
            segmentation_results[f'{operation}_IoU'] = 0
            segmentation_results[f'{operation}_mAP'] = 0
            segmentation_results[f'{operation}_pixAcc'] = 0

        total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
        total_ap, total_f1 = [], []
        for i, (img_norm, target, img_resize) in enumerate(tqdm(ds)):
            tgt = target
            img = img_resize
            segmentation_results = {}
            heatmaps = methods.generate_heatmap(model, operations, img, device=device)
            op_idx = 0
            for operation in operations:
                map = heatmaps[op_idx]
                model.zero_grad()
                correct, labeled, inter, union, ap, f1, pred, target = eval_batch(
                    torch.tensor(map).unsqueeze(0).unsqueeze(0),
                    tgt.unsqueeze(0))

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
    else:
        print('voc')
        test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_trns()
        CWD = os.getcwd()
        VOC_PATH = f'{CWD}/data/VOC/VOC_SEGMENTATION/'
        ds = VOCSegmentation(root=VOC_PATH, year='2012', image_set='val', download=False,
                             transform=test_img_trans,
                             target_transform=test_lbl_trans)

        total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
        total_ap, total_f1 = [], []
        for i, (img, target) in enumerate(tqdm(ds)):
            tgt = target
            segmentation_results = {}

            heatmaps = methods.generate_heatmap(model, operations, img, device=device)
            op_idx = 0
            for operation in operations:
                map = heatmaps[op_idx]
                model.zero_grad()
                correct, labeled, inter, union, ap, f1, pred, target = eval_batch(
                    torch.tensor(map).unsqueeze(0).unsqueeze(0),
                    tgt.unsqueeze(0))

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

    print('finished')
