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

IMAGE_SIZE = 'image_size'
BBOX = 'bbox'
IMAGE_PATH = 'image_path'
LABEL = 'label'


def voc_calculate_localization():
    global ROOT_IMAGES
    ROOT_IMAGES = "{0}/data/VOC/VOCdevkit/VOC2007/JPEGImages"

    images_by_label = {}
    with open(f'data/voc_bbox.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label, bbox_str, size = line.split('|')
            label = int(label)
            bbox_arr = bbox_str.split(',')
            bbox = [float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])]
            if label not in images_by_label:
                images_by_label[label] = [{IMAGE_PATH: file_name, BBOX: bbox, IMAGE_SIZE: size}]
            else:
                images_by_label[label].append({IMAGE_PATH: file_name, BBOX: bbox, IMAGE_SIZE: size})
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        # if i % 20 == 0:
        label = k
        label_images_paths = v
        for j, image_map in enumerate(label_images_paths):
            # if j % 20 == 0:
            set_input.append({IMAGE_PATH: image_map[IMAGE_PATH], LABEL: label, BBOX: image_map[BBOX],
                              IMAGE_SIZE: image_map[IMAGE_SIZE]})
    df = pd.DataFrame(set_input)
    return df


def apply_threshold(map):
    meanval = map.flatten().mean()
    new = np.where(map > meanval, 255, 0).astype(np.uint8)
    return new


def get_image(image_path):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)

    print(image_path)

    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)
    return im


def preprocess(image, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)


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


def init_get_normalize():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


def threshold_image(bbox, input, image_size):
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox = [x_min, y_min, x_max, y_max]
    x1, y1, x2, y2 = bbox

    w, h = input.shape[1], input.shape[2]
    image_size_arr = image_size.split(',')
    w1, h1 = int(image_size_arr[0]), int(image_size_arr[1])
    pw, ph = (w / w1), (h / h1)

    x1, y1, x2, y2 = int(x1 * pw), int(y1 * ph), int(x2 * pw), int(y2 * ph)
    empty = torch.zeros((w, h))
    empty[x1:x2, y1:y2] = torch.ones_like(empty)[x1:x2, y1:y2]

    return empty


ITERATION = 'iia'
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from tqdm import tqdm
    from salieny_models import *
    from voc_dataset import VOCSegmentation

    device = 'cuda'

    # Data
    models = ['densnet', 'convnext', 'resnet101']
    layer_options = [12, 8]
    model_name = models[-1]
    FEATURE_LAYER_NUMBER = layer_options[-1]
    operations = ['iia']

    segmentation_results = {}
    for operation in operations:
        segmentation_results[f'{operation}_IoU'] = 0
        segmentation_results[f'{operation}_mAP'] = 0
        segmentation_results[f'{operation}_pixAcc'] = 0
    results = []

    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize()
    CWD = os.getcwd()

    VOC_PATH = f'{CWD}/data/VOC/VOC_SEGMENTATION/'
    ds = VOCSegmentation(root=VOC_PATH, year='2012', image_set='val', download=False,
                         transform=test_img_trans,
                         target_transform=test_lbl_trans)
    results = []

    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = [], []
    for i, (img, target) in enumerate(tqdm(ds)):
        segmentation_results = {}

        heatmaps = methods.generate_heatmap(model_name, FEATURE_LAYER_NUMBER, operations, img, device=device)
        op_idx = 0
        for operation in operations:
            tgt = target
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