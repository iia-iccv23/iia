import os

import captum.attr
import numpy
import numpy as np
import pandas as pd
import torch.multiprocessing
import torchvision.models
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import auc
from scipy import ndimage
from imagenet_lables import label_map
from coco_labels import coco_label_list
from saliency_utils import *
from salieny_models import *
from saliency_lib import *
from torchvision.datasets import VOCDetection
from torchvision.datasets import CocoDetection
from torchgc.pytorch_grad_cam.fullgrad_cam import FullGrad
from torchgc.pytorch_grad_cam.layer_cam import LayerCAM
from torchgc.pytorch_grad_cam.ablation_cam import AblationCAM
from torchgc.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMAGE_SIZE = 'image_size'

BBOX = 'bbox'

ROOT_IMAGES = "{0}/data/ILSVRC2012_img_val"
IS_VOC = False
IS_COCO = False
IS_VOC_BBOX = False
IS_COCO_BBOX = False
INPUT_SCORE = 'score_original_image'
IMAGE_PIXELS_COUNT = 50176
INTERPOLATION_STEPS = 10
# INTERPOLATION_STEPS = 3
LABEL = 'label'
TOP_K_PERCENTAGE = 0.25
USE_TOP_K = False
BY_MAX_CLASS = False
GRADUAL_PERTURBATION = True
IMAGE_PATH = 'image_path'
DATA_VAL_TXT = 'data/val.txt'
# DATA_VAL_TXT = f'data/pics.txt'

device = 'cuda'


def get_grads_wrt_image(model, label, images_batch, device='cuda', steps=50):
    model.eval()
    model.zero_grad()

    images_batch.requires_grad = True
    preds = model(images_batch.to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    with torch.no_grad():
        image_grads = images_batch.grad.detach()
    images_batch.requires_grad = False
    return image_grads


def backward_class_score_and_get_activation_grads(model, label, x, only_post_features=False, device='cuda',
                                                  is_middle=False):
    model.zero_grad()

    preds = model(x.to(device), hook=True, only_post_features=only_post_features,
                  is_middle=is_middle)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()

    activations_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()

    return activations_gradients


def backward_class_score_and_get_images_grads(model, label, x, only_post_features=False, device='cuda'):
    model.zero_grad()
    preds = model(x.squeeze(1).to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()

    images_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()

    return images_gradients


def get_blurred_values(target, num_steps):
    num_steps += 1
    if num_steps <= 0: return np.array([])
    target = target.squeeze()
    tshape = len(target.shape)
    blurred_images_list = []
    for step in range(num_steps):
        sigma = int(step) / int(num_steps)
        sigma_list = [sigma, sigma, 0]

        if tshape == 4:
            sigma_list = [sigma, sigma, sigma, 0]

        blurred_image = ndimage.gaussian_filter(
            target.detach().cpu().numpy(), sigma=sigma_list, mode="grid-constant")
        blurred_images_list.append(blurred_image)

    return numpy.array(blurred_images_list)


def get_images(image_path, interpolation_on_images_steps):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)

    print(image_path)

    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)
    X = torch.stack([im])

    if interpolation_on_images_steps > 0:
        X = get_interpolated_values(torch.zeros_like(im), im, num_steps=interpolation_on_images_steps)

    return X


def get_images_blur(image_path, interpolation_on_images_steps):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)

    print(image_path)

    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)
    X = torch.stack([im])

    if interpolation_on_images_steps > 0:
        X = torch.tensor(get_blurred_values(im.detach(),
                                            interpolation_on_activations_steps_arr[-1]))

    return X


def get_by_class_saliency_iia_ablation_study_ac(image_path,
                                                label,
                                                operations,
                                                model_name='densnet',
                                                layers=[12],
                                                interpolation_on_images_steps_arr=[0, 50],
                                                interpolation_on_activations_steps_arr=[0, 50],
                                                device='cuda',
                                                use_mask=False):
    images, integrated_heatmaps = heatmap_of_layer_acts(device, image_path, interpolation_on_activations_steps_arr,
                                                        interpolation_on_images_steps_arr, label, layers, model_name)

    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def get_by_class_saliency_iia_ablation_study_im(image_path,
                                                label,
                                                operations,
                                                model_name='densnet',
                                                layers=[12],
                                                interpolation_on_images_steps_arr=[0, 50],
                                                interpolation_on_activations_steps_arr=[0, 50],
                                                device='cuda',
                                                use_mask=False):
    images, integrated_heatmaps = heatmap_of_layer_images(device, image_path, interpolation_on_activations_steps_arr,
                                                          interpolation_on_images_steps_arr, label, layers, model_name)
    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def get_by_class_saliency_iia_triple(image_path,
                                     label,
                                     operations,
                                     model_name='densnet',
                                     layers=[12],
                                     interpolation_on_images_steps_arr=[0, 50],
                                     interpolation_on_activations_steps_arr=[0, 50],
                                     device='cuda',
                                     use_mask=False):
    images, integrated_heatmaps = heatmap_of_triple_iia(device, image_path,
                                                        interpolation_on_activations_steps_arr,
                                                        interpolation_on_images_steps_arr, label, layers, model_name)
    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def get_by_class_saliency_iia(image_path,
                              label,
                              operations,
                              model_name='densnet',
                              layers=[12],
                              interpolation_on_images_steps_arr=[0, 50],
                              interpolation_on_activations_steps_arr=[0, 50],
                              device='cuda',
                              use_mask=False):
    images, integrated_heatmaps = heatmap_of_layer(device, image_path,
                                                   interpolation_on_activations_steps_arr,
                                                   interpolation_on_images_steps_arr, label, layers, model_name)

    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(last_image)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def top_k_heatmap(heatmap, percent):
    top = int(IMAGE_PIXELS_COUNT * percent)
    heatmap = heatmap.reshape(IMAGE_PIXELS_COUNT)
    ind = np.argpartition(heatmap, -top)[-top:]
    all = np.arange(IMAGE_PIXELS_COUNT)
    rem = [item for item in all if item not in ind]
    heatmap[rem] = 0.
    heatmap = heatmap.reshape(224, 224)
    return heatmap


def make_resize_norm(act_grads):
    heatmap = torch.sum(act_grads.squeeze(0), dim=0)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()
    return heatmap


def heatmap_of_triple_iia(device, image_path, interpolation_on_activations_steps_arr,
                          interpolation_on_images_steps_arr,
                          label, layers, model_name):
    images, integrated_heatmaps_triple = heatmap_of_triple_layers(device, image_path,
                                                                  interpolation_on_activations_steps_arr,
                                                                  interpolation_on_images_steps_arr, label,
                                                                  layers, model_name)
    integrated_heatmaps_triple = torch.tensor(make_resize_norm(integrated_heatmaps_triple))

    integrated_heatmaps = integrated_heatmaps_triple.unsqueeze(0).unsqueeze(0)

    return images, integrated_heatmaps


def heatmap_of_triple_layers(device, image_path, interpolation_on_activations_steps_arr,
                             interpolation_on_images_steps_arr,
                             label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=INTERPOLATION_STEPS)
    label = torch.tensor(label, dtype=torch.long, device=device)

    activations = model.get_activations(images.to(device)).cpu()
    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=0)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(0)

    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()

    grads = []
    for act in igacts:
        act.requires_grad = True
        next_activations = model.get_post_activations(act.squeeze(1).to(device)).cpu()

        next_activations = (next_activations.unsqueeze(1))

        x2, _ = torch.min(next_activations, dim=0)
        basel2 = torch.ones_like(next_activations) * x2.unsqueeze(0)

        igacts2 = get_interpolated_values(basel2.detach(), next_activations,
                                          INTERPOLATION_STEPS).detach()

        inside_grads = []
        for act2 in igacts2:
            act2.requires_grad = True
            inside_grads.append(
                calc_grads_model_post(model, act2, device, label, post_feat=True, is_middle=True).detach() * F.relu(
                    act2))
            act2 = act2.detach()
            act2.requires_grad = False

        grads.append(torch.stack(inside_grads).detach())

        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
    mul_grad_act = F.relu(igrads.squeeze().detach())
    integrated_heatmaps = torch.sum(mul_grad_act, dim=[0, 1, 2])

    return images, integrated_heatmaps


def heatmap_of_layers_layer_no_interpolation(device, image_path, interpolation_on_activations_steps_arr,
                                             interpolation_on_images_steps_arr,
                                             label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    images = get_images(image_path, interpolation_on_images_steps=0)

    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()
    activations_featmap_list = (activations.unsqueeze(1))
    gradients = calc_grads_model(model, activations_featmap_list, device, label).detach()
    gradients_squeeze = gradients.detach().squeeze()
    act_grads = F.relu(activations.squeeze()) * F.relu(gradients_squeeze) ** 2
    integrated_heatmaps = torch.sum(act_grads.squeeze(0), dim=0).unsqueeze(0).unsqueeze(0)
    return images, integrated_heatmaps


def heatmap_of_layer(device, image_path, interpolation_on_activations_steps_arr,
                     interpolation_on_images_steps_arr,
                     label, layers, model_name):
    images = get_images(image_path, interpolation_on_images_steps=INTERPOLATION_STEPS)

    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(images.to(device)).cpu()
    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=1)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(1)
    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     interpolation_on_activations_steps_arr[-1]).detach()
    grads = []
    for act in igacts:
        act.requires_grad = True
        grads.append(calc_grads_model(model, (act.squeeze()), device, label).detach())
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        igacts[1:] = igacts[1:] - igacts[0]
        gradsum = torch.sum(igrads.squeeze().detach() * F.softplus(igacts.squeeze()),
                            dim=[0])
        gradsum = torch.sum(igrads.squeeze().detach() * F.relu(igacts.squeeze()),
                            dim=[0])
        integrated_heatmaps = torch.sum(gradsum, dim=[0])

    return images, integrated_heatmaps


def heatmap_of_layer_images(device, image_path, interpolation_on_activations_steps_arr,
                            interpolation_on_images_steps_arr,
                            label, layers, model_name):
    images = get_images(image_path, interpolation_on_images_steps=interpolation_on_images_steps_arr[-1])

    label = torch.tensor(label, dtype=torch.long, device=device)

    activations = model.get_activations(images.to(device)).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=1)
    grads = calc_grads(activations_featmap_list, device, label).detach()

    with torch.no_grad():
        integrated_heatmaps = torch.sum((F.relu(grads) * F.relu(activations_featmap_list)), dim=[0, 1])

    return images, integrated_heatmaps


def heatmap_of_layer_acts(device, image_path, interpolation_on_activations_steps_arr, interpolation_on_images_steps_arr,
                          label, layers, model_name):
    images = get_images(image_path, 0)

    label = torch.tensor(label, dtype=torch.long, device=device)

    activations = model.get_activations(images.to(device)).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=1)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(1)
    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     interpolation_on_activations_steps_arr[-1]).detach()
    grads = calc_grads(activations_featmap_list, device, label).detach()

    with torch.no_grad():
        integrated_heatmaps = torch.sum((F.relu(grads.unsqueeze(0)) * F.relu(igacts)), dim=[0, 1])

    return images, integrated_heatmaps


def calc_grads_model_post(model, activations_featmap_list, device, label, post_feat, is_middle=False):
    activations_gradients = backward_class_score_and_get_activation_grads(model, label, activations_featmap_list,
                                                                          only_post_features=post_feat,
                                                                          device=device,
                                                                          is_middle=is_middle)
    return activations_gradients


def calc_grads_model(model, activations_featmap_list, device, label):
    activations_gradients = backward_class_score_and_get_activation_grads(model, label, activations_featmap_list,
                                                                          only_post_features=True,
                                                                          device=device)
    return activations_gradients


def calc_grads(activations_featmap_list, device, label):
    activations_gradients = backward_class_score_and_get_activation_grads(model, label, activations_featmap_list,
                                                                          only_post_features=True,
                                                                          device=device)
    return activations_gradients


# LIFT-CAM
from captum.attr import DeepLift


class Model_Part(nn.Module):
    def __init__(self, model):
        super(Model_Part, self).__init__()
        self.model_type = None
        if model.model_str == 'convnext':
            self.avg_pool = model.avgpool
            self.classifier = model.classifier[-1]
        else:
            self.avg_pool = model.avgpool
            self.classifier = model.classifier

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def lift_cam(model, image_path, label, device, use_mask):
    images = get_images(image_path, 0)

    model.eval()
    model.zero_grad()
    output = model(images.to(device), hook=True)

    class_id = label
    if class_id is None:
        class_id = torch.argmax(output, dim=1)

    act_map = model.get_activations(images.to(device))

    model_part = Model_Part(model)
    model_part.eval()
    dl = DeepLift(model_part)
    ref_map = torch.zeros_like(act_map).to(device)
    dl_contributions = dl.attribute(act_map, ref_map, target=class_id, return_convergence_delta=False).detach()

    scores_temp = torch.sum(dl_contributions, (2, 3), keepdim=False).detach()
    scores = torch.squeeze(scores_temp, 0)
    scores = scores.cpu()

    vis_ex_map = (scores[None, :, None, None] * act_map.cpu()).sum(dim=1, keepdim=True)
    vis_ex_map = F.relu(vis_ex_map).float()

    with torch.no_grad():
        heatmap = vis_ex_map
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()
        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def ig(model, image_path, label, device, use_mask):
    return saliency.ig(model, image_path, label, device, use_mask)


def blurig(model, image_path, label, device, use_mask):
    return saliency.blurig(model, image_path, label, device, use_mask)


def guidedig(model, image_path, label, device, use_mask):
    return saliency.guidedig(model, image_path, label, device, use_mask)


def ig_captum(model, image_path, label, device, use_mask):
    images = get_images(image_path, 0)

    model.eval()
    model.zero_grad()
    class_id = label

    integrated_grads = captum.attr.IntegratedGradients(model)
    baseline = torch.zeros_like(images).to(device)
    attr = integrated_grads.attribute(images.to(device), baseline, class_id)

    with torch.no_grad():
        heatmap = torch.mean(attr, dim=1, keepdim=True)
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()
        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def get_torchgc_model_layer(network_name, device):
    if network_name.__contains__('resnet'):
        resnet101 = torchvision.models.resnet101(pretrained=True).to(device)
        resnet101_layer = resnet101.layer4
        return resnet101, resnet101_layer
    elif network_name.__contains__('convnext'):
        convnext = torchvision.models.convnext_base(pretrained=True).to(device)
        convnext_layer = convnext.features[-1]
        return convnext, convnext_layer

    densnet201 = torchvision.models.densenet201(pretrained=True).to(device)
    densnet201_layer = densnet201.features
    return densnet201, densnet201_layer


def ablation_cam_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = AblationCAM(model.to(device), layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def fullgrad_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = FullGrad(model, layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def layercam_torchcam(network_name, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(network_name, device)
    cam_extractor = LayerCAM(model.to(device), layer)
    images = get_images(image_path, 0)
    targets = [ClassifierOutputTarget(label)]
    hm = cam_extractor(images.to(device), targets)

    heatmap = torch.tensor(hm).unsqueeze(0)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().detach().cpu().data.numpy()

    t = tensor2cv(images[-1])
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                       use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def run_all_operations(model, image_path, label, model_name='densenet', device='cpu', features_layer=8,
                       operations=['iia'],
                       use_mask=False):
    results = []
    for operation in operations:
        t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = run_by_class_grad(model, image_path, label,
                                                                                              model_name,
                                                                                              device,
                                                                                              features_layer,
                                                                                              operation, use_mask)
        results.append((t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap))
    return results


def run_by_class_grad(model, image_path, label, model_name='densenet', device='cpu', features_layer=8, operation='ours',
                      use_mask=False):
    CWD = os.getcwd()
    root = ROOT_IMAGES.format(CWD)
    print(image_path)
    img = Image.open(root + '/' + image_path).convert('RGB')
    im = preprocess(img)

    label = torch.tensor(label, dtype=torch.long, device=device)
    t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = by_class_map(model, im, label,
                                                                                     operation=operation,
                                                                                     use_mask=use_mask)

    return t1, blended_img, heatmap_cv, blended_img_mask, im, score, heatmap


def by_class_map(model, image, label, operation='ours', use_mask=False):
    weight_ratio = []
    model.eval()
    model.zero_grad()
    preds = model(image.unsqueeze(0).to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)

    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    preds.to(device)
    one_hot.to(device)
    gradients = model.get_activations_gradient()
    heatmap = grad2heatmaps(model, image.unsqueeze(0).to(device), gradients, activations=None, operation=operation,
                            score=score, do_nrm_rsz=True,
                            weight_ratio=weight_ratio)

    t = tensor2cv(image)
    blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, blended_img, heatmap_cv, blended_img_mask, t, score, heatmap


def image_show(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def calc_score_original(input):
    global label
    preds_original_image = model(input.to(device), hook=False).detach()
    one_hot = torch.zeros(preds_original_image.shape).to(device)
    one_hot[:, label] = 1

    score_original_image = torch.sum(one_hot * preds_original_image, dim=1).detach()
    return score_original_image


def calc_score_masked(masked_image):
    global label
    preds_masked_image = model(masked_image.unsqueeze(0).to(device), hook=False).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score_masked_image = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score_masked_image


def calc_img_score(img):
    global label
    preds_masked_image = model(img.unsqueeze(0).to(device), hook=False).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score


def calc_blended_image_score(heatmap):
    global label
    img_cv = tensor2cv(input.squeeze())
    heatmap_cv = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    masked_image = np.uint8((np.repeat(heatmap_cv.reshape(224, 224, 1), 3, axis=2) * img_cv))
    img = preprocess(Image.fromarray(masked_image))
    preds_masked_image = model(img.unsqueeze(0).to(device), hook=False).detach()
    one_hot = torch.zeros(preds_masked_image.shape).to(device)
    one_hot[:, label] = 1
    score = torch.sum(one_hot * preds_masked_image, dim=1).detach()
    return score


def coco_calculate_localization():
    global ROOT_IMAGES
    ROOT_IMAGES = "{0}/data/COCO/val2014/"

    images_by_label = {}
    with open(f'data/coco_bbox.txt') as f:
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
        label = k
        label_images_paths = v
        for j, image_map in enumerate(label_images_paths):
            set_input.append({IMAGE_PATH: image_map[IMAGE_PATH], LABEL: label, BBOX: image_map[BBOX],
                              IMAGE_SIZE: image_map[IMAGE_SIZE]})
    df = pd.DataFrame(set_input)
    return df


def create_set_from_coco():
    images_by_label = {}
    with open(f'data/coco.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)
    return df


def voc_calculate_localization():
    global ROOT_IMAGES
    ROOT_IMAGES = "{0}/data/VOC/VOCdevkit/VOC2007/JPEGImages"

    CWD = os.getcwd()
    ROOT_IMAGES = "{0}/data/VOC/VOCdevkit/VOC2007/JPEGImages".format(CWD)

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
        label = k
        label_images_paths = v
        for j, image_map in enumerate(label_images_paths):
            set_input.append({IMAGE_PATH: image_map[IMAGE_PATH], LABEL: label, BBOX: image_map[BBOX],
                              IMAGE_SIZE: image_map[IMAGE_SIZE]})
    df = pd.DataFrame(set_input)
    return df


def create_set_from_voc():
    global ROOT_IMAGES
    CWD = os.getcwd()
    ROOT_IMAGES = "{0}/data/VOC/VOCdevkit/VOC2007/JPEGImages".format(CWD)

    images_by_label = {}
    with open(f'data/voc.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def create_set_from_txt():
    images_by_label = {}
    with open(f'data/pics.txt') as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def create_set_from_txt_prod():
    images_by_label = {}
    with open(DATA_VAL_TXT) as f:
        lines = f.readlines()
        for line in lines:
            file_name, label = line.split()
            label = int(label)
            if label not in images_by_label:
                images_by_label[label] = [file_name]
            else:
                images_by_label[label].append(file_name)
    set_input = []
    for i, (k, v) in tqdm(enumerate(images_by_label.items())):
        label = k
        image_paths = v
        for j, image_path in enumerate(image_paths):
            set_input.append({IMAGE_PATH: image_path, LABEL: label})
    df = pd.DataFrame(set_input)

    return df


def handle_image_saving(blended_im, blended_img_mask, label, operation, save_image=False, save_mask=False):
    im_to_save = blended_im
    if save_mask:
        im_to_save = blended_img_mask

    if save_image:
        title = f'method: {operation}, label: {int(label)}'
        img_dict.append({"image": im_to_save, "title": title})


def write_imgs_iterate(img_name):
    num_rows = 3
    num_col = 4
    f = plt.figure(figsize=(30, 20))
    plt.subplot(num_rows, num_col, 1)
    plt.imshow(t)
    plt.title('ground truth')
    plt.axis('off')

    i = 2
    for item in img_dict:
        plt.subplot(num_rows, num_col, i)
        plt.imshow(item["image"])
        plt.title(item["title"])
        plt.axis('off')
        i += 1

    if img_name is not None:
        plt.savefig(img_name)

    plt.clf()
    plt.close('all')


class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=False)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def write_heatmap(model_name, image_path, operation, heatmap_cv):
    CWD = os.getcwd()
    np.save("{0}/data/heatmaps/{1}_{2}_{3}".format(CWD, image_path[:-5], operation, model_name), heatmap_cv)


def write_mask(model_name, image_path, operation, masked_image):
    CWD = os.getcwd()
    np.save("{0}/data/masks/{1}_{2}_{3}".format(CWD, image_path[:-5], operation, model_name), masked_image)


ITERATION = 'iia'
models = ['densnet', 'convnext', 'resnet101']
layer_options = [12, 8]

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    image = None

    model_name = models[2]
    FEATURE_LAYER_NUMBER = layer_options[1]

    PREV_LAYER = FEATURE_LAYER_NUMBER - 1
    interpolation_on_activations_steps_arr = [INTERPOLATION_STEPS]
    interpolation_on_images_steps_arr = [INTERPOLATION_STEPS]
    num_layers_options = [1]

    USE_MASK = True
    save_img = True
    save_heatmaps_masks = False
    operations = ['iia', 'fullgrad', 'ablation-cam', 'lift-cam', 'layercam', 'ig', 'blurig', 'guidedig',
                  'gradcam', 'gradcampp', 'x-gradcam']
    operations = ['iia', 'iia-ablation-im', 'iia-ablation-ac']
    operations = ['iia']

    torch.nn.modules.activation.ReLU.forward = ReLU.forward
    if model_name.__contains__('vgg'):
        torch.nn.modules.activation.ReLU.forward = ReLU.forward

    model = GradModel(model_name, feature_layer=FEATURE_LAYER_NUMBER)
    model.to(device)
    model.eval()
    model.zero_grad()

    df = create_set_from_txt()
    print(len(df))
    df_len = len(df)

    for index, row in tqdm(df.iterrows()):

        image_path = row[IMAGE_PATH]
        label = row[LABEL]
        target_label = label
        input = get_images(image_path, 0)
        input_predictions = model(input.to(device), hook=False).detach()
        predicted_label = torch.max(input_predictions, 1).indices[0].item()

        if BY_MAX_CLASS:
            label = predicted_label

        res_class_saliency = run_all_operations(model, image_path=image_path,
                                                label=label, model_name=model_name, device=device,
                                                features_layer=FEATURE_LAYER_NUMBER,
                                                operations=operations[1:], use_mask=USE_MASK)

        operation_index = 0
        score_original_image = 0
        img_dict = []
        for operation in operations:
            if operation == 'iia':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                    get_by_class_saliency_iia(image_path=image_path,
                                              label=label,
                                              operations=['iia'],
                                              model_name=model_name,
                                              layers=[FEATURE_LAYER_NUMBER],
                                              interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                              interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                              device=device,
                                              use_mask=USE_MASK)
            elif operation == 'iia-ablation-ac':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                    get_by_class_saliency_iia_ablation_study_ac(
                        image_path=image_path,
                        label=label,
                        operations=['iia'],
                        model_name=model_name,
                        layers=[FEATURE_LAYER_NUMBER],
                        interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                        interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                        device=device,
                        use_mask=USE_MASK)
            elif operation == 'iia-ablation-im':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                    get_by_class_saliency_iia_ablation_study_im(
                        image_path=image_path,
                        label=label,
                        operations=['iia'],
                        model_name=model_name,
                        layers=[FEATURE_LAYER_NUMBER],
                        interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                        interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                        device=device,
                        use_mask=USE_MASK)
            elif operation == 'iia-triple':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                    get_by_class_saliency_iia_triple(image_path=image_path,
                                                     label=label,
                                                     operations=['iia'],
                                                     model_name=model_name,
                                                     layers=[PREV_LAYER],
                                                     interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
                                                     interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
                                                     device=device,
                                                     use_mask=USE_MASK)
            elif operation == 'lift-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = lift_cam(
                    model,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'ablation-cam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ablation_cam_torchcam(
                    model_name,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'ig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ig(
                    model,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'blurig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = blurig(
                    model,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'guidedig':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = guidedig(
                    model,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'layercam':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = layercam_torchcam(
                    model_name,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            elif operation == 'fullgrad':
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = fullgrad_torchcam(
                    model_name,
                    image_path=image_path,
                    label=label,
                    device=device,
                    use_mask=USE_MASK)
            else:
                t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = res_class_saliency[
                    operation_index]
                operation_index = operation_index + 1

            handle_image_saving(blended_im, blended_img_mask, label, operation, save_image=True, save_mask=False)
            if save_heatmaps_masks:
                write_heatmap(model_name, image_path, operation, heatmap_cv)
                write_mask(model_name, image_path, operation, blended_img_mask)

        if save_img:
            string_lbl = label_map.get(int(label))
            write_imgs_iterate(f'qualitive_results/{model_name}_{string_lbl}_{image_path}')
        score_original_image = calc_score_original(input)
        torch.cuda.empty_cache()
