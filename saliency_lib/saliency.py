import os
import captum.attr
import numpy
import numpy as np
import pandas as pd
import torch.multiprocessing
import torchvision.models
import transformers.models.vit.modeling_vit
from PIL import Image
from tqdm import tqdm

from saliency_lib.integrated_gradients import *
from saliency_lib.utils import *
from saliency_lib.visualization import *
import saliency.core as saliency
from cnn_saliency_map_generator import get_images
from saliency_utils import *

conv_layer_outputs = None


def call_model_function(images, call_model_args=None, expected_keys=None):
    # images = PreprocessImages(images)
    target_class_idx = call_model_args['class_idx_str']
    images = torch.tensor(images).float().cuda()
    images.requires_grad = True
    output = model(images)
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().cpu().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs


def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    images = np.array(images)
    images = images / 255
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images, dtype=torch.float32)
    print(f'shape is {images.shape}')
    images = transformer.forward(images)
    return images.requires_grad_(True)


def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title(title)


def ig(model, image_path, label, device, use_mask):
    eval_mode = model.eval()

    # Register hooks for Grad-CAM, which uses the last convolution layer
    conv_layer = model.features[-1]
    conv_layer_outputs = {}

    def conv_layer_forward(m, i, o):
        # move the RGB dimension to the last dimension
        conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()

    def conv_layer_backward(m, i, o):
        # move the RGB dimension to the last dimension
        conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1,
                                                                                       3).detach().cpu().numpy()

    conv_layer.register_forward_hook(conv_layer_forward)
    conv_layer.register_full_backward_hook(conv_layer_backward)

    images = get_images(image_path, 0)
    im = images
    integrated_gradients = saliency.IntegratedGradients()
    # Baseline is a black image.
    baseline = np.zeros(im.squeeze().shape)
    predictions = model(im.cuda())
    predictions = predictions.detach().clone().cpu().numpy()

    call_model_args = {'class_idx_str': label}
    # Compute the vanilla mask and the smoothed mask.
    vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
        im.squeeze(), call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    # image_show(vanilla_mask_grayscale, 'ig')

    heatmap = torch.tensor(vanilla_mask_grayscale)
    with torch.no_grad():
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)

        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()

        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def blurig(model, image_path, label, device, use_mask):
    eval_mode = model.eval()

    # Register hooks for Grad-CAM, which uses the last convolution layer
    conv_layer = model.features[-1]
    conv_layer_outputs = {}

    def conv_layer_forward(m, i, o):
        # move the RGB dimension to the last dimension
        conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()

    def conv_layer_backward(m, i, o):
        # move the RGB dimension to the last dimension
        conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1,
                                                                                       3).detach().cpu().numpy()

    conv_layer.register_forward_hook(conv_layer_forward)
    conv_layer.register_full_backward_hook(conv_layer_backward)

    images = get_images(image_path, 0)
    im = images
    blur_ig = saliency.BlurIG()

    # Baseline is a black image.
    baseline = np.zeros(im.squeeze().shape)
    predictions = model(im.cuda())
    predictions = predictions.detach().clone().cpu().numpy()
    call_model_args = {'class_idx_str': label}
    # Compute the vanilla mask and the smoothed mask.
    blur_ig_mask_3d = blur_ig.GetMask(
        im.squeeze(), call_model_function, call_model_args, batch_size=20)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    blur_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)
    # image_show(blur_mask_grayscale, 'blurig')
    heatmap = torch.tensor(blur_mask_grayscale)
    with torch.no_grad():
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)

        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()

        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap


def guidedig(model, image_path, label, device, use_mask):
    eval_mode = model.eval()

    # Register hooks for Grad-CAM, which uses the last convolution layer
    conv_layer = model.features[-1]
    conv_layer_outputs = {}

    def conv_layer_forward(m, i, o):
        # move the RGB dimension to the last dimension
        conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()

    def conv_layer_backward(m, i, o):
        # move the RGB dimension to the last dimension
        conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1,
                                                                                       3).detach().cpu().numpy()

    conv_layer.register_forward_hook(conv_layer_forward)
    conv_layer.register_full_backward_hook(conv_layer_backward)

    images = get_images(image_path, 0)
    im = images
    guided_ig = saliency.GuidedIG()

    # Baseline is a black image.
    baseline = np.zeros(im.squeeze().shape)
    predictions = model(im.cuda())
    predictions = predictions.detach().clone().cpu().numpy()

    call_model_args = {'class_idx_str': label}
    # Compute the vanilla mask and the smoothed mask.
    guided_ig_mask_3d = guided_ig.GetMask(
        im.squeeze(), call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    guided_mask_grayscale = saliency.VisualizeImageGrayscale(guided_ig_mask_3d)
    # image_show(guided_mask_grayscale, 'guidedig')
    heatmap = torch.tensor(guided_mask_grayscale)
    with torch.no_grad():
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)

        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = heatmap.squeeze().cpu().data.numpy()

        t = tensor2cv(images[-1])
        blended_img, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap,
                                                                                           use_mask=use_mask)
    return t, blended_img, heatmap_cv, blended_img_mask, images[-1], score, heatmap
