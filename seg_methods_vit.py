import os

import numpy
import numpy as np
import pandas as pd
import torch.multiprocessing
from PIL import Image
from tqdm import tqdm
from saliency_utils import *
from salieny_models import *
from torchvision.datasets import VOCDetection

ROOT_IMAGES = "{0}/data/ILSVRC2012_img_val"

INPUT_SCORE = 'score_original_image'
IMAGE_PIXELS_COUNT = 50176
INTERPOLATION_STEPS = 50
ITERATION_STEPS = 5
LABEL = 'label'
TOP_K_PERCENTAGE = 0.25
USE_TOP_K = False
BY_MAX_CLASS = False
GRADUAL_PERTURBATION = True
IMAGE_PATH = 'image_path'
DATA_VAL_TXT = 'data/val.txt'
# DATA_VAL_TXT = f'data/pics.txt'
device = 'cuda'


def image_show(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def avg_heads_iia(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0)
    return cam


def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def transformer_attribution(vitmodel, feature_extractor, device, label, image, start_layer):
    lrp = ViT_explanation_generator.LRP(vitmodel)
    res = lrp.generate_LRP(image.unsqueeze(0), start_layer=1, method="grad", index=label).reshape(1, 1, 14, 14)
    return res.squeeze().detach()


def generate_map_tattr(model, input, index=None, start_layer=1):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    cams = []
    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        with torch.no_grad():
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams.append(cam.unsqueeze(0).detach())
    rollout = compute_rollout_attention(cams, start_layer=start_layer)
    cam = rollout[:, 0, 1:]
    return cam


def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_map_gae(model, input, index=None):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda()).detach()
    return R[0, 1:]


def get_iia_vit(vit_ours_model, feature_extractor, device, index, image):
    AGGREGATE_BEFORE = True
    imgs = get_interpolated_values(torch.zeros_like(image.squeeze().cpu()), image.squeeze().cpu(),
                                   num_steps=ITERATION_STEPS)
    attention_probs = []
    attention_grads = []
    for image in imgs:
        output = vit_ours_model(image.unsqueeze(0).cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        vit_ours_model.zero_grad()
        one_hot.backward(retain_graph=True)
        cams = []
        block_attn = []
        for blk in vit_ours_model.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attention_map()
            block_attn.append(cam)
            cam = avg_heads_iia(cam, grad)
            cams.append(cam)

        attn = torch.stack(block_attn)
        iiattn = get_interpolated_values(torch.zeros_like(attn.cpu()), attn.cpu(),
                                         num_steps=ITERATION_STEPS)
        cams = []
        grads = []
        for iatn in iiattn:
            iatn.requires_grad = True
            output = vit_ours_model(image.unsqueeze(0).cuda(), register_hook=True, attn_prob=iatn[-1].cuda())
            if index == None:
                index = np.argmax(output.cpu().data.numpy(), axis=-1)

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)
            vit_ours_model.zero_grad()
            one_hot.backward(retain_graph=True)
            with torch.no_grad():
                block_cams = []
                block_grads = []
                for blk in vit_ours_model.blocks:
                    grad = blk.attn.get_attn_gradients()
                    cam = blk.attn.get_attention_map()
                    if AGGREGATE_BEFORE:
                        cam = avg_heads_iia(cam, grad)

                    else:
                        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
                        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
                    block_cams.append(cam)
                    block_grads.append(grad)
            cams.append(torch.stack(block_cams))
            grads.append(torch.stack(block_grads))
        attention_probs.append(torch.stack(cams))
        attention_grads.append(torch.stack(grads))
    integrated_attention = torch.stack(attention_probs)
    integrated_gradients = torch.stack(attention_grads)
    final_attn = torch.mean(integrated_attention, dim=[0, 1])
    final_grads = torch.mean(integrated_gradients, dim=[0, 1])
    if AGGREGATE_BEFORE:
        final_rollout = rollout(
            attentions=final_attn.unsqueeze(1),
            head_fusion="mean",
            gradients=None
        )
    else:
        final_rollout = rollout(
            attentions=final_attn.unsqueeze(1),
            head_fusion="mean",
            gradients=final_grads.unsqueeze(1)
        )
    return final_rollout


def rollout(
        attentions,
        discard_ratio: float = 0,
        start_layer=0,
        head_fusion="max",
        gradients=None,
        return_resized: bool = True,
):
    all_layer_attentions = []
    attn = []
    if gradients is not None:
        for attn_score, grad in zip(attentions, gradients):
            score_grad = attn_score * grad
            attn.append(score_grad.clamp(min=0).detach())
    else:
        attn = attentions
    for attn_heads in attn:
        if head_fusion == "mean":
            fused_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        elif head_fusion == "max":
            fused_heads = (attn_heads.max(dim=1)[0]).detach()
        elif head_fusion == "sum":
            fused_heads = (attn_heads.sum(dim=1)[0]).detach()
        elif head_fusion == "median":
            fused_heads = (attn_heads.median(dim=1)[0]).detach()
        elif head_fusion == "min":
            fused_heads = (attn_heads.min(dim=1)[0]).detach()
        flat = fused_heads.view(fused_heads.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)

        flat[0, indices] = 0
        all_layer_attentions.append(fused_heads)

    rollout_arr = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    mask = rollout_arr[:, 0, 1:]

    if return_resized:
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width)
    return mask


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = (
        torch.eye(num_tokens)
        .expand(batch_size, num_tokens, num_tokens)
        .to(all_layer_matrices[0].device)
    )

    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [
        all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
        for i in range(len(all_layer_matrices))
    ]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


def blend_transformer_heatmap(image, x1):
    heatmap = x1.unsqueeze(0).unsqueeze(0)
    heatmap = torch.nn.functional.interpolate(heatmap, scale_factor=16, mode='bilinear')
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()
    t = tensor2cv(image, is_transformer=True)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=True)

    return im, score, heatmap_cv, blended_img_mask, heatmap, t


from vit_model import ViTmodel, ViT_LRP, ViT_explanation_generator


def generate_heatmap(model, operations, input, device):
    input = input.unsqueeze(0)
    input_predictions = model(input.to(device))
    label = torch.max(input_predictions, 1).indices[0].item()

    heatmaps = []
    for operation in operations:
        if operation == 'gae':
            gae = generate_map_gae(model, input.to(device), label).reshape(14, 14).detach()
            im, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(input, gae)
            heatmaps.append(heatmap)
        elif operation == 't-attr':
            model_tattr = ViT_LRP.vit_base_patch16_224(pretrained=True).to(device)
            # model_tattr = ViT_LRP.vit_small_patch16_224(pretrained=True).to(device)

            t_attr = transformer_attribution(model_tattr, [], device, label, input.to(device), 0).detach()
            im2, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(input, t_attr)
            heatmaps.append(heatmap)
        elif operation == 'iia':
            iia_attribution = get_iia_vit(model, [], device, label,
                                          input)
            im3, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(input, iia_attribution)
            heatmaps.append(heatmap)

    return heatmaps
