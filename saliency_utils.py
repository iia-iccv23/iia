import torch
import torchvision.transforms as transforms
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F


def download(url, fname):
    response = requests.get(url)
    with open(fname, "wb") as f:
        f.write(response.content)


def preprocess(image, size=224, is_transformer=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_transformer:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(image)


def tensor2cv(inp, is_transformer=False):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if is_transformer:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = np.uint8(255 * inp)
    return inp


def get_similarity_func(metric):
    if metric == 'cos':
        return torch.nn.CosineSimilarity(dim=0)
    if metric == 'dot':
        return torch.dot


def blend_image_and_heatmap(img_cv, heatmap, use_mask=False):
    heatmap -= np.min(heatmap)

    if heatmap.max() != torch.tensor(0.):
        heatmap /= heatmap.max()

    blended_img_mask = None

    if use_mask:
        score = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_cv = np.uint8(score)
        blended_img_mask = np.uint8((np.repeat(score.reshape(224, 224, 1), 3, axis=2) * img_cv))

    heatmap = np.max(heatmap) - heatmap
    if np.max(heatmap) < 255.:
        heatmap *= 255

    score = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_cv = np.uint8(score)
    heatmap_cv = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)

    blended_img = heatmap_cv * 0.9 + img_cv
    blended_img = cv2.normalize(blended_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    blended_img[blended_img < 0] = 0

    return blended_img, score, heatmap_cv, blended_img_mask, img_cv


def plot_blended_image(X, blended_im, title=None, save_name=None):
    img1 = tensor2cv(X[0])
    img2 = tensor2cv(X[1])

    f = plt.figure(figsize=(12, 12))
    f.suptitle(title, fontsize=36)
    plt.title(title)
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    plt.subplot(221)
    plt.imshow(img1)
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img2)
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(blended_im[0])
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(blended_im[1])
    plt.axis('off')

    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


def plot_all_sim_images(t, s, im1, im2, im3, im4, im5, save_name=None):
    num_rows = 6
    if im5 is None:
        num_rows -= 1
    if im4 is None:
        num_rows -= 1
    if im3 is None:
        num_rows -= 1
    if im2 is None:
        num_rows -= 1

    f = plt.figure(figsize=(5, 12))
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    plt.subplot(num_rows, 2, 1)
    plt.imshow(t)
    plt.axis('off')
    plt.subplot(num_rows, 2, 2)
    plt.imshow(s)
    plt.axis('off')

    plt.subplot(num_rows, 2, 3)
    plt.imshow(im1[0])
    plt.axis('off')
    plt.subplot(num_rows, 2, 4)
    plt.imshow(im1[1])
    plt.axis('off')

    if im2 is not None:
        plt.subplot(num_rows, 2, 5)
        plt.imshow(im2[0])
        plt.axis('off')
        plt.subplot(num_rows, 2, 6)
        plt.imshow(im2[1])
        plt.axis('off')

    if im3 is not None:
        plt.subplot(num_rows, 2, 7)
        plt.imshow(im3[0])
        plt.axis('off')
        plt.subplot(num_rows, 2, 8)
        plt.imshow(im3[1])
        plt.axis('off')

    if im4 is not None:
        plt.subplot(num_rows, 2, 9)
        plt.imshow(im4[0])
        plt.axis('off')
        plt.subplot(num_rows, 2, 10)
        plt.imshow(im4[1])
        plt.axis('off')

    if im5 is not None:
        plt.subplot(num_rows, 2, 11)
        plt.imshow(im5[0])
        plt.axis('off')
        plt.subplot(num_rows, 2, 12)
        plt.imshow(im5[1])
        plt.axis('off')

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


def plot_all_by_class_images(t, im1, im2, im3, im4, im5, save_name=None):
    num_rows = 6
    if im5 is None:
        num_rows -= 1
    if im4 is None:
        num_rows -= 1
    if im3 is None:
        num_rows -= 1
    if im2 is None:
        num_rows -= 1

    f = plt.figure(figsize=(2, 6))
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    plt.subplot(num_rows, 1, 1)
    plt.imshow(t)
    plt.axis('off')

    plt.subplot(num_rows, 1, 2)
    plt.imshow(im1)
    plt.axis('off')

    if im2 is not None:
        plt.subplot(num_rows, 1, 3)
        plt.imshow(im2)
        plt.axis('off')

    if im3 is not None:
        plt.subplot(num_rows, 1, 4)
        plt.imshow(im3)
        plt.axis('off')

    if im4 is not None:
        plt.subplot(num_rows, 1, 5)
        plt.imshow(im4)
        plt.axis('off')

    if im5 is not None:
        plt.subplot(num_rows, 1, 6)
        plt.imshow(im5)
        plt.axis('off')

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


def get_interpolated_values(baseline, target, num_steps):
    """this function returns a list of all the images interpolation steps."""
    if num_steps <= 0: return np.array([])
    if num_steps == 1: return np.array([baseline, target])

    delta = target - baseline

    if baseline.ndim == 3:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis,
                 np.newaxis]  # newaxis = unsqueeze
    elif baseline.ndim == 4:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    elif baseline.ndim == 5:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                 np.newaxis]
    elif baseline.ndim == 6:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                 np.newaxis, np.newaxis]

    shape = (num_steps + 1,) + delta.shape
    deltas = scales * np.broadcast_to(delta.detach().numpy(), shape)
    interpolated_activations = baseline + deltas

    return interpolated_activations


def grad2heatmaps(model, X, gradients, activations=None, operation='iia', score=None, do_nrm_rsz=True, weight_ratio=[]):
    if activations is None:
        activations = model.get_activations(
            X).detach()

    if operation == 'iia':
        act_grads = F.relu(activations) * F.relu(gradients.detach()) ** 2
        heatmap = torch.sum(act_grads.squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif operation == 'gradcam':
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)
        heatmap = torch.mean(activations * pooled_gradients, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
    elif operation == 'x-gradcam':
        sum_activations = np.sum(activations.detach().cpu().numpy(), axis=(2, 3))
        eps = 1e-7
        weights = gradients.detach().cpu().numpy() * activations.detach().cpu().numpy() / \
                  (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        weights_tensor = torch.tensor(weights).unsqueeze(2).unsqueeze(3)
        heatmap = F.relu(
            torch.sum(gradients.detach().cpu() * weights_tensor.detach().cpu() * activations.detach().cpu(), dim=1,
                      keepdim=True))
    elif operation == 'activations':
        heatmap = torch.sum(F.relu(activations).squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif operation == 'gradients':
        heatmap = torch.sum((F.relu(gradients.detach())).squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif operation == 'neg_gradients':
        heatmap = torch.sum((F.relu(-1 * gradients.detach())).squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif operation == 'gradcampp':
        gradients = gradients.detach()
        activations = activations.detach()
        score = score.detach()
        square_grad = gradients.pow(2)
        denominator = 2 * square_grad + activations.mul(gradients.pow(3)).sum(dim=[2, 3], keepdim=True)
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        alpha = torch.div(square_grad, denominator + 1e-6)
        pos_grads = F.relu(score.exp() * gradients).detach()
        weights = torch.sum(alpha * pos_grads, dim=[2, 3], keepdim=True).detach()
        heatmap = torch.sum(activations * weights, dim=1, keepdim=True).detach()

    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()

    return heatmap


def nrm_rsz(act_grads, operation):
    if operation == 'iia':
        heatmap = torch.sum(act_grads.squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif operation == 'gradcam':
        heatmap = torch.sum(act_grads, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
    elif operation == 'gradcampp':
        heatmap = torch.sum(act_grads, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
    elif operation == 'gradcam_wo_relu':
        heatmap = torch.sum(act_grads, dim=1, keepdim=True)
    elif operation == 'activations':
        heatmap = torch.sum(act_grads.squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif operation == 'gradients':
        heatmap = torch.sum(act_grads.squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif operation == 'neg_gradients':
        heatmap = torch.sum(act_grads.squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif operation == 'gradients_wo_relu':
        heatmap = torch.sum(act_grads.squeeze(0), dim=0)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap -= heatmap.min()
    if heatmap.max() != torch.tensor(0):
        heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()

    return heatmap
