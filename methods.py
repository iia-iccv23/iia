import os

import torch.multiprocessing
from PIL import Image

from saliency_utils import *
from salieny_models import *
from torchgc.pytorch_grad_cam.ablation_cam import AblationCAM
from torchgc.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT_IMAGES = "{0}/data/ILSVRC2012_img_val"

INPUT_SCORE = 'score_original_image'
IMAGE_PIXELS_COUNT = 50176
INTERPOLATION_STEPS = 10
LABEL = 'label'
TOP_K_PERCENTAGE = 0.25
USE_TOP_K = False
BY_MAX_CLASS = False
GRADUAL_PERTURBATION = True
IMAGE_PATH = 'image_path'
DATA_VAL_TXT = 'data/val.txt'
# DATA_VAL_TXT = f'data/pics.txt'
device = 'cuda'
MODEL_NAME = 'densnet'


def calc_weight_ratio(activations_new, device, label, model, score, acts):
    ch_dim = acts.shape[1]
    yks = []
    for c in range(ch_dim):
        activations_new[0, c, :, :] = 0.

        preds_new = model(activations_new.to(device), hook=False, only_post_features=True)
        one_hot2 = torch.zeros(preds_new.shape).to(device)
        one_hot2[:, label] = 1
        mask_score = torch.sum(one_hot2 * preds_new)
        yks.append(mask_score.detach().cpu().numpy())
    # yc = torch.repeat_interleave(score, ch_dim)
    yc = np.array([score.detach().cpu().numpy()] * ch_dim)
    weight_ratio = (yc - np.array(yks)) / yc
    return weight_ratio


def get_grads_wrt_image(model, label, images_batch, device='cuda', steps=50):
    model.eval()
    model.zero_grad()

    images_batch.requires_grad = True
    preds = model(images_batch.squeeze().to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    with torch.no_grad():
        image_grads = images_batch.grad.detach()
    images_batch.requires_grad = False
    return image_grads


def backward_class_score_and_get_activation_grads(model, label, x, only_post_features=False, device='cuda'):
    model.zero_grad()
    preds = model(x.squeeze(1).to(device), hook=True, only_post_features=only_post_features)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()

    activations_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()
    return activations_gradients


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


def get_by_class_saliency_iia(model, input,
                              label,
                              operations,
                              model_name='densnet',
                              layers=[12],
                              interpolation_on_images_steps_arr=[0, 50],
                              interpolation_on_activations_steps_arr=[0, 50],
                              device='cuda',
                              use_mask=False):
    images, integrated_heatmaps = heatmap_of_layer(device, input, interpolation_on_activations_steps_arr,
                                                   interpolation_on_images_steps_arr, label, layers, model)
    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]
    t = tensor2cv(input)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask)

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap


def make_resize_norm(act_grads):
    heatmap = torch.sum(act_grads.squeeze(0), dim=0)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()
    return heatmap

def heatmap_of_layer(device, input, interpolation_on_activations_steps_arr, interpolation_on_images_steps_arr,
                     label, layers, model):
    im = torch.stack([input])
    images = get_interpolated_values(torch.zeros_like(im), im, num_steps=interpolation_on_images_steps_arr[-1])

    label = torch.tensor(label, dtype=torch.long, device=device)

    activations = model.get_activations(images.squeeze().to(device)).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=1)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(1)
    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     interpolation_on_activations_steps_arr[-1]).detach()
    grads = []
    for act in igacts:
        act.requires_grad = True
        grads.append(calc_grads(model, (act.squeeze()), device, label).detach())
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        igacts[1:] = igacts[1:] - igacts[0]
        gradsum = torch.sum(igrads.squeeze().detach() * F.relu(igacts.squeeze()),
                            dim=[0])
        integrated_heatmaps = torch.sum(gradsum, dim=[0])

    return images, integrated_heatmaps


def calc_grads(model, activations_featmap_list, device, label):
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


def lift_cam(model, input, label, device, use_mask):
    images = torch.stack([input])

    model.eval()
    model.zero_grad()
    output = model(images.to(device))

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


def get_torchgc_model_layer(device):
    if MODEL_NAME.__contains__('resnet'):
        resnet101 = torchvision.models.resnet101(pretrained=True).to(device)
        resnet101_layer = resnet101.layer4
        return resnet101, resnet101_layer
    elif MODEL_NAME.__contains__('convnext'):
        convnext = torchvision.models.convnext_base(pretrained=True).to(device)
        convnext_layer = convnext.features[-1]
        return convnext, convnext_layer

    densnet201 = torchvision.models.densenet201(pretrained=True).to(device)
    densnet201_layer = densnet201.features
    return densnet201, densnet201_layer


def ablation_cam_torchcam(model, image_path, label, device, use_mask):
    model, layer = get_torchgc_model_layer(device)
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


def run_all_operations(model, input, label, model_name='densenet', device='cpu', features_layer=8,
                       operations=['iia'],
                       use_mask=False):
    results = []
    for operation in operations:
        t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = run_by_class_grad(model, input, label,
                                                                                              model_name,
                                                                                              device,
                                                                                              features_layer,
                                                                                              operation, use_mask)
        results.append((t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap))
    return results


def run_by_class_grad(model, input, label, model_name='densenet', device='cpu', features_layer=8, operation='ours',
                      use_mask=False):
    label = torch.tensor(label, dtype=torch.long, device=device)
    t1, blended_img, heatmap_cv, blended_img_mask, t2, score, heatmap = by_class_map(model, input, label,
                                                                                     operation=operation,
                                                                                     use_mask=use_mask, device=device)

    return t1, blended_img, heatmap_cv, blended_img_mask, input, score, heatmap


def by_class_map(model, image, label, operation='ours', use_mask=False, device='cpu'):
    weight_ratio = []
    model.eval()
    model.zero_grad()

    if device != 'cpu':
        preds = model(image.unsqueeze(0).cuda(), hook=True)
    else:
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


def generate_heatmap(model_name, features_layer, operations, input, device):
    FEATURE_LAYER_NUMBER = features_layer

    if model_name.__contains__('vgg'):
        torch.nn.modules.activation.ReLU.forward = ReLU.forward
    MODEL_NAME = model_name
    model = GradModel(model_name, feature_layer=FEATURE_LAYER_NUMBER)
    model.to(device)
    model.eval()
    model.zero_grad()
    input_predictions = model(input.unsqueeze(0).to(device), hook=False).detach()
    label = torch.max(input_predictions, 1).indices[0].item()

    USE_MASK = True
    res_class_saliency = run_all_operations(model, input,
                                            label=label, model_name=model_name, device=device,
                                            features_layer=FEATURE_LAYER_NUMBER,
                                            operations=operations[1:], use_mask=USE_MASK)
    heatmaps = []
    op_idx = 0
    for method in operations:
        if method == 'iia':
            t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = \
                get_by_class_saliency_iia(
                    model, input,
                    label=label,
                    operations=['iia'],
                    model_name=model_name,
                    layers=[FEATURE_LAYER_NUMBER],
                    interpolation_on_images_steps_arr=[INTERPOLATION_STEPS],
                    interpolation_on_activations_steps_arr=[INTERPOLATION_STEPS],
                    device=device,
                    use_mask=USE_MASK)
            heatmaps.append(heatmap)
        elif method == 'lift-cam':
            t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = lift_cam(
                model, input,
                label=label,
                device=device,
                use_mask=USE_MASK)
            heatmaps.append(heatmap)
        elif method == 'ablation-cam':
            t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = ablation_cam_torchcam(
                model, input,
                label=label,
                device=device,
                use_mask=USE_MASK)
            heatmaps.append(heatmap)
        else:
            t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = res_class_saliency[op_idx]
            heatmaps.append(heatmap)
            op_idx += 1
    return heatmaps
