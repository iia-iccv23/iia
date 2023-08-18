import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Base_Weights


def create_sequential_model_without_top(model, num_top_layers=1):
    model_children_without_top_layers = list(model.children())[:-num_top_layers]
    return nn.Sequential(*model_children_without_top_layers)


def create_resnet18_module(pretrained=True, requires_grad=False):
    model = torchvision.models.resnet18(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = requires_grad

    return create_sequential_model_without_top(model)


def create_resnet50_module(pretrained=True, requires_grad=False):
    model = torchvision.models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = requires_grad

    return create_sequential_model_without_top(model)


def create_resnet101_module(pretrained=True, requires_grad=False):
    model = torchvision.models.resnet101(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = requires_grad

    return create_sequential_model_without_top(model)


class GradModel(nn.Module):
    def __init__(self, model_name='resnet50', feature_layer=8):
        super(GradModel, self).__init__()
        self.post_features = None
        self.model_str = 'None'
        if model_name == 'resnet101':
            model = create_resnet101_module(requires_grad=True, pretrained=True)
            self.features = model[:feature_layer]
            self.post_features = model[feature_layer:-1]
            self.avgpool = model[-1:]
            self.classifier = torchvision.models.resnet101(pretrained=True).fc
        elif model_name == 'resnet50':
            model = create_resnet50_module(requires_grad=True, pretrained=True)
            self.features = model[:feature_layer]
            self.post_features = model[feature_layer:-1]
            self.avgpool = model[feature_layer:]
            self.classifier = torchvision.models.resnet50(pretrained=True).fc
        elif model_name == 'resnet18':
            model = create_resnet18_module(requires_grad=True, pretrained=True)
            self.features = model[:feature_layer]
            self.post_features = model[feature_layer:-1]
            self.avgpool = model[-1:]
            self.classifier = torchvision.models.resnet18(pretrained=True).fc
        elif model_name == 'densnet':
            model = torchvision.models.densenet201(pretrained=True)
            model.eval()
            self.features = model.features[:feature_layer]
            self.post_features = model.features[feature_layer:-1]
            self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1)
            self.classifier = model.classifier
        elif model_name == 'convnext':
            self.model_str = model_name
            model = torchvision.models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            self.features = model.features
            self.post_features = model.features[feature_layer:-1]
            self.avgpool = model.avgpool
            self.classifier = model.classifier
        elif model_name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
        elif model_name == 'vgg19':
            model = torchvision.models.vgg19(pretrained=True)
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier
        else:
            raise NotImplementedError('No such model')

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad.detach().clone()

    def activations_hook_t(self, grad):
        self.gradients_t = grad.detach().clone()

    def activations_hook_s(self, grad):
        self.gradients_s = grad.detach().clone()

    def get_activations_gradient(self):
        return self.gradients

    def get_activations_gradient_t(self):
        return self.gradients_t

    def get_activations_gradient_s(self):
        return self.gradients_s

    def get_activations(self, x):
        x = self.features(x)
        return x

    def get_post_activations(self, x):
        x = self.post_features(x)
        return x

    def get_post_activations_middle(self, x):
        x = self.post_features[:-1](x)
        return x

    def compute_representation(self, x, hook=True):
        x = self.forward_(x, hook)
        return x

    def compute_t_s_representation(self, t, s, only_post_features=False):
        t = self.forward_(t, True, only_post_features, hook_func=lambda grad: self.activations_hook_t(grad))
        s = self.forward_(s, True, only_post_features, hook_func=lambda grad: self.activations_hook_s(grad))

        return t, s

    def activations_to_features(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_(self, x, hook=True, only_post_features=False, hook_func=None, is_middle=False):
        if len(list(x.shape)) > 4:
            x = x.squeeze(1)

        if not only_post_features:
            x = self.features(x)

        if hook:
            hook_func = self.activations_hook if hook_func is None else hook_func
            x.register_hook(hook_func)

        if self.post_features is not None:
            if is_middle:
                x = self.post_features[:-1](x)
            else:
                x = self.post_features(x)


        if self.model_str == 'convnext':
            x = F.relu(x)
            x = self.avgpool(x)
        else:
            x = F.relu(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return x

    def forward(self, x, hook=True, only_post_features=False, is_middle=False):
        x = self.forward_(x, hook=hook, only_post_features=only_post_features, is_middle=is_middle)
        x = self.classifier(x)
        return x


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.av_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.av_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.drop2 = nn.Dropout(0.5)
        self.out = nn.Linear(84, 10)

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.av_pool1,
            self.conv2,
            self.bn2,
        )

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.features(x)
        return x

    def forward(self, x, hook=False):
        x = self.features(x)
        if hook:
            x.register_hook(self.activations_hook)

        x = F.relu(x)
        x = self.av_pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x
