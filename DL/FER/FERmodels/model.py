from turtle import forward
import torch
import torch.nn as nn
import torchvision


class ssyNet(nn.Module):
    def __init__(self, model_type='densenet', num_classes=7) -> None:
        super().__init__()
        if model_type.startswith('alexnet'):
            self.net = torchvision.models.alexnet(num_classes=num_classes)
        elif model_type.startswith('vgg16'):
            self.net = torchvision.models.vgg16_bn(num_classes=num_classes)
        elif model_type.startswith('resnet18'):
            self.net = torchvision.models.resnet18(num_classes=num_classes)
        elif model_type.startswith('densenet'):
            self.net = torchvision.models.densenet121(num_classes=num_classes)
        elif model_type.startswith('googlenet'):
            self.net = torchvision.models.googlenet(num_classes=num_classes)
        elif model_type.startswith('convnext'):
            self.net = torchvision.models.convnext_tiny(num_classes=num_classes)
        elif model_type.startswith('mobilenet'):
            self.net = torchvision.models.mobilenet_v3_small(num_classes=num_classes)
        elif model_type.startswith('squeezenet'):
            self.net = torchvision.models.squeezenet1_1(num_classes=num_classes)
        else:
            raise NotImplementedError(f"Unkown model: {model_type}")

    def forward(self, x):
        out = self.net(x)
        return out
