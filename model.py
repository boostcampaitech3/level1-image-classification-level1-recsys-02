from re import S
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn.init as init

def initialize_weights(model):
    """
    Xavier uniform 분포로 모든 weight 를 초기화합니다.
    더 많은 weight 초기화 방법은 다음 문서에서 참고해주세요. https://pytorch.org/docs/stable/nn.init.html
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Linear(in_features=512, out_features=num_classes)
        torch.nn.init.xavier_uniform_(resnet18.fc.weight)
        stdv = 1. /math.sqrt(resnet18.fc.weight.size(1))
        resnet18.fc.bias.data.uniform_(-stdv, stdv)
        self.resnet18 = resnet18

    def forward(self, x):
        return self.resnet18(x)


class EfficientNet_b1(nn.Module):
    def __init__(self,num_classes):
        super(EfficientNet_b1, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')

        self.classifier_layer = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, inputs):
        x = self.model.extract_features(inputs)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        # x = self.model._dropout(x)

        x = self.classifier_layer(x)
        return x

class ResnetCls(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        resnet_cls = models.resnet18(pretrained=True)

        resnet_cls.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        initialize_weights(resnet_cls.fc)

        self.resnet_cls = resnet_cls

    def forward(self, x):
        return self.resnet_cls(x)

class EfficientNet_b3(nn.Module):
    def __init__(self,num_classes):
        super(EfficientNet_b3, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')

        self.classifier_layer = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, inputs):
        x = self.model.extract_features(inputs)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        # x = self.model._dropout(x)

        x = self.classifier_layer(x)
        return x