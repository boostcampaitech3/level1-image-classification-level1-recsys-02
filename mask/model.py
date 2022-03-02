from re import S
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet18 = models.Resnet18(pretrained=True)
        resnet18.fc = nn.Linear(in_features=512, out_features=num_classes)
        torch.nn.init.xavier_uniform_(resnet18.fc.weight)
        stdv = 1. /math.sqrt(resnet18.fc.weight.size(1))
        resnet18.fc.bias.data.uniform_(-stdv, stdv)
        self.resnet18 = resnet18

    def forward(self, x):
        return self.resnet18(x)

# class Efficientnet_b3(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
#         for n, p in model.named_parameters():
#             if '_fc' not in n:
#                 p.requires_grad = False
#         model = torch.nn.parallel.DistributedDataParallel(model)
#
#         feats_list = list(model.features)
#         new_feats_list = []
#         for feat in feats_list:
#             new_feats_list.append(feat)
#             if isinstance(feat, nn.Conv2d):
#                 new_feats_list.append(nn.Dropout(p=0.5, inplace=True))
#         model.features = nn.Sequential(*new_feats_list)
#         # torch.nn.init.xavier_uniform_(Efficientnet_b3.fc.weight)
#         # stdv = 1. /math.sqrt(Efficientnet_b3.fc.weight.size(1))
#         # Efficientnet_b3.fc.bias.data.uniform_(-stdv, stdv)
#         self.Efficientnet_b3 = model
#
#     def forward(self, x):
#         return self.Efficientnet_b3(x)
class Efficientnet_b3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        Efficientnet_b3 = EfficientNet.from_pretrained('efficientnet-b3')
        Efficientnet_b3._fc = nn.Linear(in_features=Efficientnet_b3._fc.in_features, out_features=num_classes)

        torch.nn.init.xavier_uniform_(Efficientnet_b3._fc.weight)
        stdv = 1. /math.sqrt(Efficientnet_b3._fc.weight.size(1))
        Efficientnet_b3._fc.bias.data.uniform_(-stdv, stdv)
        self.Efficientnet_b3 = Efficientnet_b3

    def forward(self, x):
        return self.Efficientnet_b3(x)

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
