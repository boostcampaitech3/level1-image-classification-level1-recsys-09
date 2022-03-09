import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

num_classes = 18

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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.dropout1 = nn.Dropout(0.7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout2 = nn.Dropout(0.7)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout3 = nn.Dropout(0.7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)


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

        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)
        # return x


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU(),
        nn.Dropout(0.5))


class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        self.fc1 = nn.Linear(input, num_classes)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)    
        out = out1 + out2
        # for model in self.models:
        #     out += model(x)

        x = self.fc1(out)
        return x
        # return torch.softmax(x, dim=1)

class MyEnsemble2(nn.Module):

    def __init__(self, modelA, modelB, modelC, input):
        super(MyEnsemble2, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        self.fc1 = nn.Linear(input, num_classes)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)       
        out = out1 + out2 + out3
        # for model in self.models:
        #     out += model(x)

        x = self.fc1(out)
        return x
        # return torch.softmax(x, dim=1)
