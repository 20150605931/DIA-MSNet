import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from fightingcv_attention.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.SEAttention import SEAttention

class Tokken_ResPartNet(nn.Module):      

    '''''''''''
    Tokken_Attention - ResPartNet
    '''''''''''

    def __init__(self, part_num):
        super(Tokken_ResPartNet, self).__init__()

        # attributes
        self.part_num = part_num

        # backbone and optimize its architecture
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
        resnet.layer4[0].conv2.stride = (1,1)

        # cnn feature
        self.resnet_conv = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        

        self.TokenAttention = TokenAttention(in_channels=3, ratio=3)
        

    def forward(self, x):

        ##### Tokken_attention mechinism ######
        x = self.TokenAttention(x)
        #print("resnet input size:", x.size())

        features = self.resnet_conv(x)
        
        return features


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_channels // 16, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out * x

class TokenAttention(nn.Module):
    """
    token注意力机制
    """

    def __init__(self, in_channels, ratio=16):
        super(TokenAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # 全连接层
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=3, keepdim=True)
        max_out, _ = torch.max(x, dim=3, keepdim=True)  
        out = avg_out + max_out
        OUT = self.fc(out)
        out = self.sigmoid(out)
        return out * x


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6












