import torch.nn as nn
import torch


class PartGlobalAveragePooling(nn.Module):

    def __init__(self, part_num):
        super(PartGlobalAveragePooling, self).__init__()
        self.avgpool_c = nn.AdaptiveAvgPool2d((part_num, 1))
        self.avgpool_r = nn.AdaptiveAvgPool2d((1, part_num))
        dropout = nn.Dropout(p=0.5)
        self.maxpool_g = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_c = nn.Sequential(self.avgpool_c, dropout)
        self.pool_r = nn.Sequential(self.avgpool_r, dropout)
        self.pool_g = nn.Sequential(self.maxpool_g)

    def forward(self, features):
        features_part_c = self.pool_c(features)
        features_part_r = self.pool_r(features)
        features_global = self.pool_g(features)
        return features_part_c, features_part_r, features_global