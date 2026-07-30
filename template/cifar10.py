"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_loader
import os
import random
from datetime import datetime
import multiprocessing
from utils import StatusUpdateTool
import time



############# Inception ##########

class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )






############# SE-Inception (refactored) ##########

class InceptionSE_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        red_3x3, out_3x3,
        red_5x5, out_5x5,
        out_1x1pool,
        reduction_ratio
    ):
        super(InceptionSE_block, self).__init__()

        # Inception branches
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

        # SE block (re-use)
        total_out_channels = out_1x1 + out_3x3 + out_5x5 + out_1x1pool
        self.se_block = SEBlock(in_channels=total_out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)

        # Apply SE
        out = self.se_block(out)
        return out


############# CBAM-Inception (refactored) ##########

class CBAMInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        red_3x3, out_3x3,
        red_5x5, out_5x5,
        out_1x1pool,
        reduction_ratio
    ):
        super(CBAMInceptionBlock, self).__init__()

        # Inception branches
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

        # CBAM block (re-use)
        total_out_channels = out_1x1 + out_3x3 + out_5x5 + out_1x1pool
        self.cbam_block = CBAM(in_channels=total_out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)

        # Apply CBAM
        out = self.cbam_block(out)
        return out










############# SE-Inception_old ##########

class InceptionSE_block_old(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio
    ):
        super(InceptionSE_block, self).__init__()

        # Inception branches
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

        # Squeeze-and-Excitation block
        total_out_channels = out_1x1 + out_3x3 + out_5x5 + out_1x1pool
        self.fc1 = nn.Linear(total_out_channels, total_out_channels // reduction_ratio, bias=False)
        self.fc2 = nn.Linear(total_out_channels // reduction_ratio, total_out_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Inception branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        # Concatenate all branch outputs
        out = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], 1)

        # Squeeze-and-Excitation
        se = F.adaptive_avg_pool2d(out, (1, 1))
        se = se.view(se.size(0), -1)  # Flatten the tensor
        se = self.fc1(se)
        se = F.relu(se, inplace=True)
        se = self.fc2(se)
        se = self.sigmoid(se).view(se.size(0), se.size(1), 1, 1)

        # Scale the output
        out = out * se
        return out



############# CBAM-Inception old##########

class CBAMInceptionBlock_old(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio
    ):
        super(CBAMInceptionBlock, self).__init__()

        # Inception branches
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

        # CBAM: Channel and Spatial Attention blocks
        total_out_channels = out_1x1 + out_3x3 + out_5x5 + out_1x1pool
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(total_out_channels, total_out_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_out_channels // reduction_ratio, total_out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Inception branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        # Concatenate all branch outputs
        out = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], 1)

        # Channel Attention
        avg_out = self.channel_attention[0](out)
        max_out = self.channel_attention[1](out)
        channel_out = avg_out + max_out
        channel_out = self.channel_attention[2:](channel_out)
        out = out * channel_out

        # Spatial Attention
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_attention(spatial_out)
        out = out * spatial_out

        return out




############# Coordinate Attention ##########

class h_swish(nn.Module):
    #h swish
    def __init__(self):
        super().__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class h_sigmoid(nn.Module):
    #h sigmoid
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class CoordinateAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(CoordinateAttentionBlock, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Dikey havuzlama
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Yatay havuzlama

        mip = max(8, in_channels // reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Coordinate attention: Dikey ve Yatay havuzlama
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # Yatay havuzlama

        # Birleştirme ve sıkma
        x_cat = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(x_cat)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Dönüşümler
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w.permute(0, 1, 3, 2))

        # Dikkat haritalarını birleştirme
        attention = self.sigmoid(x_h + x_w)
        
        return identity * attention



class CAInceptionBlock(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio
    ):
        super(CAInceptionBlock, self).__init__()

        # Inception branches
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

        # Coordinate Attention block
        total_out_channels = out_1x1 + out_3x3 + out_5x5 + out_1x1pool
        self.coord_attention = CoordinateAttentionBlock(in_channels=total_out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        # Inception branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        # Concatenate all branch outputs
        out = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], 1)

        # Apply Coordinate Attention
        out = self.coord_attention(out)

        return out


###########################################################33


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class ResNetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetBasic(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBasic, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetUnit(nn.Module):
    def __init__(self, amount, in_channel, out_channel):
        super(ResNetUnit, self).__init__()
        self.in_planes = in_channel
        #below function tries to generate random seed number
        seed_value = (lambda amount, in_channel, out_channel: int(abs((amount * (in_channel + 3) - (out_channel ** 2) + (amount // (out_channel + 1)) + (in_channel * out_channel))% 13)))(amount, in_channel, out_channel)

        random.seed(seed_value)
        u_ = random.random()
        if u_ < 0.5:
            self.layer = self._make_layer(ResNetBottleneck, out_channel, amount, stride=1)
        else:
            self.layer = self._make_layer(ResNetBasic, out_channel, amount, stride=1)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.layer(x)
        return out

class DenseNetBottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(DenseNetBottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class DenseNetUnit(nn.Module):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel):
        super(DenseNetUnit, self).__init__()
        self.out_channel = out_channel
        if in_channel > max_input_channel:
            self.need_conv = True
            self.bn = nn.BatchNorm2d(in_channel)
            self.conv = nn.Conv2d(in_channel, max_input_channel, kernel_size=1, bias=False)
            in_channel = max_input_channel

        self.layer = self._make_dense(in_channel, k, amount)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for _ in range(int(nDenseBlocks)):
            layers.append(DenseNetBottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
    def forward(self, x):
        out = x
        if hasattr(self, 'need_conv'):
            out = self.conv(F.relu(self.bn(out)))
        out = self.layer(out)
        assert(out.size()[1] == self.out_channel)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Squeeze operation: Global Average Pooling
        y = self.global_pool(x).view(batch_size, channels)

        # Excitation operation: Fully Connected Layers
        y = self.fc(y).view(batch_size, channels, 1, 1)

        # Scale the input features with the calculated weights
        return x * y.expand_as(x)


class SEResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, amount, reduction_ratio):
        super(SEResNetBlock, self).__init__()
        self.resnet_unit = ResNetUnit(amount, in_channel, out_channel)
        self.se_block = SEBlock(in_channels=out_channel, reduction_ratio=reduction_ratio)

    def forward(self, x):
        res_out = self.resnet_unit(x)

        # Use SEBlock's forward method to apply SE operation
        se_out = self.se_block(res_out)
        
        return se_out



class SEResNetBlock_old(nn.Module):
    def __init__(self, in_channel, out_channel, amount, reduction_ratio):
        super(SEResNetBlock, self).__init__()
        self.resnet_unit = ResNetUnit(amount, in_channel, out_channel)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // reduction_ratio, out_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_out = self.resnet_unit(x)
        se_out = self.se_block(res_out)
        return res_out * se_out


class CBAMResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, amount, reduction_ratio):
        super(CBAMResNetBlock, self).__init__()
        self.resnet_unit = ResNetUnit(amount, in_channel, out_channel)

        # Use CBAM block for channel and spatial attention
        self.cbam_block = CBAM(in_channels=out_channel, reduction_ratio=reduction_ratio)

    def forward(self, x):
        res_out = self.resnet_unit(x)

        # Apply CBAM block for channel and spatial attention
        cbam_out = self.cbam_block(res_out)

        return cbam_out


# CBAM components
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x





class CBAMResNetBlock_old(nn.Module):
    def __init__(self, in_channel, out_channel, amount, reduction_ratio):
        super(CBAMResNetBlock, self).__init__()
        self.resnet_unit = ResNetUnit(amount, in_channel, out_channel)

        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(out_channel, out_channel // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // reduction_ratio, out_channel, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        res_out = self.resnet_unit(x)

        # Channel Attention
        avg_out = torch.mean(res_out, dim=1, keepdim=True)
        max_out, _ = torch.max(res_out, dim=1, keepdim=True)
        ca_out = self.channel_attention(avg_out + max_out)
        res_out = res_out * ca_out

        # Spatial Attention
        avg_out = torch.mean(res_out, dim=1, keepdim=True)
        max_out, _ = torch.max(res_out, dim=1, keepdim=True)
        sa_out = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        res_out = res_out * sa_out

        return res_out






class CAResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, amount, reduction_ratio):
        super(CAResNetBlock, self).__init__()
        self.resnet_unit = ResNetUnit(amount, in_channel, out_channel)
        self.coord_attention = CoordinateAttentionBlock(out_channel, reduction_ratio=reduction_ratio)

    def forward(self, x):
        res_out = self.resnet_unit(x)
        ca_out = self.coord_attention(res_out)
        return ca_out  #res_out * ca_out


class SEDenseNetBlock(nn.Module):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel, reduction_ratio):
        super(SEDenseNetBlock, self).__init__()

        # DenseNet unit tanımlaması
        self.densenet_unit = DenseNetUnit(k, amount, in_channel, out_channel, max_input_channel)
        
        # SE block tanımlaması (önceden tanımladığınız SEBlock sınıfını kullanarak)
        self.se_block = SEBlock(in_channels=out_channel, reduction_ratio=reduction_ratio)

    def forward(self, x):
        # DenseNet işlemleri
        dense_out = self.densenet_unit(x)
        
        # Squeeze-and-Excitation işlemleri
        se_out = self.se_block(dense_out)
        
        # Çıkış hesaplaması
        return se_out



class SEDenseNetBlock_old(nn.Module):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel, reduction_ratio):
        super(SEDenseNetBlock, self).__init__()
        self.densenet_unit = DenseNetUnit(k, amount, in_channel, out_channel, max_input_channel)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // reduction_ratio, out_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        dense_out = self.densenet_unit(x)
        se_out = self.se_block(dense_out)
        return dense_out * se_out



class CBAMDenseNetBlock(nn.Module):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel, reduction_ratio):
        super(CBAMDenseNetBlock, self).__init__()

        # DenseNet unit tanımlaması
        self.densenet_unit = DenseNetUnit(k, amount, in_channel, out_channel, max_input_channel)
        
        # CBAM block tanımlaması (önceden tanımladığınız CBAM sınıfını kullanarak)
        self.cbam_block = CBAM(in_channels=out_channel, reduction_ratio=reduction_ratio)

    def forward(self, x):
        # DenseNet işlemleri
        dense_out = self.densenet_unit(x)
        
        # CBAM (Channel and Spatial Attention) işlemleri
        cbam_out = self.cbam_block(dense_out)
        
        # Çıkış hesaplaması
        return cbam_out


#old olan fonksiyonlarda out_channel içerde ki hesaplamayı karıştırıyordu. bu sebeple attention için ayrı fonksiyon tanımladık ve o şekilde problem şimdilik çözüldü? inşallah.. 
class CBAMDenseNetBlock_old(nn.Module):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel, reduction_ratio):
        super(CBAMDenseNetBlock, self).__init__()
        self.densenet_unit = DenseNetUnit(k, amount, in_channel, out_channel, max_input_channel)

        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(out_channel, out_channel // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // reduction_ratio, out_channel, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        dense_out = self.densenet_unit(x)

        # Channel Attention
        avg_out = torch.mean(dense_out, dim=1, keepdim=True)
        max_out, _ = torch.max(dense_out, dim=1, keepdim=True)
        ca_out = self.channel_attention(avg_out + max_out)
        dense_out = dense_out * ca_out

        # Spatial Attention
        avg_out = torch.mean(dense_out, dim=1, keepdim=True)
        max_out, _ = torch.max(dense_out, dim=1, keepdim=True)
        sa_out = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        dense_out = dense_out * sa_out

        return dense_out



class CADenseNetBlock(nn.Module):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel, reduction_ratio):
        super(CADenseNetBlock, self).__init__()
        self.densenet_unit = DenseNetUnit(k, amount, in_channel, out_channel, max_input_channel)
        self.coord_attention = CoordinateAttentionBlock(out_channel, reduction_ratio=reduction_ratio)

    def forward(self, x):
        dense_out = self.densenet_unit(x)
        ca_out = self.coord_attention(dense_out)
        return ca_out #dense_out * ca_out




# ------------------------------------------------------------
# ECA (Efficient Channel Attention) 
# ------------------------------------------------------------
class ECA_Block(nn.Module):
    """
    Efficient Channel Attention (ECA)
    Input/Output: (N, C, H, W) -> (N, C, H, W)
    """
    def __init__(self, k_size=3):
        super().__init__()
        if k_size % 2 == 0:
            raise ValueError("ECA k_size should be odd (e.g., 3,5,7) for symmetric padding.")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, C, H, W)
        y = self.avg_pool(x)                    # (N, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)     # (N, 1, C)
        y = self.conv(y)                        # (N, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)   # (N, C, 1, 1)
        y = self.sigmoid(y)
        return x * y


# ------------------------------------------------------------
# ECA-Inception (refactored style: concat -> ECA)
# ------------------------------------------------------------
class ECAInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        red_3x3, out_3x3,
        red_5x5, out_5x5,
        out_1x1pool,
        k_size=3
    ):
        super().__init__()

        # Inception branches
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            conv_block(in_channels, out_1x1pool, kernel_size=(1, 1)),
        )

        # ECA after concatenation
        self.eca_block = ECA_Block(k_size=k_size)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.eca_block(out)
        return out


# ------------------------------------------------------------
# ECA-ResNet (refactored style: ResNetUnit -> ECA)
# ------------------------------------------------------------
class ECAResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, amount, k_size=3):
        super().__init__()
        self.resnet_unit = ResNetUnit(amount, in_channel, out_channel)
        self.eca_block = ECA_Block(k_size=k_size)

    def forward(self, x):
        res_out = self.resnet_unit(x)
        out = self.eca_block(res_out)
        return out


# ------------------------------------------------------------
# ECA-DenseNet (refactored style: DenseNetUnit -> ECA)
# ------------------------------------------------------------
class ECADenseNetBlock(nn.Module):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel, k_size=3):
        super().__init__()
        self.densenet_unit = DenseNetUnit(k, amount, in_channel, out_channel, max_input_channel)
        self.eca_block = ECA_Block(k_size=k_size)

    def forward(self, x):
        dense_out = self.densenet_unit(x)
        out = self.eca_block(dense_out)
        return out





class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #generated_init


    def forward(self, x):
        #generate_forward
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#################################### TRAINING PART ################################################################################


class EarlyStopping:
    #Early stops the training if validation loss doesn't improve or training loss surpasses validation loss.
    def __init__(self, patience=5, verbose=False, delta=0.001, monitor_training_vs_validation=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.monitor_training_vs_validation = monitor_training_vs_validation  # Flag for training vs validation comparison
        self.train_vs_val_counter = 0

    def __call__(self, val_loss, train_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        # Compare training loss with validation loss
        if self.monitor_training_vs_validation:
            if train_loss > val_loss + self.delta:
                self.train_vs_val_counter += 1
                if self.verbose:
                    print(f'Training loss {train_loss:.4f} > Validation loss {val_loss:.4f}')
                if self.train_vs_val_counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"Early stopping triggered due to training loss surpassing validation loss.")
            else:
                self.train_vs_val_counter = 0




# Import Horovod conditionally
# (Assuming StatusUpdateTool.is_horovod_enabled() returns True if Horovod should be used)
if StatusUpdateTool.is_horovod_enabled():
    import horovod.torch as hvd



class TrainModel(object):
    def __init__(self):
        # --- Horovod Enabled Check ---
        self.horovod_enabled = StatusUpdateTool.is_horovod_enabled()
        
        #print("self.horovod_enabled",self.horovod_enabled, flush=True)
        if self.horovod_enabled:           
            device = torch.device('cuda', hvd.local_rank())
            # Süreç ve GPU bilgisini yazdırın
            #print(f"Process {hvd.rank()} is using GPU {hvd.local_rank()} ({torch.cuda.get_device_name(device)})", flush=True)                     
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 by default
        # --------------------------------
        
        dataset_dir = StatusUpdateTool.is_dataset_directory_set()
        
        #print("Rank X - Reach the trainloader", flush=True)

        if self.horovod_enabled:
            # Veri yükleyicilerini oluşturma
            trainloader, validate_loader = data_loader.get_train_valid_loader(
                dataset_dir, batch_size=64, augment=True, valid_size=0.1, 
                shuffle=True, random_seed=2312390, show_sample=False, num_workers=20, pin_memory=True
            )
        else:
            # This is for kaggle servers. Lower the worker = 4
            trainloader, validate_loader = data_loader.get_train_valid_loader(
                dataset_dir, batch_size=64, augment=True, valid_size=0.1, 
                shuffle=True, random_seed=2312390, show_sample=False, num_workers=4, pin_memory=True
            )
        
        if self.horovod_enabled:
            hvd.barrier()
            
        net = EvoCNNModel()



        #from torch.utils.flop_counter import FlopCounterMode        
        # Giriş verisini tanımlayın (örneğin, batch_size=1)
        #input_tensor = torch.randn(1, 3, 32, 32)

        # FLOP sayacını başlatın
        #with FlopCounterMode(net) as flop_counter:
        #    # İleri geçişi çalıştırın
        #    output = net(input_tensor)
        #    # Toplam FLOP sayısını alın
        #    total_flops = flop_counter.get_total_flops()
            
        #total_mflops = total_flops / 1_000_000
        #print(f"Toplam MegaFLOP sayısı: {total_mflops:.2f} MFLOP")
        ##print(f"Toplam FLOP sayısı: {total_flops}")

        # Giriş boyutunu belirleyin
        #input_size = (3, 32, 32)  # CINIC-10 veri seti için giriş boyutu
        # FLOP ve parametre hesaplamalarını yapın
        #flops, params = get_model_complexity_info(net, input_size, as_strings=True, print_per_layer_stat=True)
        #print(f"FLOPs: {flops}")
        #print(f"Parametre Sayısı: {params}")

        
        #print(net, flush=True)
        cudnn.benchmark = True
        net = net.cuda()
        best_acc = 0.0

        #print("net = net.cuda()", flush=True)
        
        #if self.horovod_enabled:
        #    hvd.barrier()
            
        # Early stopping örneği
        self.early_stopping_enabled = StatusUpdateTool.is_early_stopping_enabled()
        if self.early_stopping_enabled:
            self.early_stopping = EarlyStopping(patience=5, verbose=True, monitor_training_vs_validation=False)
        else:
            self.early_stopping = None

        # --- Label Smoothing Enabled Check ---
        self.label_smoothing_enabled = StatusUpdateTool.is_label_smoothing_enabled()
        if self.label_smoothing_enabled:
            # Adjust the smoothing factor as needed
            smoothing = 0.1
            self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # --------------------------------------

        # --- Gradient Clipping Enabled Check ---
        self.gradient_clipping_enabled = StatusUpdateTool.is_gradient_clipping_enabled()
        if self.gradient_clipping_enabled:
            # Set the maximum norm for gradient clipping
            self.max_norm = 1.0  # Adjust as needed
        # ----------------------------------------

        self.net = net
        self.best_acc = best_acc
        self.trainloader = trainloader
        self.validate_loader = validate_loader
        self.file_id = os.path.basename(__file__).split('.')[0]

        #print("Before lr_custom function", flush=True)

        
        
        from torch.optim.lr_scheduler import LambdaLR
        
        # LambdaLR mutlak öğrenme oranı değil,
        # başlangıç öğrenme oranına uygulanacak çarpanı bekler.
        def custom_lr_scheduler(epoch):
            if epoch == 0:
                return 1.0
            elif epoch > 0 and epoch <= 10:
                return 10.0
            elif epoch > 10 and epoch <= 20:
                return 1.0
            else:  # epoch > 20
                return 0.1
        
        # Her GPU/rank için batch size sabit olduğu için
        # global batch büyüdükçe LR doğrusal olarak ölçeklenir.
        if self.horovod_enabled:
            lr_scaler = hvd.size()
        else:
            lr_scaler = 1
        
        # 1 GPU: 0.01
        # 2 GPU: 0.02
        # 4 GPU: 0.04
        base_lr = 0.01 * lr_scaler
        
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=base_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=custom_lr_scheduler
        )
        
        

        
        #if self.horovod_enabled:
        #    hvd.barrier()

        #print("Before optimizier and hvd.broadcast..", flush=True)
        
        # Wrap optimizer with Horovod DistributedOptimizer
        if self.horovod_enabled:
            self.optimizer = hvd.DistributedOptimizer(
                self.optimizer,
                named_parameters=self.net.named_parameters()
            )
            #print("After hvd.DistributedOptimizer...", flush=True)
            
            #hvd.barrier()
            
            # Broadcast parameters and optimizer state from rank 0 to all other processes
            hvd.broadcast_parameters(self.net.state_dict(), root_rank=0)
            
            #print("hvd.broadcast_parameters", flush=True)
            
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            
            #print("hvd.broadcast_optimizer_state", flush=True)


    #Bu fonksiyon rank 0 a atanacak
    
    def log_record(self, _str, first_time=None):
        dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_mode = 'w' if first_time else 'a+'
        with open('./log/%s.txt' % (self.file_id), file_mode) as f:
            f.write('[%s] - %s\n' % (dt, _str))

    def train(self, epoch):
        self.net.train()

        running_loss = 0.0
        total = 0
        correct = 0

        # Set epoch for sampler if Horovod is enabled
        if self.horovod_enabled:
            self.trainloader.sampler.set_epoch(epoch)
            hvd.barrier()

        for _, data in enumerate(self.trainloader, 0):
            inputs, labels = data  
            inputs, labels = inputs.cuda(), labels.cuda()

            self.optimizer.zero_grad()  
            outputs = self.net(inputs)  

            # --- Label Smoothing Application ---
            if self.label_smoothing_enabled:
                loss = self.criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            # ------------------------------------


            if self.horovod_enabled:
                hvd.barrier()
            loss.backward()  
            torch.cuda.synchronize()
            
            # --- Gradient Clipping Application ---
            if self.gradient_clipping_enabled:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)
            # --------------------------------------

            # Horovod: synchronize gradients
            if self.horovod_enabled:
                self.optimizer.synchronize()
                hvd.barrier()

                # Horovod: skip synchronization during optimizer step
                with self.optimizer.skip_synchronize():
                    self.optimizer.step()  
            else:
                self.optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        average_loss = running_loss / total  # Ortalama eğitim kaybı

        if self.horovod_enabled:
            hvd.barrier()

        return average_loss, accuracy

    def test(self):
        self.net.eval()

        test_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for _, data in enumerate(self.validate_loader, 0):
                inputs, labels = data  
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.net(inputs)  
                loss = self.criterion(outputs, labels)  
                test_loss += loss.item() * labels.size(0)  
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        average_loss = test_loss / total

        
        
        # Horovod: average metrics across all workers
        #if self.horovod_enabled:
        #   average_loss_tensor = torch.tensor(average_loss).cuda()
        #  accuracy_tensor = torch.tensor(accuracy).cuda()

            # Use Horovod allreduce to average the metrics
         #   average_loss = hvd.allreduce(average_loss_tensor, name='avg_loss').item()
          #  accuracy = hvd.allreduce(accuracy_tensor, name='avg_accuracy').item()
        
        
        # En iyi doğruluk değerini güncelle
        if accuracy > self.best_acc:
            self.best_acc = accuracy

        return average_loss  # Ortalama doğrulama kaybını döndür

    def process(self):
        #print("Rank call the process function", flush=True)
        total_epoch = StatusUpdateTool.get_epoch_size()
        start_time = time.time()
        for p in range(total_epoch):
            # Bir epoch boyunca modeli eğit ve eğitim kaybını ve doğruluğunu al
            #print("epoch starts...", flush=True)
            #start_time = time.time()

            if self.horovod_enabled:
                hvd.barrier() 
            
            current_train_loss, train_accuracy = self.train(p)
            
            if self.horovod_enabled:
                hvd.barrier() 
            #end_time = time.time()
            #print("A TRAINING EPOCH FINISHED...", flush=True)
            #epoch_duration = end_time - start_time
            #print(f"Epoch completed in {epoch_duration:.2f} seconds.")
            
            # Scheduler adımını at
            self.scheduler.step()

            # Modeli doğrulama verisinde test et ve doğrulama kaybını al
            if self.horovod_enabled:
                hvd.barrier() 


            # Bu test kısmı daha sonra birden fazla gpu ile yapılabilir.
            if not self.horovod_enabled or (self.horovod_enabled and hvd.rank() == 0):    
                current_val_loss = self.test()
                

            # Eğitim ve doğrulama metriklerini logla
            if not self.horovod_enabled or (self.horovod_enabled and hvd.rank() == 0):
                self.log_record(f'Epoch {p+1}, Train Loss: {current_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {self.best_acc:.4f}')

            if self.horovod_enabled:
                hvd.barrier() 
            
            # Early stopping kontrolü
            if self.early_stopping_enabled and self.early_stopping is not None:
                self.early_stopping(current_val_loss, current_train_loss, self.net)

                if self.early_stopping.early_stop:
                    if not self.horovod_enabled or (self.horovod_enabled and hvd.rank() == 0):
                        print(f"Early stopping at epoch {p+1}")
                    break  # Eğitim döngüsünden çık
        end_time = time.time()
        epoch_duration = end_time - start_time
        if not self.horovod_enabled or (self.horovod_enabled and hvd.rank() == 0):
            print(f"Training completed in {epoch_duration:.2f} seconds.", flush=True)
            
        return self.best_acc

class RunModel(object):
    # --- Horovod Enabled Check ---
    horovod_enabled = StatusUpdateTool.is_horovod_enabled()
    
    #if not horovod_enabled or (horovod_enabled and hvd.rank() == 0):
    #print("RUN MODEL...")
    
    #def do_work(self, gpu_id, file_id): #original version
    def do_work(self, file_id, gpu_id=None):
        """Train one architecture and save fitness only after success."""
    
        horovod_enabled = (
            StatusUpdateTool.is_horovod_enabled()
        )
    
        model_runner = None
    
        try:
            if not horovod_enabled:
                os.environ[
                    'CUDA_VISIBLE_DEVICES'
                ] = '0'
    
            model_runner = TrainModel()
    
            if (
                not horovod_enabled
                or hvd.rank() == 0
            ):
                model_runner.log_record(
                    'Used GPU#%s, worker name:%s[%d]'
                    % (
                        gpu_id,
                        multiprocessing
                        .current_process()
                        .name,
                        os.getpid(),
                    ),
                    first_time=True,
                )
    
            best_acc = model_runner.process()
    
            # Keep the existing Horovod barrier.
            if horovod_enabled:
                hvd.barrier()
    
            if (
                not horovod_enabled
                or hvd.rank() == 0
            ):
                model_runner.log_record(
                    'Finished-Acc:%.3f'
                    % best_acc
                )
    
                with open(
                    './populations/after_%s.txt'
                    % file_id[4:6],
                    'a+',
                ) as handle:
                    handle.write(
                        '%s=%.5f\n'
                        % (
                            file_id,
                            best_acc,
                        )
                    )
                    handle.flush()
    
            return best_acc
    
        except Exception as exc:
            if (
                not horovod_enabled
                or hvd.rank() == 0
            ):
                print(
                    'Exception occurs, '
                    'file:%s, pid:%d...%s'
                    % (
                        file_id,
                        os.getpid(),
                        str(exc),
                    ),
                    flush=True,
                )
    
                if model_runner is not None:
                    model_runner.log_record(
                        'Exception occur:%s'
                        % str(exc)
                    )
    
            # Do not save a failed run
            # as a real zero-fitness result.
            raise

            
"""
