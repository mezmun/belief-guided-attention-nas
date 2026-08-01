import numpy as np
import hashlib
import copy
import random

import pickle

#### Newly added attention blocks SE,    #############################################
import torch
import torch.nn as nn

from copy import deepcopy
from collections import Counter  # ✅ bunu mutlaka ekle


try:
    import horovod.torch as hvd
except ImportError:
    horovod_enabled = False  # Eğer yüklü değilse Horovod'u kapat

'''
######################################################
# ECA (Efficient Channel Attention)
class ECA_Block(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(ECA_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
######################################################
# SEBlock
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
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


###################################################3

#CBAMBlock
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
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
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

###################################################################3


#Coordinate Attention
class h_sigmoid(nn.Module):
    """ h sigmoid """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    """ h swish """
    def __init__(self):
        super().__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

# CoordinateAttentionBlock
class CoordinateAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CoordinateAttentionBlock, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Vertical pooling
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Horizontal pooling

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

        # Coordinate attention: Horizontal and Vertical pooling
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # Horizontal pooling

        # Excitation
        x_cat = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(x_cat)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Apply transformations
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w.permute(0, 1, 3, 2))

        # Combine the attention maps
        attention = self.sigmoid(x_h + x_w)
        
        return identity * attention


  ##################################################################################################################

''' 



class Unit(object):
    def __init__(self, number):
        self.number = number


class ResUnit(Unit):
    def __init__(self, number, amount, in_channel, out_channel): #prob < 0.5
        super().__init__(number)
        self.type = 1
        self.amount = amount
        self.in_channel = in_channel
        self.out_channel = out_channel


class PoolUnit(Unit):
    def __init__(self, number, max_or_avg):
        super().__init__(number)
        self.type = 2
        self.max_or_avg = max_or_avg #max_pool for < 0.5 otherwise avg_pool


class DenseUnit(Unit):
    def __init__(self, number, amount, k, max_input_channel, in_channel, out_channel):
        super().__init__(number)
        self.type = 3
        self.amount = amount
        self.k = k
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.max_input_channel = max_input_channel

class InceptionBlock(Unit):
    def __init__(self, number, in_channel, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super().__init__(number)
        self.type = 4
        # self.amount = amount
        self.in_channel = in_channel
        self.out_1x1 = out_1x1
        self.red_3x3 = red_3x3
        self.out_3x3 = out_3x3
        self.red_5x5 = red_5x5
        self.out_5x5 = out_5x5
        self.out_1x1pool = out_1x1pool
        self.inception_type = inception_type
        


##########################################
class InceptionSEBlock(Unit):
    def __init__(self, number, in_channel, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio=16):
        super().__init__(number)
        self.type = 5  # Unique type identifier for Inception-SE
        self.in_channel = in_channel
        self.out_1x1 = out_1x1
        self.red_3x3 = red_3x3
        self.out_3x3 = out_3x3
        self.red_5x5 = red_5x5
        self.out_5x5 = out_5x5
        self.out_1x1pool = out_1x1pool
        self.inception_type = inception_type
        self.reduction_ratio = reduction_ratio
        
        # Initialize the SE block
        #self.se_block = SEBlock(in_channels=self.out_1x1 + self.out_3x3 + self.out_5x5 + self.out_1x1pool, reduction_ratio=self.reduction_ratio)
    """
    def forward(self, x):
        # Apply Inception operations here
        x1 = self.out_1x1(x)
        x3 = self.out_3x3(self.red_3x3(x))
        x5 = self.out_5x5(self.red_5x5(x))
        x_pool = self.out_1x1pool(x)

        # Concatenate the outputs
        inception_out = torch.cat([x1, x3, x5, x_pool], dim=1)
        
        # Apply SE block
        se_output = self.se_block(inception_out)
        
        return se_output
    """

##############################################################################33

class CBAMInceptionBlock(Unit):
    def __init__(self, number, in_channel, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio=16):
        super().__init__(number)
        self.type = 6  # Unique type identifier for CBAM-Inception
        self.in_channel = in_channel
        self.out_1x1 = out_1x1
        self.red_3x3 = red_3x3
        self.out_3x3 = out_3x3
        self.red_5x5 = red_5x5
        self.out_5x5 = out_5x5
        self.out_1x1pool = out_1x1pool
        self.inception_type = inception_type
        self.reduction_ratio = reduction_ratio

        # Initialize the CBAM block
        #self.cbam_block = CBAM(in_channels=self.out_1x1 + self.out_3x3 + self.out_5x5 + self.out_1x1pool, reduction_ratio=self.reduction_ratio)
    """
    def forward(self, x):
        # Apply Inception operations here
        x1 = self.out_1x1(x)
        x3 = self.out_3x3(self.red_3x3(x))
        x5 = self.out_5x5(self.red_5x5(x))
        x_pool = self.out_1x1pool(x)

        # Concatenate the outputs
        inception_out = torch.cat([x1, x3, x5, x_pool], dim=1)
        
        # Apply CBAM block after concatenation
        cbam_output = self.cbam_block(inception_out)
        
        return cbam_output

    """

#########################################################################################3

class CAInceptionBlock(nn.Module):
    def __init__(self, number, in_channel, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio=16):
        super().__init__()
        self.type = 7  
        self.number = number  # Number attribute added
        self.in_channel = in_channel
        self.out_1x1 = out_1x1
        self.red_3x3 = red_3x3
        self.out_3x3 = out_3x3
        self.red_5x5 = red_5x5
        self.out_5x5 = out_5x5
        self.out_1x1pool = out_1x1pool
        self.inception_type = inception_type
        self.reduction_ratio = reduction_ratio

        # Inception blocks
        self.conv_1x1 = nn.Conv2d(in_channel, out_1x1, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        self.conv_1x1pool = nn.Conv2d(in_channel, out_1x1pool, kernel_size=1)

        # Coordinate Attention block
        #self.ca_block = CoordinateAttentionBlock(in_channels=self.out_1x1 + self.out_3x3 + self.out_5x5 + self.out_1x1pool, reduction_ratio=self.reduction_ratio)
    """
    def forward(self, x):
        # Inception işlemleri
        x1 = self.conv_1x1(x)
        x3 = self.conv_3x3(self.red_3x3(x))
        x5 = self.conv_5x5(self.red_5x5(x))
        x_pool = self.conv_1x1pool(x)

        # 
        inception_out = torch.cat([x1, x3, x5, x_pool], dim=1)
        
        # Coordinate Attention uygulama
        ca_output = self.ca_block(inception_out)
        
        return ca_output
    """
##################################################################3



class SEResNetUnit(Unit):
    def __init__(self, number, amount, in_channel, out_channel, reduction_ratio=16):
        super().__init__(number)
        self.type = 8  # SE-ResNet için benzersiz tip tanımlayıcı
        self.amount = amount
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.reduction_ratio = reduction_ratio
        
        
         # ResNet bloklarını burada tanımlayın
        #self.resnet_blocks = nn.ModuleList([self._build_resnet_block() for _ in range(self.amount)])      
        
        # SE bloğunu başlat
        #self.se_block = SEBlock(in_channels=self.out_channel, reduction_ratio=self.reduction_ratio)


    """
    def _build_resnet_block(self):
        # Burada ResNet bloğu tanımlanabilir (örneğin, Conv2D, BatchNorm, ReLU)
        block = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        # ResNet bloklarını uygula
        for block in self.resnet_blocks:
            x = block(x)

        # SE bloğunu uygula
        se_output = self.se_block(x)

        return se_output

    """

##########################################################################3

class CBAMResNetUnit(Unit):
    def __init__(self, number, amount, in_channel, out_channel, reduction_ratio=16):
        super().__init__(number)
        self.type = 9  # CBAM-ResNet için benzersiz tip tanımlayıcı
        self.amount = amount
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.reduction_ratio = reduction_ratio

        # CBAM bloğunu başlat
        #self.cbam_block = CBAM(in_channels=self.out_channel, reduction_ratio=self.reduction_ratio)

        # ResNet bloklarını burada tanımlayın
        #self.resnet_blocks = nn.ModuleList([self._build_resnet_block() for _ in range(self.amount)])
    
    """
    def _build_resnet_block(self):
        # Burada ResNet bloğu tanımlanabilir (örneğin, Conv2D, BatchNorm, ReLU)
        block = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        # ResNet bloklarını uygula
        for block in self.resnet_blocks:
            x = block(x)

        # CBAM bloğunu uygula
        cbam_output = self.cbam_block(x)

        return cbam_output
    """
    
##########################################################3    
    
class CAResNetUnit(Unit):
    def __init__(self, number, amount, in_channel, out_channel, reduction_ratio=16):
        super().__init__(number)
        self.type = 10  # CA-ResNet için benzersiz tip tanımlayıcı
        self.amount = amount
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.reduction_ratio = reduction_ratio

        # ResNet bloklarını burada tanımlayın
        #self.resnet_blocks = nn.ModuleList([self._build_resnet_block() for _ in range(self.amount)])

        # Coordinate Attention bloğunu başlat
        #self.ca_block = CoordinateAttentionBlock(in_channels=self.out_channel, reduction_ratio=self.reduction_ratio)



    """
    def _build_resnet_block(self):
        # Burada ResNet bloğu tanımlanabilir (örneğin, Conv2D, BatchNorm, ReLU)
        block = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        return block
    
    
    def forward(self, x):
        # ResNet bloklarını uygula
        for block in self.resnet_blocks:
            x = block(x)

        # Coordinate Attention bloğunu uygula
        ca_output = self.ca_block(x)

        return ca_output

    """
################################################


class SEDenseNetUnit(Unit):
    def __init__(self, number, amount, k, max_input_channel, in_channel, out_channel, reduction_ratio=16):
        super().__init__(number)
        self.type = 11  # SE-DenseNet için benzersiz tip tanımlayıcı
        self.amount = amount
        self.k = k
        self.max_input_channel = max_input_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.reduction_ratio = reduction_ratio

        # DenseNet bloklarını burada tanımlayın
        #self.densenet_blocks = nn.ModuleList([self._build_densenet_block(i) for i in range(self.amount)])

        # SE bloğunu başlat
        #self.se_block = SEBlock(in_channels=self.out_channel, reduction_ratio=self.reduction_ratio)

    """
    def _build_densenet_block(self, layer_index):
        # Burada DenseNet bloğu tanımlanabilir
        growth_rate = self.k
        input_channels = self.in_channel + layer_index * growth_rate
        block = nn.Sequential(
            nn.Conv2d(input_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        return block
    """
    """
    def forward(self, x):
        # DenseNet bloklarını uygula
        for block in self.densenet_blocks:
            x = torch.cat([x, block(x)], dim=1)

        # SE bloğunu uygula
        se_output = self.se_block(x)

        return se_output
    """



class CBAMDenseNetUnit(Unit):
    def __init__(self, number, amount, k, max_input_channel, in_channel, out_channel, reduction_ratio=16):
        super().__init__(number)
        self.type = 12  # CBAM-DenseNet için benzersiz tip tanımlayıcı
        self.amount = amount
        self.k = k
        self.max_input_channel = max_input_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.reduction_ratio = reduction_ratio

        # DenseNet bloklarını burada tanımlayın
        #self.densenet_blocks = nn.ModuleList([self._build_densenet_block(i) for i in range(self.amount)])

        # CBAM bloğunu başlat
        #self.cbam_block = CBAM(in_channels=self.out_channel, reduction_ratio=self.reduction_ratio)


    """
    def _build_densenet_block(self, layer_index):
        # Burada DenseNet bloğu tanımlanabilir
        growth_rate = self.k
        input_channels = self.in_channel + layer_index * growth_rate
        block = nn.Sequential(
            nn.Conv2d(input_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        return block
    
    """
    """
    def forward(self, x):
        # DenseNet bloklarını uygula
        for block in self.densenet_blocks:
            x = torch.cat([x, block(x)], dim=1)

        # CBAM bloğunu uygula
        cbam_output = self.cbam_block(x)

        return cbam_output
    """


class CADenseNetUnit(Unit):
    def __init__(self, number, amount, k, max_input_channel, in_channel, out_channel, reduction_ratio=16):
        super().__init__(number)
        self.type = 13  # CA-DenseNet için benzersiz tip tanımlayıcı
        self.amount = amount
        self.k = k
        self.max_input_channel = max_input_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.reduction_ratio = reduction_ratio

        # DenseNet bloklarını burada tanımlayın
        #self.densenet_blocks = nn.ModuleList([self._build_densenet_block(i) for i in range(self.amount)])

        # Coordinate Attention bloğunu başlat
        #self.ca_block = CoordinateAttentionBlock(in_channels=self.out_channel, reduction_ratio=self.reduction_ratio)


    """
    def _build_densenet_block(self, layer_index):
        # Burada DenseNet bloğu tanımlanabilir
        growth_rate = self.k
        input_channels = self.in_channel + layer_index * growth_rate
        block = nn.Sequential(
            nn.Conv2d(input_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        return block
    """ 
    """
    def forward(self, x):
        # DenseNet bloklarını uygula
        for block in self.densenet_blocks:
            x = torch.cat([x, block(x)], dim=1)

        # Coordinate Attention bloğunu uygula
        ca_output = self.ca_block(x)

        return ca_output

    """

class ECAInceptionBlock(Unit):
    def __init__(self, number, in_channel, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, k_size=3):
        super().__init__(number)
        self.type = 14  # Benzersiz type kodu, SE'den farklı
        self.in_channel = in_channel
        self.out_1x1 = out_1x1
        self.red_3x3 = red_3x3
        self.out_3x3 = out_3x3
        self.red_5x5 = red_5x5
        self.out_5x5 = out_5x5
        self.out_1x1pool = out_1x1pool
        self.inception_type = inception_type
        self.k_size = k_size

        # ECA modülünü tanımla
        #self.eca_block = ECA_Block(k_size=self.k_size) #gereksiz olabilir

class ECAResNetUnit(Unit):
    def __init__(self, number, amount, in_channel, out_channel, k_size=3):
        super().__init__(number)
        self.type = 15
        self.amount = amount
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k_size = k_size
        # self.eca_block = ECA_Block(k_size=self.k_size)  # GEREKSİZ, kullanmayacaksan


class ECADenseNetUnit(Unit):
    def __init__(self, number, amount, k, max_input_channel, in_channel, out_channel, k_size=3):
        super().__init__(number)
        self.type = 16  # ECA-DenseNet için tanımlayıcı numara
        self.amount = amount
        self.k = k
        self.max_input_channel = max_input_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k_size = k_size  # ECA attention çekirdek boyutu


def get_inception_params(inception_type):
    params = {
        "3a":  (64, 96, 128, 16, 32, 32),
        "3b":  (128, 128, 192, 32, 96, 64),
        "4a":  (192, 96, 208, 16, 48, 64),
        "4b":  (160, 112, 224, 24, 64, 64),
        "4c":  (128, 128, 256, 24, 64, 64),
        "4d":  (112, 144, 288, 32, 64, 64),
        "4e":  (256, 160, 320, 32, 128, 128),
        "5a":  (256, 160, 320, 32, 128, 128),
        "5b":  (384, 192, 384, 48, 128, 128),
    }
    return params[inception_type]

class Individual(object):
    def __init__(self, params, indi_no):
        self.acc = -1.0
        self.id = indi_no  # for recording the id of the current individual
        self.number_id = 0  # for recording the latest number of basic units
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_channles = params['output_channel']

        self.min_resnet = params['min_resnet']  # minimal number of resnet units
        self.max_resnet = params['max_resnet']  # maximal number of resnet units
        self.min_pool = params['min_pool']  # minimal number of pool units
        self.max_pool = params['max_pool']  # maximal number of pool units
        self.min_densenet = params['min_densenet']  # minimal number of densenet units
        self.max_densenet = params['max_densenet']  # maximal number of densenet units
        self.min_inception = params['min_inception']  # minimal number of inception blocks
        self.max_inception = params['max_inception']  # maximal number of inception blocks

        ########### Added new module parameters ###########
        self.min_inception_se = params.get('min_inception_se', 0)  # minimal number of inception-SE blocks
        self.max_inception_se = params.get('max_inception_se', 1)  # maximal number of inception-SE blocks

        self.min_inception_cbam = params.get('min_inception_cbam', 0)  # minimal number of CBAM-Inception blocks
        self.max_inception_cbam = params.get('max_inception_cbam', 1)  # maximal number of CBAM-Inception blocks

        self.min_inception_ca = params.get('min_inception_ca', 0)  # minimal number of Inception-CA blocks
        self.max_inception_ca = params.get('max_inception_ca', 1)  # maximal number of Inception-CA blocks

        self.min_inception_eca = params.get('min_inception_eca', 0)  # minimal number of inception-ECA blocks
        self.max_inception_eca = params.get('max_inception_eca', 0)  # maximal number of inception-ECA blocks
        
        # Parameters for SE-ResNet, CBAM-ResNet, and CA-ResNet
        self.min_resnet_se = params.get('min_resnet_se', 0)
        self.max_resnet_se = params.get('max_resnet_se', 1)
        
        self.min_resnet_cbam = params.get('min_resnet_cbam', 0)
        self.max_resnet_cbam = params.get('max_resnet_cbam', 1)
        
        self.min_resnet_ca = params.get('min_resnet_ca', 0)
        self.max_resnet_ca = params.get('max_resnet_ca', 1)

        self.min_resnet_eca = params.get('min_resnet_eca', 0)
        self.max_resnet_eca = params.get('max_resnet_eca', 0)
        
        # Parameters for SE-DenseNet, CBAM-DenseNet, and CA-DenseNet
        self.min_densenet_se = params.get('min_densenet_se', 0)
        self.max_densenet_se = params.get('max_densenet_se', 1)
        
        self.min_densenet_cbam = params.get('min_densenet_cbam', 0)
        self.max_densenet_cbam = params.get('max_densenet_cbam', 1)
        
        self.min_densenet_ca = params.get('min_densenet_ca', 0)
        self.max_densenet_ca = params.get('max_densenet_ca', 1)

        self.min_densenet_eca = params.get('min_densenet_eca', 0)
        self.max_densenet_eca = params.get('max_densenet_eca', 0)
        ###################################################

        self.min_resnet_unit = params['min_resnet_unit']
        self.max_resnet_unit = params['max_resnet_unit']

        self.k_list = params['k_list']
        self.min_k12 = params['min_k12']  # minimal number of k_12 for densenet
        self.max_k12 = params['max_k12']
        self.min_k20 = params['min_k20']
        self.max_k20 = params['max_k20']
        self.min_k40 = params['min_k40']
        self.max_k40 = params['max_k40']

        self.max_k12_input_channel = params['max_k12_input_channel']  # if the k is set to 12, its input channel cannot exceed this setting
        self.max_k20_input_channel = params['max_k20_input_channel']
        self.max_k40_input_channel = params['max_k40_input_channel']
        
        self.units = []

    #@staticmethod
    def population_with_all_positions(self, pop_size=20, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
        module_name_to_id = {
            'resnet': 1, 'pool': 2, 'densenet': 3, 'inception': 4,
            'inception_se': 5, 'inception_cbam': 6, 'inception_ca': 7,
            'resnet_se': 8, 'resnet_cbam': 9, 'resnet_ca': 10,
            'densenet_se': 11, 'densenet_cbam': 12, 'densenet_ca': 13, 'inception_eca': 14, 'resnet_eca': 15, 'densenet_eca': 16
        }
        
        #params = params()
    
        population = []
        for _ in range(pop_size):
    
            num_resnet = np.random.randint(self.min_resnet, self.max_resnet + 1)
            num_pool = np.random.randint(self.min_pool, self.max_pool + 1)
            num_densenet = np.random.randint(self.min_densenet, self.max_densenet + 1)
            num_inception = np.random.randint(self.min_inception, self.max_inception + 1)
            
            num_inception_se = np.random.randint(self.min_inception_se, self.max_inception_se + 1)
            num_inception_cbam = np.random.randint(self.min_inception_cbam, self.max_inception_cbam + 1)
            num_inception_ca = np.random.randint(self.min_inception_ca, self.max_inception_ca + 1)
            num_resnet_se = np.random.randint(self.min_resnet_se, self.max_resnet_se + 1)
            num_resnet_cbam = np.random.randint(self.min_resnet_cbam, self.max_resnet_cbam + 1)
            num_resnet_ca = np.random.randint(self.min_resnet_ca, self.max_resnet_ca + 1)
            num_densenet_se = np.random.randint(self.min_densenet_se, self.max_densenet_se + 1)
            num_densenet_cbam = np.random.randint(self.min_densenet_cbam, self.max_densenet_cbam + 1)
            num_densenet_ca = np.random.randint(self.min_densenet_ca, self.max_densenet_ca + 1)

            num_inception_eca = np.random.randint(self.min_inception_eca, self.max_inception_eca + 1)
            num_resnet_eca = np.random.randint(self.min_resnet_eca, self.max_resnet_eca + 1)
            num_densenet_eca = np.random.randint(self.min_densenet_eca, self.max_densenet_eca + 1)
            
            """
            num_resnet = np.random.randint(params.min_resnet, params.max_resnet + 1)
            num_pool = np.random.randint(params.min_pool, params.max_pool + 1)
            num_densenet = np.random.randint(params.min_densenet, params.max_densenet + 1)
            num_inception = np.random.randint(params.min_inception, params.max_inception + 1)
            num_inception_se = np.random.randint(params.min_inception_se, params.max_inception_se + 1)
            num_inception_cbam = np.random.randint(params.min_inception_cbam, params.max_inception_cbam + 1)
            num_inception_ca = np.random.randint(params.min_inception_ca, params.max_inception_ca + 1)
            num_resnet_se = np.random.randint(params.min_resnet_se, params.max_resnet_se + 1)
            num_resnet_cbam = np.random.randint(params.min_resnet_cbam, params.max_resnet_cbam + 1)
            num_resnet_ca = np.random.randint(params.min_resnet_ca, params.max_resnet_ca + 1)
            num_densenet_se = np.random.randint(params.min_densenet_se, params.max_densenet_se + 1)
            num_densenet_cbam = np.random.randint(params.min_densenet_cbam, params.max_densenet_cbam + 1)
            num_densenet_ca = np.random.randint(params.min_densenet_ca, params.max_densenet_ca + 1)
            """
            
            total_length = (
                num_resnet + num_pool + num_densenet + num_inception +
                num_inception_se + num_inception_cbam + num_inception_ca +
                num_resnet_se + num_resnet_cbam + num_resnet_ca +
                num_densenet_se + num_densenet_cbam + num_densenet_ca + num_inception_eca + num_resnet_eca + num_densenet_eca
            )
    
            all_positions = np.zeros(total_length, np.int32)
    
            current_index = 0
            if num_resnet > 0:
                all_positions[current_index:current_index + num_resnet] = 1
                current_index += num_resnet
            if num_pool > 0:
                all_positions[current_index:current_index + num_pool] = 2
                current_index += num_pool
            if num_densenet > 0:
                all_positions[current_index:current_index + num_densenet] = 3
                current_index += num_densenet
            if num_inception > 0:
                all_positions[current_index:current_index + num_inception] = 4
                current_index += num_inception
    
            if num_inception_se > 0:
                all_positions[current_index:current_index + num_inception_se] = 5
                current_index += num_inception_se
            if num_inception_cbam > 0:
                all_positions[current_index:current_index + num_inception_cbam] = 6
                current_index += num_inception_cbam
            if num_inception_ca > 0:
                all_positions[current_index:current_index + num_inception_ca] = 7
                current_index += num_inception_ca
            if num_resnet_se > 0:
                all_positions[current_index:current_index + num_resnet_se] = 8
                current_index += num_resnet_se
            if num_resnet_cbam > 0:
                all_positions[current_index:current_index + num_resnet_cbam] = 9
                current_index += num_resnet_cbam
            if num_resnet_ca > 0:
                all_positions[current_index:current_index + num_resnet_ca] = 10
                current_index += num_resnet_ca
            if num_densenet_se > 0:
                all_positions[current_index:current_index + num_densenet_se] = 11
                current_index += num_densenet_se
            if num_densenet_cbam > 0:
                all_positions[current_index:current_index + num_densenet_cbam] = 12
                current_index += num_densenet_cbam
            if num_densenet_ca > 0:
                all_positions[current_index:current_index + num_densenet_ca] = 13
                current_index += num_densenet_ca
                
            if num_inception_eca > 0:
                all_positions[current_index:current_index + num_inception_eca] = 14
                current_index += num_inception_eca
            if num_resnet_eca > 0:
                all_positions[current_index:current_index + num_resnet_eca] = 15
                current_index += num_resnet_eca
            if num_densenet_eca > 0:
                all_positions[current_index:current_index + num_densenet_eca] = 16
                current_index += num_densenet_eca
    
            for __ in range(10):
                np.random.shuffle(all_positions)
            while len(all_positions) and all_positions[0] == 2:
                np.random.shuffle(all_positions)
    
            population.append(all_positions)
        print("population",population)
    
        return population, module_name_to_id

    @staticmethod
    def balance_population_arrays(population_arrays, module_name_to_id, tolerance=1):
        disabled_modules = {
            'inception_eca',
            'resnet_eca',
            'densenet_eca',
        }


        adjustable_ids = [
            module_id
            for module_name, module_id in module_name_to_id.items()
            if (
                module_name != 'pool'
                and module_name not in disabled_modules
            )
        ]
        
                
        #adjustable_ids = [v for k, v in module_name_to_id.items() if k != 'pool']
        usage = Counter()
        for arr in population_arrays:
            usage.update(arr.tolist())
    
        target_avg = sum(usage[mid] for mid in adjustable_ids) / len(adjustable_ids)
    
        adjusted_pop = deepcopy(population_arrays)
        adjusted_usage = deepcopy(usage)
    
        for mod_id in adjustable_ids:
            diff = adjusted_usage[mod_id] - round(target_avg)
            # Fazla
            while diff > tolerance:
                for i, arr in enumerate(adjusted_pop):
                    idxs = np.where(arr == mod_id)[0]
                    if len(idxs) > 0:
                        adjusted_pop[i] = np.delete(arr, idxs[0])
                        adjusted_usage[mod_id] -= 1
                        diff -= 1
                    if diff <= tolerance:
                        break
            # Eksik
            while diff < -tolerance:
                for i, arr in enumerate(adjusted_pop):
                    insert_pos = np.random.randint(0, len(arr)+1)
                    adjusted_pop[i] = np.insert(arr, insert_pos, mod_id)
                    adjusted_usage[mod_id] += 1
                    diff += 1
                    if diff >= -tolerance:
                        break
    
        # Son shuffle
        final_pop = []
        for arr in adjusted_pop:
            arr2 = arr.copy()
            for _ in range(100):
                np.random.shuffle(arr2)
                if len(arr2) == 0 or arr2[0] != module_name_to_id['pool']:
                    final_pop.append(arr2)
                    break
            else:
                # guarantee fix
                non_pool = np.where(arr2 != module_name_to_id['pool'])[0]
                if len(non_pool) > 0:
                    first = non_pool[0]
                    arr2[0], arr2[first] = arr2[first], arr2[0]
                final_pop.append(arr2)
    
        return final_pop
    
    

    
    def reset_acc(self):
        self.acc = -1.0
        
    def initialize(self,all_positions):
    #def initialize(self):
        
        """
        # Initialize how many units of each type will be used
        num_resnet = np.random.randint(self.min_resnet, self.max_resnet + 1)
        num_pool = np.random.randint(self.min_pool, self.max_pool + 1)
        num_densenet = np.random.randint(self.min_densenet, self.max_densenet + 1)
        num_inception = np.random.randint(self.min_inception, self.max_inception + 1)
        num_inception_se = np.random.randint(self.min_inception_se, self.max_inception_se + 1)
        num_inception_cbam = np.random.randint(self.min_inception_cbam, self.max_inception_cbam + 1)
        num_inception_ca = np.random.randint(self.min_inception_ca, self.max_inception_ca + 1)
    
        # Initialize how many of the new types of units will be used
        num_resnet_se = np.random.randint(self.min_resnet_se, self.max_resnet_se + 1)
        num_resnet_cbam = np.random.randint(self.min_resnet_cbam, self.max_resnet_cbam + 1)
        num_resnet_ca = np.random.randint(self.min_resnet_ca, self.max_resnet_ca + 1)
    
        num_densenet_se = np.random.randint(self.min_densenet_se, self.max_densenet_se + 1)
        num_densenet_cbam = np.random.randint(self.min_densenet_cbam, self.max_densenet_cbam + 1)
        num_densenet_ca = np.random.randint(self.min_densenet_ca, self.max_densenet_ca + 1)

        
        # Total number of units to be created
        total_length = (num_resnet + num_pool + num_densenet + num_inception +
                        num_inception_se + num_inception_cbam + num_inception_ca +
                        num_resnet_se + num_resnet_cbam + num_resnet_ca +
                        num_densenet_se + num_densenet_cbam + num_densenet_ca)

        """
        """
        # Total number of units to be created
        total_length = (num_resnet + num_pool + num_densenet + num_inception)
        """
        """
        # Define positions for each type of unit
        all_positions = np.zeros(total_length, np.int32)
        
        current_index = 0
        if num_resnet > 0: 
            all_positions[current_index:current_index + num_resnet] = 1
            current_index += num_resnet
        if num_pool > 0: 
            all_positions[current_index:current_index + num_pool] = 2
            current_index += num_pool
        if num_densenet > 0: 
            all_positions[current_index:current_index + num_densenet] = 3
            current_index += num_densenet
        if num_inception > 0: 
            all_positions[current_index:current_index + num_inception] = 4
            current_index += num_inception

        
        if num_inception_se > 0: 
            all_positions[current_index:current_index + num_inception_se] = 5
            current_index += num_inception_se
        if num_inception_cbam > 0: 
            all_positions[current_index:current_index + num_inception_cbam] = 6
            current_index += num_inception_cbam
        if num_inception_ca > 0: 
            all_positions[current_index:current_index + num_inception_ca] = 7
            current_index += num_inception_ca
        if num_resnet_se > 0: 
            all_positions[current_index:current_index + num_resnet_se] = 8
            current_index += num_resnet_se
        if num_resnet_cbam > 0: 
            all_positions[current_index:current_index + num_resnet_cbam] = 9
            current_index += num_resnet_cbam
        if num_resnet_ca > 0: 
            all_positions[current_index:current_index + num_resnet_ca] = 10
            current_index += num_resnet_ca
        if num_densenet_se > 0: 
            all_positions[current_index:current_index + num_densenet_se] = 11
            current_index += num_densenet_se
        if num_densenet_cbam > 0: 
            all_positions[current_index:current_index + num_densenet_cbam] = 12
            current_index += num_densenet_cbam
        if num_densenet_ca > 0: 
            all_positions[current_index:current_index + num_densenet_ca] = 13
            current_index += num_densenet_ca
    

        
        # Shuffle the positions randomly
        for _ in range(10):
            np.random.shuffle(all_positions)
        
        while all_positions[0] == 2:  # Pooling should not be the first unit
            np.random.shuffle(all_positions)
        """
        #print("all_positions",all_positions)


        #sabit bir model oluşturmakiçin.test amaçlı 
        # resnet    1
        # pooling   2
        # dense     3
        # inception 4
        # resnet se 8
        # resnet cbam 9
        # resnet ca 10
        
        #all_positions = [1,2,1,2,1,2,1,2,1] # resnet + pooling
        #all_positions = [8,2,8,2,8,2,8,2,8] # resnet-se + pooling
        #all_positions = [9,2,9,2,9,2,9,2,9] # resnet-cbam + pooling
        #all_positions = [10,2,10,2,10,2,10,2,10] # resnet-ca + pooling
        
        # initialize the layers based on their positions
        input_channel = self.image_channel
        reduction_ratio_list = [16] #[8,16,32]
        k_size = 3
        for i in all_positions:
            if i == 1:
                resnet = self.init_a_resnet(_number=None, _amount=None, _in_channel=input_channel, _out_channel=None)
                input_channel = resnet.out_channel
                self.units.append(resnet)
            elif i == 2:
                pool = self.init_a_pool(_number=None, _max_or_avg=None)
                self.units.append(pool)
            elif i == 3:
                densenet = self.init_a_densenet(_number=None, _amount=None, _k=None, _max_input_channel=None, _in_channel=input_channel)
                input_channel = densenet.out_channel
                self.units.append(densenet)
            elif i == 4:

                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)
                # inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type="3a", out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32)

                if chosen_type == "3a":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32)
                elif chosen_type == "3b":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64)
                elif chosen_type == "4a":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64)
                elif chosen_type == "4b":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4c":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4d":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4e":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
                elif chosen_type == "5a":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
                else:
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128)

                # print("chosen type:", chosen_type)

                inception_out = inception.out_1x1 + inception.out_3x3 + inception.out_5x5 + inception.out_1x1pool
                input_channel = inception_out
                self.units.append(inception)



            elif i == 5:  # For Inception-SE module

                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)

                if chosen_type == "3a":
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32, reduction_ratio=reduction_ratio)
                elif chosen_type == "3b":
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4a":
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4b":
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4c":
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4d":
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4e":
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                elif chosen_type == "5a":
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                else:
                    inception_se = self.init_an_inception_se(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)

                inception_out_channels = inception_se.out_1x1 + inception_se.out_3x3 + inception_se.out_5x5 + inception_se.out_1x1pool
                
                input_channel = inception_out_channels  # SE block doesn't change channel count
                self.units.append(inception_se)
            



            elif i == 6:  # For CBAM-Inception module
                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)

                if chosen_type == "3a":
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32, reduction_ratio=reduction_ratio)
                elif chosen_type == "3b":
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4a":
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4b":
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4c":
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4d":
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4e":
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                elif chosen_type == "5a":
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                else:
                    inception_cbam = self.init_an_inception_cbam(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)

                inception_out_channels = inception_cbam.out_1x1 + inception_cbam.out_3x3 + inception_cbam.out_5x5 + inception_cbam.out_1x1pool

                input_channel = inception_out_channels  # CBAM block doesn't change channel count
                self.units.append(inception_cbam)
            
            
            elif i == 7:  # For Inception-CA module
                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
            
                if chosen_type == "3a":
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32, reduction_ratio=reduction_ratio)
                elif chosen_type == "3b":
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4a":
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4b":
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4c":
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4d":
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64, reduction_ratio=reduction_ratio)
                elif chosen_type == "4e":
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                elif chosen_type == "5a":
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
                else:
                    inception_ca = self.init_an_inception_ca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128, reduction_ratio=reduction_ratio)
            
                inception_out_channels = inception_ca.out_1x1 + inception_ca.out_3x3 + inception_ca.out_5x5 + inception_ca.out_1x1pool
                
                input_channel = inception_out_channels  # CA block doesn't change channel count
                self.units.append(inception_ca)
            
                        
            # Additions to handle new module types
            
            elif i == 8:  # For SE-ResNet module
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
                
                se_resnet = self.init_a_resnet_se(_number=None, _amount=None, _in_channel=input_channel, _out_channel=None, reduction_ratio=reduction_ratio)
                input_channel = se_resnet.out_channel
                self.units.append(se_resnet)
            
            elif i == 9:  # For CBAM-ResNet module
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
                
                cbam_resnet = self.init_a_resnet_cbam(_number=None, _amount=None, _in_channel=input_channel, _out_channel=None, reduction_ratio=reduction_ratio)
                input_channel = cbam_resnet.out_channel
                self.units.append(cbam_resnet)
            
            elif i == 10:  # For CA-ResNet module
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
                
                ca_resnet = self.init_a_resnet_ca(_number=None, _amount=None, _in_channel=input_channel, _out_channel=None, reduction_ratio=reduction_ratio)
                input_channel = ca_resnet.out_channel
                self.units.append(ca_resnet)
            
            elif i == 11:  # For SE-DenseNet module
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
                
                se_densenet = self.init_a_densenet_se(_number=None, _amount=None, _k=None, _max_input_channel=None, _in_channel=input_channel, reduction_ratio=reduction_ratio)
                input_channel = se_densenet.out_channel
                self.units.append(se_densenet)
            
            elif i == 12:  # For CBAM-DenseNet module
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
                
                cbam_densenet = self.init_a_densenet_cbam(_number=None, _amount=None, _k=None, _max_input_channel=None, _in_channel=input_channel, reduction_ratio=reduction_ratio)
                input_channel = cbam_densenet.out_channel
                self.units.append(cbam_densenet)
            
            elif i == 13:  # For CA-DenseNet module
                #reduction_ratio_list = [8,16,32]
                reduction_ratio = random.choice(reduction_ratio_list)
                
                ca_densenet = self.init_a_densenet_ca(_number=None, _amount=None, _k=None, _max_input_channel=None, _in_channel=input_channel, reduction_ratio=reduction_ratio)
                input_channel = ca_densenet.out_channel
                self.units.append(ca_densenet)

            
            elif i == 14:  # For Inception-ECA module

                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)
                #reduction_ratio_list = [8,16,32]
                #reduction_ratio = random.choice(reduction_ratio_list)
                #init_an_inception_eca(self, _number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, k_size)
                if chosen_type == "3a":
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32, k_size=k_size)
                elif chosen_type == "3b":
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64, k_size=k_size)
                elif chosen_type == "4a":
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64, k_size=k_size)
                elif chosen_type == "4b":
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64, k_size=k_size)
                elif chosen_type == "4c":
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64, k_size=k_size)
                elif chosen_type == "4d":
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64, k_size=k_size)
                elif chosen_type == "4e":
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, k_size=k_size)
                elif chosen_type == "5a":
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128, k_size=k_size)
                else:
                    inception_eca = self.init_an_inception_eca(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128, k_size=k_size)

                inception_out_channels = inception_eca.out_1x1 + inception_eca.out_3x3 + inception_eca.out_5x5 + inception_eca.out_1x1pool
                
                input_channel = inception_out_channels  # SE block doesn't change channel count
                self.units.append(inception_eca)

            elif i == 15:  # For ECA-ResNet module
                
                eca_resnet = self.init_a_resnet_eca(_number=None, _amount=None, _in_channel=input_channel, _out_channel=None, k_size=k_size)
                input_channel = eca_resnet.out_channel
                self.units.append(eca_resnet)
            
            elif i == 16:  # For ECA-DenseNet module
                
                eca_densenet = self.init_a_densenet_eca(_number=None, _amount=None, _k=None, _max_input_channel=None, _in_channel=input_channel, k_size=k_size)
                input_channel = eca_densenet.out_channel
                self.units.append(eca_densenet)
            
        
    """
    Initialize a resnet layer
    """
    def init_a_resnet(self, _number, _amount, _in_channel, _out_channel):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _amount:
            amount = _amount
        else:
            amount = np.random.randint(self.min_resnet_unit, self.max_resnet_unit+1)
        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
        resnet = ResUnit(number, amount, _in_channel, out_channel)
        return resnet

    def init_a_pool(self, _number, _max_or_avg):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _max_or_avg:
            max_or_avg = _max_or_avg
        else:
            max_or_avg = np.random.rand()
        pool = PoolUnit(number, max_or_avg)
        return pool

    def init_a_densenet(self, _number, _amount, _k, _max_input_channel, _in_channel):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _k:
            k = _k;
        else:
            k = self.k_list[np.random.randint(0, len(self.k_list))]
        if _amount:
            amount = _amount
        else:
            amount_upper_limit = getattr(self, 'max_k%d'%(k))
            amount_lower_limit = getattr(self, 'min_k%d'%(k))
            amount = np.random.randint(amount_lower_limit, amount_upper_limit+1)
        if _max_input_channel:
            max_input_channel = _max_input_channel
        else:
            max_input_channel = getattr(self, 'max_k%d_input_channel'%(k))

        true_input = _in_channel
        densenet = DenseUnit(number, amount, k, max_input_channel, in_channel=_in_channel, out_channel=None)
        if true_input > densenet.max_input_channel:
            true_input = densenet.max_input_channel
        out_channel = true_input + k * amount
        densenet.out_channel = out_channel
        return densenet

    def init_an_inception(self, _number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        inception = InceptionBlock(number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool)
        return inception




    def init_an_inception_se(self, _number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        # Initialize the Inception-SE block with the provided parameters.
        inception_se = InceptionSEBlock(number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio)

        # No need to pre-calculate SE output during initialization
        # The SE block application should be handled in the forward pass of the network.

        return inception_se


    
    def init_an_inception_cbam(self, _number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        
            
        # Initialize the CBAM-Inception block with the provided parameters.
        inception_cbam = CBAMInceptionBlock(number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio)
    
        # No need to pre-calculate CBAM output during initialization
        # The CBAM block application should be handled in the forward pass of the network.
    
        return inception_cbam

    
    
    def init_an_inception_ca(self, _number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        
        # Initialize the Inception-CA block with the provided parameters.
        inception_ca = CAInceptionBlock(number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, reduction_ratio)
    
        # No need to pre-calculate CA output during initialization
        # The CA block application should be handled in the forward pass of the network.
    
        return inception_ca

    
    def init_a_resnet_se(self, _number, _amount, _in_channel, _out_channel, reduction_ratio):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _amount:
            amount = _amount
        else:
            amount = np.random.randint(self.min_resnet_unit, self.max_resnet_unit+1)
        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
            
        # Initialize the SE-ResNet unit
        resnet_se = SEResNetUnit(number, amount, _in_channel, out_channel, reduction_ratio)
        return resnet_se

    
    
    def init_a_resnet_cbam(self, _number, _amount, _in_channel, _out_channel, reduction_ratio):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _amount:
            amount = _amount
        else:
            amount = np.random.randint(self.min_resnet_unit, self.max_resnet_unit+1)
        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
                    
        # Initialize the CBAM-ResNet unit
        resnet_cbam = CBAMResNetUnit(number, amount, _in_channel, out_channel, reduction_ratio)
        return resnet_cbam
    
    
    def init_a_resnet_ca(self, _number, _amount, _in_channel, _out_channel, reduction_ratio): #reduction_ratio=16
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _amount:
            amount = _amount
        else:
            amount = np.random.randint(self.min_resnet_unit, self.max_resnet_unit+1)
        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
            
        # Initialize the CA-ResNet unit
        resnet_ca = CAResNetUnit(number, amount, _in_channel, out_channel, reduction_ratio)
        return resnet_ca
        
    def init_a_densenet_se(self, _number, _amount, _k, _max_input_channel, _in_channel, reduction_ratio):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _k:
            k = _k
        else:
            k = self.k_list[np.random.randint(0, len(self.k_list))]
        if _amount:
            amount = _amount
        else:
            amount_upper_limit = getattr(self, 'max_k%d' % (k))
            amount_lower_limit = getattr(self, 'min_k%d' % (k))
            amount = np.random.randint(amount_lower_limit, amount_upper_limit + 1)
        if _max_input_channel:
            max_input_channel = _max_input_channel
        else:
            max_input_channel = getattr(self, 'max_k%d_input_channel' % (k))
    
        true_input = _in_channel
        if true_input > max_input_channel:  # `densenet_se.max_input_channel` yerine `max_input_channel` kullanıldı
            true_input = max_input_channel
    
        out_channel = true_input + k * amount

            
        # SE-DenseNet unit oluşturuluyor
        densenet_se = SEDenseNetUnit(
            number,
            amount,
            k,
            max_input_channel,
            in_channel=_in_channel,
            out_channel=out_channel,
            reduction_ratio=reduction_ratio
        )
    
        return densenet_se
    
     
     
    def init_a_densenet_cbam(self, _number, _amount, _k, _max_input_channel, _in_channel, reduction_ratio):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _k:
            k = _k
        else:
            k = self.k_list[np.random.randint(0, len(self.k_list))]
        if _amount:
            amount = _amount
        else:
            amount_upper_limit = getattr(self, 'max_k%d'%(k))
            amount_lower_limit = getattr(self, 'min_k%d'%(k))
            amount = np.random.randint(amount_lower_limit, amount_upper_limit+1)
        if _max_input_channel:
            max_input_channel = _max_input_channel
        else:
            max_input_channel = getattr(self, 'max_k%d_input_channel'%(k))
    
        true_input = _in_channel
        if true_input > max_input_channel:
            true_input = max_input_channel
    
        # Eğer out_channel None ise, doğru bir şekilde hesaplayın.
        out_channel = true_input + k * amount  # out_channel değeri burada tanımlanıyor
        
        densenet_cbam = CBAMDenseNetUnit(
            number,
            amount,
            k,
            max_input_channel,
            in_channel=_in_channel,
            out_channel=out_channel,  # Artık out_channel tanımlı
            reduction_ratio=reduction_ratio
        )
        
        return densenet_cbam
    
    def init_a_densenet_ca(self, _number, _amount, _k, _max_input_channel, _in_channel, reduction_ratio):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _k:
            k = _k
        else:
            k = self.k_list[np.random.randint(0, len(self.k_list))]
        if _amount:
            amount = _amount
        else:
            amount_upper_limit = getattr(self, 'max_k%d'%(k))
            amount_lower_limit = getattr(self, 'min_k%d'%(k))
            amount = np.random.randint(amount_lower_limit, amount_upper_limit+1)
        if _max_input_channel:
            max_input_channel = _max_input_channel
        else:
            max_input_channel = getattr(self, 'max_k%d_input_channel'%(k))
    
        true_input = _in_channel
        if true_input > max_input_channel:
            true_input = max_input_channel
    
        # `out_channel` değeri burada hesaplanıyor
        out_channel = true_input + k * amount

        # `out_channel` değeri artık CADenseNetUnit'e atanıyor
        densenet_ca = CADenseNetUnit(
            number,
            amount,
            k,
            max_input_channel,
            in_channel=_in_channel,
            out_channel=out_channel,  # Düzeltme yapıldı
            reduction_ratio=reduction_ratio
        )
        
        return densenet_ca


    def init_an_inception_eca(self, _number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, k_size):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
    
        # ECA destekli Inception unit'ini oluştur
        inception_eca = ECAInceptionBlock(number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool, k_size)
    
        return inception_eca
    

    def init_a_resnet_eca(self, _number, _amount, _in_channel, _out_channel, k_size):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
    
        if _amount:
            amount = _amount
        else:
            amount = np.random.randint(self.min_resnet_unit, self.max_resnet_unit + 1)
    
        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
    
        # ECA-ResNet unit oluşturuluyor
        resnet_eca = ECAResNetUnit(number, amount, _in_channel, out_channel, k_size)
    
        return resnet_eca


    def init_a_densenet_eca(self, _number, _amount, _k, _max_input_channel, _in_channel, k_size):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
    
        if _k:
            k = _k
        else:
            k = self.k_list[np.random.randint(0, len(self.k_list))]
    
        if _amount:
            amount = _amount
        else:
            amount_upper_limit = getattr(self, 'max_k%d' % (k))
            amount_lower_limit = getattr(self, 'min_k%d' % (k))
            amount = np.random.randint(amount_lower_limit, amount_upper_limit + 1)
    
        if _max_input_channel:
            max_input_channel = _max_input_channel
        else:
            max_input_channel = getattr(self, 'max_k%d_input_channel' % (k))
    
        true_input = _in_channel
        if true_input > max_input_channel:
            true_input = max_input_channel
    
        out_channel = true_input + k * amount
    
        # ECA-DenseNet unit oluşturuluyor
        densenet_eca = ECADenseNetUnit(
            number=number,
            amount=amount,
            k=k,
            max_input_channel=max_input_channel,
            in_channel=_in_channel,
            out_channel=out_channel,
            k_size=k_size
        )
    
        return densenet_eca

    
    def uuid(self):
        _str = []
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d' % (unit.number))
                _pool_type = 0.25 if unit.max_or_avg < 0.5 else 0.75
                _sub_str.append('type:%.2f' % (_pool_type))
            if unit.type == 3:
                _sub_str.append('densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                true_in_channel = unit.in_channel
                if true_in_channel > unit.max_input_channel:
                    true_in_channel = unit.max_input_channel
                _sub_str.append('in:%d' % (true_in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
            if unit.type == 4:
                _sub_str.append('inception')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))

            
            if unit.type == 5:  # Inception-SE Block
                _sub_str.append('inception-se')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            
            if unit.type == 6:  # CBAM-Inception Block
                _sub_str.append('inception-cbam')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            if unit.type == 7:  # Inception-CA Block
                _sub_str.append('inception-ca')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            
            if unit.type == 8:  # SE-ResNet Block
                _sub_str.append('se-resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            if unit.type == 9:  # CBAM-ResNet Block
                _sub_str.append('cbam-resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            if unit.type == 10:  # CA-ResNet Block
                _sub_str.append('ca-resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))

            if unit.type == 11:  # SE-DenseNet Block
                _sub_str.append('se-densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                true_in_channel = unit.in_channel
                if true_in_channel > unit.max_input_channel:
                    true_in_channel = unit.max_input_channel
                _sub_str.append('in:%d' % (true_in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))     
            if unit.type == 12:  # CBAM-DenseNet Block
                _sub_str.append('cbam-densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                true_in_channel = unit.in_channel
                if true_in_channel > unit.max_input_channel:
                    true_in_channel = unit.max_input_channel
                _sub_str.append('in:%d' % (true_in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))  
            if unit.type == 13:  # CA-DenseNet Block
                _sub_str.append('ca-densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                true_in_channel = unit.in_channel
                if true_in_channel > unit.max_input_channel:
                    true_in_channel = unit.max_input_channel
                _sub_str.append('in:%d' % (true_in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))

            if unit.type == 14:  # Inception-ECA Block
                _sub_str.append('inception-eca')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
                _sub_str.append('k_size:%d' % (unit.k_size))

            if unit.type == 15:  # ECA-ResNet Block
                _sub_str.append('eca-resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('k_size:%d' % (unit.k_size))

            if unit.type == 16:  # ECA-DenseNet Block
                _sub_str.append('eca-densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                true_in_channel = unit.in_channel
                if true_in_channel > unit.max_input_channel:
                    true_in_channel = unit.max_input_channel
                _sub_str.append('in:%d' % (true_in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('k_size:%d' % (unit.k_size))

                        
            _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
        _final_str_ = '-'.join(_str)
        _final_utf8_str_ = _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_


    
    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Acc:%.5f' % (self.acc))
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
    
            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('type:%.1f' % (unit.max_or_avg))
    
            if unit.type == 3:
                _sub_str.append('densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
    
            if unit.type == 4:
                _sub_str.append('inception')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
    
            if unit.type == 5:  # SE-Inception Block
                _sub_str.append('inception-se')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            
            if unit.type == 6:  # CBAM-Inception Block
                _sub_str.append('inception-cbam')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            
            if unit.type == 7:  # Inception-CA Block
                _sub_str.append('inception-ca')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))    
                
                
                # Additions to handle new module types
            
            if unit.type == 8:  # SE-ResNet Block
                _sub_str.append('se-resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))

            
            if unit.type == 9:  # CBAM-ResNet Block
                _sub_str.append('cbam-resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            
            if unit.type == 10:  # CA-ResNet Block
                _sub_str.append('ca-resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            
            if unit.type == 11:  # SE-DenseNet Block
                _sub_str.append('se-densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            
            if unit.type == 12:  # CBAM-DenseNet Block
                _sub_str.append('cbam-densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))
            
            if unit.type == 13:  # CA-DenseNet Block
                _sub_str.append('ca-densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('reduction_ratio:%d' % (unit.reduction_ratio))


            if unit.type == 14:  # Inception-ECA Block
                _sub_str.append('inception-eca')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_1x1pool + unit.out_1x1 + unit.out_3x3 + unit.out_5x5))
                _sub_str.append('type:%s' % (unit.inception_type))
                _sub_str.append('k_size:%d' % (unit.k_size))


            if unit.type == 15:  # ECA-ResNet Block
                _sub_str.append('eca-resnet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('in:%d' % (unit.in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('k_size:%d' % (unit.k_size))

            if unit.type == 16:  # ECA-DenseNet Block
                _sub_str.append('eca-densenet')
                _sub_str.append('number:%d' % (unit.number))
                _sub_str.append('amount:%d' % (unit.amount))
                _sub_str.append('k:%d' % (unit.k))
                true_in_channel = unit.in_channel
                if true_in_channel > unit.max_input_channel:
                    true_in_channel = unit.max_input_channel
                _sub_str.append('in:%d' % (true_in_channel))
                _sub_str.append('out:%d' % (unit.out_channel))
                _sub_str.append('k_size:%d' % (unit.k_size))

            _str.append('%s%s%s' % ('[', ','.join(_sub_str), ']'))
        return '\n'.join(_str)
    
    



class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0 # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []
        from utils import StatusUpdateTool

        self.horovod_enabled = StatusUpdateTool.is_horovod_enabled()
        if self.horovod_enabled:
            #hvd.init()
            self.rank = hvd.rank()
            self.size = hvd.size()
        else:
            self.rank = 0
            self.size = 1

    def initialize(self):
        if (not self.horovod_enabled) or (self.rank == 0):
            dummy_indi = Individual(self.params, indi_no="temp")
            raw_pop, name2id = dummy_indi.population_with_all_positions(pop_size=self.pop_size, seed=123)
            final_pop = Individual.balance_population_arrays(raw_pop, name2id, tolerance=1)
            #for _ in range(self.pop_size):
            for indi_index in range(self.pop_size):
                indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
                self.number_id += 1
                indi = Individual(self.params, indi_no)
                indi.initialize(final_pop[indi_index])
                self.individuals.append(indi)
                
        # Tüm rank'lar bariyerde buluşsun.
        if self.horovod_enabled:
            hvd.barrier()

            """
            # 1) Rank 0 tarafında self.individuals'ı pickle'a çevir.
            if self.rank == 0:
                pickled_data = pickle.dumps(self.individuals, protocol=pickle.HIGHEST_PROTOCOL)
                data_size = torch.IntTensor([len(pickled_data)])
            else:
                # Rank != 0 henüz data yok
                data_size = torch.IntTensor([0])

            # 2) data_size bilgisini broadcast et, her rank kaç byte alacağını bilsin
            data_size = hvd.broadcast(data_size, root_rank=0)

            # 3) Rank != 0: gelen boyut kadar boş byte alanı hazırla
            if self.rank != 0:
                pickled_data = bytearray(data_size.item())

            # 4) Bu byte array'i ByteTensor'a dönüştür
            if self.rank == 0:
                pickled_tensor = torch.ByteTensor(list(pickled_data))
            else:
                pickled_tensor = torch.empty(data_size.item(), dtype=torch.uint8)

            # 5) Asıl veriyi broadcast et
            pickled_tensor = hvd.broadcast(pickled_tensor, root_rank=0)

            # 6) Her rank, gelen tensörü unpickle ile orijinal listeye çevirir
            np_data = pickled_tensor.cpu().numpy()
            unpickled_bytes = np_data.tobytes()
            
            
            #if self.rank != 0:
                # Rank 0 zaten kendi 'individuals' listesini oluşturmuştu.
            self.individuals = pickle.loads(unpickled_bytes)
            # Eğer tüm rank’lar tamamen aynı individuals listesine sahip olsun diyorsanız,
            # rank 0 için de "self.individuals = pickle.loads(unpickled_bytes)" yapabilirsiniz.
            """
           
            
    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)


    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-'*100)
        return '\n'.join(_str)






def test_individual(params):
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())

def test_population(params):
    pop = Population(params, 0)
    pop.initialize()
    print(pop)



#if __name__ == '__main__':
#     params = StatusUpdateTool.get_init_params()
#     test_individual(params)
#     test_population(params)
#     print("hello")

