import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import os
from pdb import set_trace as stx
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ['cuda_visible_device']='2'
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# --------------------------------------------- Binarized Basic Units -----------------------------------------------------------------

def binaryconv3x3(in_planes, out_planes, stride=1,groups=1,bias=False):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,groups=groups,bias=bias)

def binaryconv1x1(in_planes, out_planes, stride=1,groups=1,bias=False):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,groups=groups,bias=bias)

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x
    
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        # stx()
        out = x + self.bias.expand_as(x)
        return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        cliped_ac = torch.clamp(x, -1.0, 1.0)
        out = out_forward.detach() - cliped_ac.detach() + cliped_ac

        return out
    
class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,groups=1, bias=True):
        super(HardBinaryConv, self).__init__(in_chn,
            out_chn,
            kernel_size,stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)   
  
    def forward(self, x):

        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, self.bias,stride=self.stride, padding=self.padding, groups=self.groups)

        return y

# --------------------------------------------- Binarized Conv Units -----------------------------------------------------------------

class BinaryConv2dBBCU(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, groups=1, bias=False):
        super(BinaryConv2dBBCU, self).__init__()
        self.in_channels = inplanes
        self.out_channels = planes
        if kernel_size == 3:
            self.binary_conv = binaryconv3x3(inplanes, planes, stride=stride, groups=groups,bias=bias)
        elif kernel_size == 1:
            self.binary_conv= binaryconv1x1(inplanes, planes, stride=stride, groups=groups,bias=bias)
        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.relu=RPReLU(planes)
        if stride == 2:
            self.with_downsample = True
        else:
            self.with_downsample = False
        if self.with_downsample:
            self.downsample = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        if self.with_downsample:
            identity = self.downsample(x)
        else:
            identity = x
        if self.out_channels == self.in_channels * 2:
            identity = torch.cat([identity, identity], dim=1)
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.relu(out)
        out = out + identity
        return out      
           

class BinaryConv2dBBCU_Fusion_Decrease(nn.Module):
    '''
    空间尺寸不变且通道数减半 - Upsample 去掉上采样
    input: b,c,h,w
    output: b,c/2,h,w
    '''
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,groups=1,bias=False):
        super(BinaryConv2dBBCU_Fusion_Decrease, self).__init__()

        self.binary_1x1= binaryconv1x1(inplanes, planes, stride=stride,groups=groups,bias=bias)
        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.relu=RPReLU(planes)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_1x1(out)
        out =self.relu(out)

        return out       

    
class BinaryConv2dBBCU_Fusion_Increase(nn.Module):
    '''
    空间尺寸不变且通道数翻倍 - Downsample 去掉下采样
    input: b,c,h,w
    output: b,2c,h,w
    '''
    def __init__(self, inplanes, planes, stride=1, groups=1,bias=False):
        super(BinaryConv2dBBCU_Fusion_Increase, self).__init__()

        self.binary_1x1= binaryconv1x1(inplanes, planes, stride=stride,groups=groups,bias=bias)
        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.relu=RPReLU(planes)


    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_1x1(out)
        out =self.relu(out)

        return out  


class BinaryConv2dBBCU_Down(nn.Module):
    '''
    降采样且通道数翻倍
    input: b,c,h,w
    output: b,2c,h/2,w/2
    '''
    def __init__(self, inplanes, planes, kernel_size=3, stride=1,groups=1,bias=False):
        super(BinaryConv2dBBCU_Down, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.binary_3x3= binaryconv3x3(inplanes, planes, stride=stride,groups=groups,bias=bias)
        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.relu=RPReLU(planes)


    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''
        out = self.avg_pool(x)

        out = self.move0(out)
        out = self.binary_activation(out)
        out = self.binary_3x3(out)
        out = self.relu(out)

        return out      

    
class BinaryConv2dBBCU_Up(nn.Module):
    '''
    上采样且通道数减半
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False,groups=1):
        super(BinaryConv2dBBCU_Up, self).__init__()

        self.binary_3x3= binaryconv3x3(inplanes, planes, stride=stride,groups=groups,bias=bias)
        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.relu=RPReLU(planes)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,2h,2w
        '''
        b,c,h,w = x.shape
        out = F.interpolate(x, scale_factor=2, mode='bilinear')

        out = self.move0(out)
        out = self.binary_activation(out)
        out = self.binary_3x3(out)
        out = self.relu(out)

        return out      


class BinaryConv2dBBCU_Down_Constant(nn.Module):
    '''
    降采样且通道数翻倍
    input: b,c,h,w
    output: b,c,h/2,w/2
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False):
        super(BinaryConv2dBBCU_Down_Constant, self).__init__()

        self.biconv_1 = BinaryConv2dBBCU(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn)
        self.biconv_2 = BinaryConv2dBBCU(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''
        out = self.avg_pool(x)
        out_1 = out
        out_2 = out_1.clone()
        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        out = out_1 + out_2

        return out
    


    
# --------------------------------------------- Binarized Block Units -----------------------------------------------------------------

class BasicBinaryBlockBBCU(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBinaryBlockBBCU, self).__init__()
        self.conv1 = BinaryConv2dBBCU(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            bias=False,
            )
        self.conv2 = BinaryConv2dBBCU(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            bias=False,
            )
        self.downsample = downsample
        self.relu = RPReLU(channels_out)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)
    

class DownsampleBinaryBlockBBCU(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(DownsampleBinaryBlockBBCU, self).__init__()
        self.conv1 = BinaryConv2dBBCU_Down(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            bias=False,
            )
        self.conv2 = BinaryConv2dBBCU(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            bias=False,
            )
        self.downsample = downsample
        self.relu = RPReLU(channels_out)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)
    

