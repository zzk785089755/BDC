import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import os
from torch.utils.checkpoint import checkpoint
from pdb import set_trace as stx
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ['cuda_visible_device']='2'
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# --------------------------------------------- Binarized Basic Units -----------------------------------------------------------------
def binaryconv3x3(in_planes, out_planes, stride=1,groups=1, bias=False):
    """3x3 convolution with padding"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)


def binaryconv1x1(in_planes, out_planes, stride=1,groups=1, bias=False):
    """1x1 convolution"""
    return HardBinaryConv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, groups=groups, bias=bias)


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        # stx()
        out = x + self.bias.expand_as(x)
        return out


class ReDistribution(nn.Module):
    def __init__(self, out_chn):
        super(ReDistribution, self).__init__()
        self.b = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        self.k = nn.Parameter(torch.ones(1,out_chn,1,1), requires_grad=True)
    
    def forward(self, x):
        out = x * self.k.expand_as(x) + self.b.expand_as(x)
        return out


class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))      #为什么要反复设置可学习偏置？
        return x


class Spectral_Binary_Activation(nn.Module):
    def __init__(self):
        super(Spectral_Binary_Activation, self).__init__()
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(x),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # scaling_factor = scaling_factor.detach()

        binary_activation_no_grad = torch.sign(x)
        tanh_activation = torch.tanh(x*self.beta)
        
        out = binary_activation_no_grad.detach() - tanh_activation.detach() + tanh_activation  #.detach() 不需要计算其梯度，不具有梯度grad，此处为何是这样设计BinaryActivation?

        return out


class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        # stx()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)   #用torch.sign()函数进行二值化
        # stx()
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights,self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

        return y



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, with_cp = False):
        super(SEBlock, self).__init__()
        self.with_cp = with_cp
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            binaryconv1x1(in_planes=in_channels, out_planes=in_channels//reduction, stride=1, bias=False),
            RPReLU(in_channels),    
            binaryconv1x1(in_planes=in_channels//reduction, out_planes=in_channels, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        if self.with_cp:
            y = checkpoint(self.fc, y)
        else:
            y = self.fc(y)
        return x * y
    
# --------------------------------------------- Binarized Conv Units -----------------------------------------------------------------

class BinaryConv2dBDC(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC, self).__init__()
        self.with_bn = with_bn
        self.with_cp = with_cp
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.move0 = ReDistribution(in_channels)
        self.binary_activation = Spectral_Binary_Activation()
        self.binary_conv_relu = nn.Sequential(
            binaryconv3x3(in_planes=in_channels, out_planes=in_channels, stride=stride, bias=bias, groups=groups),
            nn.BatchNorm2d(in_channels) if self.with_bn else nn.Identity(),
            RPReLU(in_channels),    
            binaryconv1x1(in_planes=in_channels, out_planes=out_channels, stride=1, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels) if self.with_bn else nn.Identity(),
            RPReLU(out_channels),    
        )

        # SE layer
        self.se_block = SEBlock(out_channels, reduction=1, with_cp=with_cp)

        if self.stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        if self.with_bn:
            self.bn_fp = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.stride == 2:
            identity = self.downsample(x)
        else:
            identity = x
        if self.out_channels == self.in_channels * 2:
            identity = torch.cat([identity, identity], dim=1)
        out = self.move0(x)
        out = self.binary_activation(out)
        if self.with_cp:
            out = checkpoint(self.binary_conv_relu, out)
        else:
            out = self.binary_conv_relu(out)
        # Apply SE Block
        out = self.se_block(out)
        if self.with_bn:
            out = out + self.bn_fp(identity)
        else:
            out = out + identity
        return out



class BinaryConv2dBDC_Down(nn.Module):
    '''
    降采样且通道数翻倍
    input: b,c,h,w
    output: b,2c,h/2,w/2
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Down, self).__init__()
        self.with_cp = with_cp
        self.biconv_1 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
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
        out = torch.cat([out_1, out_2], dim=1)

        return out


class BinaryConv2dBDC_Up(nn.Module):
    '''
    上采样且通道数不变
    input: b,c,h,w
    output: b,c,2h,2w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Up, self).__init__()
        self.with_cp=with_cp
        bi_channels = out_channels // 2
        self.biconv_1 = BinaryConv2dBDC(bi_channels, bi_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(bi_channels, bi_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        # self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c,2h,2w
        '''
        b,c,h,w = x.shape
        out = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        out_1 = out[:,:c//2,:,:]
        out_2 = out[:,c//2:,:,:]

        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)

        out = torch.cat([out_1, out_2], dim=1)  # output: b,c,2h,2w
        # out = (out_1 + out_2) / 2 # output: b,c/2,2h,2w

        return out


class BinaryConv2dBDC_Fusion_Decrease(nn.Module):
    '''
    空间尺寸不变且通道数减半 - Upsample 去掉上采样
    input: b,c,h,w
    output: b,c/2,h,w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Fusion_Decrease, self).__init__()
        self.with_cp=with_cp
        self.biconv_1 = BinaryConv2dBDC(out_channels, out_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(out_channels, out_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        # self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,h,w
        '''
        b,c,h,w = x.shape
        out = x
        
        out_1 = out[:,:c//2,:,:]
        out_2 = out[:,c//2:,:,:]

        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        
        out = (out_1 + out_2) / 2

        return out


class BinaryConv2dBDC_Fusion_Increase(nn.Module):
    '''
    空间尺寸不变且通道数翻倍 - Downsample 去掉下采样
    input: b,c,h,w
    output: b,2c,h,w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Fusion_Increase, self).__init__()
        self.with_cp=with_cp
        self.biconv_1 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        # self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''
        # stx()
        out_1 = x
        out_2 = out_1.clone()
        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        out = torch.cat([out_1, out_2], dim=1)

        return out
    

class BinaryConv2dBDC_Down_Constant(nn.Module):
    '''
    降采样且通道数翻倍
    input: b,c,h,w
    output: b,c,h/2,w/2
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, with_bn=False, with_cp=False):
        super(BinaryConv2dBDC_Down_Constant, self).__init__()
        self.with_cp=with_cp
        self.biconv_1 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
        self.biconv_2 = BinaryConv2dBDC(in_channels, in_channels, kernel_size, stride, padding, bias, groups, with_bn=with_bn, with_cp=with_cp)
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

class BasicBinaryBlockBDC(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None, with_bn=False, with_cp=False):
        super(BasicBinaryBlockBDC, self).__init__()
        self.conv1 = BinaryConv2dBDC(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            with_bn=with_bn, 
            with_cp=with_cp
            )
        self.conv2 = BinaryConv2dBDC(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            with_bn=with_bn, 
            with_cp=with_cp
            )
        self.downsample = downsample
        self.relu = RPReLU(channels_out)
        self.with_cp=with_cp

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)
    

class DownsampleBinaryBlockBDC(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None, with_bn=False, with_cp=False):
        super(DownsampleBinaryBlockBDC, self).__init__()
        self.conv1 = BinaryConv2dBDC_Down(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            with_bn=with_bn,
            with_cp=with_cp
            )
        self.conv2 = BinaryConv2dBDC(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            with_bn=with_bn, 
            with_cp=with_cp
            )
        self.downsample = downsample
        self.relu = RPReLU(channels_out)
        self.with_cp=with_cp

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)
    