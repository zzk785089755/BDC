import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import os
from pdb import set_trace as stx

# --------------------------------------------- Binarized Basic Units -----------------------------------------------------------------

class BiConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BiConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, \
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        weight = self.weight
        activation = input
        weight = weight - torch.mean(torch.mean(torch.mean(weight,dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weight_no_grad = scaling_factor * torch.sign(weight)
        cliped_weight = torch.clamp(weight, -1.0, 1.0)
        binary_weight = binary_weight_no_grad.detach() - cliped_weight.detach() + cliped_weight
        output = F.conv2d(activation, binary_weight, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return BiConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return BiConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        cliped_out = torch.clamp(x, -1.0, 1.0)
        binary_out = out_forward.detach() - cliped_out.detach() + cliped_out
        out = out_forward.detach() - binary_out.detach() + binary_out
        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

# --------------------------------------------- Binarized Conv Units -----------------------------------------------------------------

class BinaryConv2dBiMatting(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BinaryConv2dBiMatting, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3= conv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = conv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)
        elif 2 * inplanes == planes:
            self.binary_pw_down1 = conv1x1(inplanes, inplanes)
            self.binary_pw_down2 = conv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)
        elif 4 * inplanes == planes:
            self.binary_pw_down1 = conv1x1(inplanes, inplanes)
            self.binary_pw_down2 = conv1x1(inplanes, inplanes)
            self.binary_pw_down3 = conv1x1(inplanes, inplanes)
            self.binary_pw_down4 = conv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)
            self.bn2_3 = norm_layer(inplanes)
            self.bn2_4 = norm_layer(inplanes)
        else:
            self.binary_pw_down1 = conv1x1(inplanes, planes)
            self.bn2_1 = norm_layer(planes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2,2,ceil_mode=True)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_activation(out1)
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binary_activation(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1
        elif self.inplanes * 2 == self.planes:
            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
        elif self.planes == self.inplanes * 4:
            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_3 = self.binary_pw_down3(out2)
            out2_4 = self.binary_pw_down4(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_3 = self.bn2_3(out2_3)
            out2_4 = self.bn2_4(out2_4)
            out2_1 += out1
            out2_2 += out1
            out2_3 += out1
            out2_4 += out1
            out2 = torch.cat([out2_1, out2_2, out2_3, out2_4], dim=1)
        else:
            assert self.planes == self.inplanes // 2
            split_result = torch.split(out1, self.planes, dim = 1)
            split_result = sum(split_result) / 2
            out2 = self.binary_pw_down1(out2)
            out2 = self.bn2_1(out2)
            out2 += split_result

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2
