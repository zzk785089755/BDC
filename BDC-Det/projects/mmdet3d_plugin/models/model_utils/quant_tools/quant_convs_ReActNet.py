import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        '''
            gradient approximation
            (https://github.com/liuzechun/Bi-Real-net/blob/master/pytorch_implementation/BiReal18_34/birealnet.py)
        '''
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        # real_weights.shape is (out_channel, in_channel, kh, kw)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights,self.bias, stride=self.stride, padding=self.padding)

        return y


#---------------------------------------------------Occ Module-----------------------------------------------------------------------------

class BinaryConv2dReActNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False, with_norm=False):
        super(BinaryConv2dReActNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.move0 = LearnableBias(in_channels)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size//2) if padding is None else padding,
            bias=bias
        )
        self.relu=RPReLU(out_channels)
        self.with_norm = with_norm
        if self.with_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
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
        if self.with_norm:
            out = self.batch_norm(out)
        out = out + identity
        out = self.relu(out)
        return out


# --------------------------------------------- Binarized Block Units -----------------------------------------------------------------

class BasicBinaryBlockReActNet(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBinaryBlockReActNet, self).__init__()
        self.conv1 = BinaryConv2dReActNet(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            with_norm=True,
            )
        self.conv2 = BinaryConv2dReActNet(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            with_norm=True,
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
    

    

    