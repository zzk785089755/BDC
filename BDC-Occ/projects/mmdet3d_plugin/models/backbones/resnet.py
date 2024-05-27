# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
from ..model_utils.quant_tools.quant_convs_ReActNet import (BinaryConv2dReActNet, BasicBinaryBlockReActNet)
from ..model_utils.quant_tools.quant_convs_BiSRNet import (BinaryConv2dBiSRNet, BasicBinaryBlockBiSRNet,
                                                                        DownsampleBinaryBlockBiSRNet, BinaryConv2dBiSRNet_Down,)
from ..model_utils.quant_tools.quant_convs_BBCU import (BinaryConv2dBBCU, BasicBinaryBlockBBCU,
                                                                        DownsampleBinaryBlockBBCU, BinaryConv2dBBCU_Down,
                                                                        )
from ..model_utils.quant_tools.quant_convs_BiMatting import BinaryConv2dBiMatting
from ..model_utils.quant_tools.quant_convs_BDC import (BinaryConv2dBDC, BasicBinaryBlockBDC, DownsampleBinaryBlockBDC, BinaryConv2dBDC_Down,)


@BACKBONES.register_module()
class CustomResNet(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 在第一个block中对输入进行downsample
                layer = [Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=1,
                                         downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 在第一个block中对输入进行downsample
                layer = [BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=1,
                                          downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False

        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class BasicBlock3D(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)


@BACKBONES.register_module()
class CustomResNet3D(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(CustomResNet3D, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock3D(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    downsample=ConvModule(
                        curr_numC,
                        num_channels[i],
                        kernel_size=3,
                        stride=stride[i],
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d', ),
                        act_cfg=None))
            ]
            curr_numC = num_channels[i]
            layer.extend([
                BasicBlock3D(curr_numC, curr_numC)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C, Dz, Dy, Dx)
        Returns:
            feats: List[
                (B, C, Dz, Dy, Dx),
                (B, 2C, Dz/2, Dy/2, Dx/2),
                (B, 4C, Dz/4, Dy/4, Dx/4),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

#---------------------------------------------------Occ Module-----------------------------------------------------------------------------

@BACKBONES.register_module()
class BinaryResNetReActNet(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(BinaryResNetReActNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []

        curr_numC = numC_input
        for i in range(len(num_layer)):
            # 在第一个block中对输入进行downsample
            layer = [BasicBinaryBlockReActNet(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
                                downsample=BinaryConv2dReActNet(curr_numC, num_channels[i], 3, stride[i], 1)
                                # downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1)   
                                )]
            curr_numC = num_channels[i]
            layer.extend([BasicBinaryBlockReActNet(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                                        downsample=None) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))


        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class BinaryResNetBiSRNet(nn.Module):
    '''
        BiSRNet
    '''
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(BinaryResNetBiSRNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []

        curr_numC = numC_input
        for i in range(len(num_layer)):
            # 在第一个block中对输入进行downsample
            layer = [BasicBinaryBlockBiSRNet(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
                                downsample=BinaryConv2dBiSRNet(curr_numC, num_channels[i], 3, stride[i], 1)
                                )]
            curr_numC = num_channels[i]
            layer.extend([BasicBinaryBlockBiSRNet(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                                        downsample=None) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))


        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class BinaryResNetBiSRNetDown(nn.Module):
    '''
        BiSRNet
    '''
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(BinaryResNetBiSRNetDown, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []

        curr_numC = numC_input
        for i in range(len(num_layer)):
            # 在第一个block中对输入进行downsample
            # layer = [DownsampleBinaryBlockBiSRNet(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
            #                     downsample=BinaryConv2dBiSRNet_Down(curr_numC, num_channels[i], 3, stride[i], 1)
            #                     )]
            if i == 0:
                layer = [BasicBinaryBlockBiSRNet(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
                    downsample=BinaryConv2dBiSRNet(curr_numC, num_channels[i], 3, stride[i], 1)
                    )]
            else:
                layer = [DownsampleBinaryBlockBiSRNet(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                        downsample=BinaryConv2dBiSRNet_Down(curr_numC, num_channels[i], 3, 1, 1)
                        )]
            curr_numC = num_channels[i]
            layer.extend([BasicBinaryBlockBiSRNet(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                                        downsample=None) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))


        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class BinaryResNetBBCU(nn.Module):
    '''
        BBCU
    '''
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(BinaryResNetBBCU, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []

        curr_numC = numC_input
        for i in range(len(num_layer)):
            # 在第一个block中对输入进行downsample
            layer = [BasicBinaryBlockBBCU(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
                                downsample=BinaryConv2dBBCU(curr_numC, num_channels[i], 3, stride[i], 1)
                                )]
            curr_numC = num_channels[i]
            layer.extend([BasicBinaryBlockBBCU(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                                        downsample=None) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))


        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class BinaryResNetBBCUDown(nn.Module):
    '''
        BBCU
    '''
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(BinaryResNetBBCUDown, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []

        curr_numC = numC_input
        for i in range(len(num_layer)):
            if i == 0:
                layer = [BasicBinaryBlockBBCU(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
                    downsample=BinaryConv2dBBCU(curr_numC, num_channels[i], 3, stride[i], 1)
                    )]
            else:
                layer = [DownsampleBinaryBlockBBCU(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                        downsample=BinaryConv2dBBCU_Down(curr_numC, num_channels[i], 3, 1, 1)
                        )]
            curr_numC = num_channels[i]
            layer.extend([BasicBinaryBlockBBCU(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                                        downsample=None) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))


        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class BinaryResNetBiMatting(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(BinaryResNetBiMatting, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []

        curr_numC = numC_input
        for i in range(len(num_layer)):
            # 在第一个block中对输入进行downsample
            layer = [BinaryConv2dBiMatting(inplanes=curr_numC, planes=num_channels[i], stride=stride[i]
                                )]
            curr_numC = num_channels[i]
            layer.extend([BinaryConv2dBiMatting(inplanes=curr_numC, planes=num_channels[i], stride=1
                                                ) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))


        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class BinaryResNetBDC(nn.Module):
    '''
        BDC
    '''
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
            with_bn=False,
    ):
        super(BinaryResNetBDC, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []

        curr_numC = numC_input
        for i in range(len(num_layer)):
            # 在第一个block中对输入进行downsample
            layer = [BasicBinaryBlockBDC(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
                                downsample=BinaryConv2dBDC(curr_numC, num_channels[i], 3, stride[i], 1, with_bn=with_bn),
                                with_bn=with_bn
                                )]
            curr_numC = num_channels[i]
            layer.extend([BasicBinaryBlockBDC(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                                        downsample=None, with_bn=with_bn) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))


        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class BinaryResNetBDCDown(nn.Module):
    '''
        BDC_Down
    '''
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
            with_bn=False,
    ):
        super(BinaryResNetBDCDown, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        layers = []

        curr_numC = numC_input
        for i in range(len(num_layer)):
            # 在第一个block中对输入进行downsample
            # layer = [DownsampleBinaryBlockBiSRNet(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
            #                     downsample=BinaryConv2dBiSRNet_Down(curr_numC, num_channels[i], 3, stride[i], 1)
            #                     )]
            if i == 0:
                # if curr_numC == num_channels[i]:
                #     downsample_module = 
                # else:
                #     downsample_module = BinaryConv2dBiSRNet_Down(curr_numC, num_channels[i], 3, stride[i], 1)
                layer = [BasicBinaryBlockBDC(channels_in=curr_numC, channels_out=num_channels[i], stride=stride[i],
                    downsample=BinaryConv2dBDC(curr_numC, num_channels[i], 3, stride[i], 1),
                    with_bn=with_bn
                    )]
            else:
                layer = [DownsampleBinaryBlockBDC(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                        downsample=BinaryConv2dBDC_Down(curr_numC, num_channels[i], 3, 1, 1),
                        with_bn=with_bn
                        )]
            curr_numC = num_channels[i]
            layer.extend([BasicBinaryBlockBDC(channels_in=curr_numC, channels_out=num_channels[i], stride=1,
                                        downsample=None, with_bn=with_bn) for _ in range(num_layer[i] - 1)])
            layers.append(nn.Sequential(*layer))


        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
