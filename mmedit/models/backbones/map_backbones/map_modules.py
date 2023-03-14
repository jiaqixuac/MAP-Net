import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmedit.models.common import PixelShufflePack, ResidualBlockNoBN, make_layer
from mmedit.models.registry import COMPONENTS


@COMPONENTS.register_module()
class ProjectionHead(BaseModule):
    """Projection head.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        outs = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        return tuple(outs)


@COMPONENTS.register_module()
class MAPUpsampler(BaseModule):
    """The upsampler for MAP.
    https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py
    https://github.com/open-mmlab/mmediting/blob/master/mmedit/models/backbones/sr_backbones/basicvsr_net.py

    Args:
        embed_dim: feature dim from decoder
        num_feat: intermediate feature dim
    """

    def __init__(self,
                 upscale=4,
                 embed_dim=64,
                 num_feat=64,
                 num_out_ch=3):
        super().__init__()
        assert upscale == 4

        self.conv_before_upsample = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
        self.upsample1 = PixelShufflePack(
            num_feat, num_feat, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            num_feat, num_feat, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        nn.init.constant_(self.conv_last.weight, 0)
        nn.init.constant_(self.conv_last.bias, 0)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        hr = self.lrelu(self.conv_before_upsample(x))
        hr = self.lrelu(self.upsample1(hr))
        hr = self.lrelu(self.upsample2(hr))
        hr = self.lrelu(self.conv_hr(hr))
        hr = self.conv_last(hr)

        return hr


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)
