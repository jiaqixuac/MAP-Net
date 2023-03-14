# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F


# from https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/utils/shape_convert.py
def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def flow_warp_5d(x,
                 flow,
                 interpolation='bilinear',
                 padding_mode='zeros',
                 align_corners=True):
    """Modified from mmedit.models.utils.flow_warp
    Warp a stack of image or a feature map with flow.

    Args:
        x (Tensor): Tensor with size (n, c, d, h, w).
        flow (Tensor): Tensor with size (n, d, h, w, 3). The last dimension is
            a three-channel, denoting the width, height and z relative offsets.
            Note that the w, h values are not normalized to [-1, 1],
            and z values are normalized to [0, d].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[2:4]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[2:4]}) are not the same.')
    _, _, d, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    # TODO: assume reference point is 0.5, ...
    # TODO: make it consistent with stda layer
    grid_z = d * 0.5 * torch.ones((h, w))
    grid = torch.stack((grid_x, grid_y, grid_z), 2).type_as(x)  # (h, w, 3)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow_z = 2.0 * grid_flow[:, :, :, :, 2] / d - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y, grid_flow_z), dim=4)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def get_flow_from_grid(grid, ref, d=None):
    """
    convert sampling grids [0, 1] to pixels
    """
    _, h, w, _ = grid.shape
    flow = grid - ref  # denormalize to flow in pixel
    flow[:, :, :, 0] *= h
    flow[:, :, :, 1] *= w
    if grid.shape[-1] == 3:
        flow[:, :, :, 2] *= d

    return flow


def get_discrete_values(num_bins, start=0, end=1):
    """
    get discrete values given predefined num_bins and range (start, end)
    """
    values = torch.linspace(
        start, end, num_bins, requires_grad=False)
    return values


# https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/ops/wrappers.py
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
