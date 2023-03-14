import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.utils import to_2tuple

from mmedit.models.backbones.map_backbones.map_utils import (
    nchw_to_nlc, nlc_to_nchw,
    flow_warp_5d, get_flow_from_grid
)


class WindowMSA(nn.Module):
    """mmseg implementation
    https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/backbones/swin.py

    Window based multi-head self-attention (W-MSA) module with relative
    position bias.
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.kv_proj = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, q, kv, mask=None):
        """
        Args:
            q (tensor): input query features with shape of (num_windows*B, N, C)
            kv (tensor): input key/value features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, Nq, C = q.shape
        Nk = kv.shape[1]
        q = self.q_proj(q).reshape(B, Nq, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(kv).reshape(B, Nk, 2, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.repeat(1, 1, Nk // Nq).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, Nq,
                             Nk) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Nq, Nk)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):
    """mmseg implementation
    https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/backbones/swin.py

    Shifted Window Multihead Self-Attention Module.
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,)

    def forward(self, q, kv, hw_shape):
        B, L, C = q.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        assert kv.ndim == 4, f'kv should have shape (b, nf, l, c) but get {kv.shape}'
        nf = kv.shape[1]
        query = q.view(B, H, W, C)
        kv = kv.view(B * nf, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        kv = F.pad(kv, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_kv = torch.roll(kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            # mask_windows = self.window_partition(img_mask)
            mask_windows = einops.rearrange(img_mask, 'b (nh w1) (nw w2) c -> (b nh nw) w1 w2 c',
                                            w1=self.window_size, w2=self.window_size).contiguous()
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.repeat(1, 1, nf)
        else:
            shifted_query = query
            shifted_kv = kv
            attn_mask = None

        # nW*B, window_size, window_size, C
        # query_windows = self.window_partition(shifted_query)
        query_windows = einops.rearrange(shifted_query, 'b (nh w1) (nw w2) c -> (b nh nw) w1 w2 c',
                                         w1=self.window_size, w2=self.window_size).contiguous()
        # nW*B, nf, window_size, window_size, C
        kv_windows = einops.rearrange(shifted_kv, '(b nf) (nh w1) (nw w2) c -> (b nh nw) nf w1 w2 c',
                                      nf=nf, w1=self.window_size, w2=self.window_size).contiguous()
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)
        # nW*B, nf*window_size*window_size, C
        kv_windows = kv_windows.view(-1, nf * self.window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, kv_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # B H' W' C
        # shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        shifted_x = einops.rearrange(attn_windows, '(b nh nw) w1 w2 c -> b (nh w1) (nw w2) c',
                                     nh=H_pad // self.window_size, nw=W_pad // self.window_size).contiguous()
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, None, None

    def window_reverse(self, windows, H, W):
        """Not used in our implementation
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """Not used in our implementation
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class LayerNormProxy(nn.Module):
    """DAT implementation
    https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class STDAWindowMSA(nn.Module):
    """Modified from DAT (https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py)
    For spatial-temporal deformable alignment.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 num_groups=1,
                 offset_range_factor=0,
                 dw_ks=7):
        super().__init__()

        self.attn = ShiftWindowMSA(embed_dims, num_heads, window_size, shift_size,
                                   qkv_bias, qk_scale, attn_drop_rate, proj_drop_rate)

        assert num_groups <= num_heads, "The number of heads should be larger than the number of groups"
        self.num_groups = num_groups
        self.num_group_channels = embed_dims // self.num_groups

        self.offset_range_factor = offset_range_factor
        ch = 3  # 3 for space-time flow
        channels = self.num_group_channels * 2 + ch
        self.conv_offset = nn.Sequential(
            nn.Conv2d(channels, channels,
                      dw_ks, 1, dw_ks // 2, groups=channels),  # flow grid as input
            LayerNormProxy(channels),
            nn.GELU(),
            nn.Conv2d(channels, ch, 1, 1, 0, bias=False))

    def init_weights(self):
        # be careful
        self._is_init = True

    def forward(self, q, kv, hw_shape, deform_inputs, grid):
        # q: b, l, c
        # kv: b, nf, l, c
        B, L, C = q.shape
        H, W = hw_shape
        assert L == H * W, f"input feature has wrong size, q: {q.shape}, hw_shape: {hw_shape}"
        assert kv.ndim == 4, f"kv should have shape (b, nf, l, c) but get {kv.shape}"
        num_kv_frames = kv.shape[1]

        # prepare variables for sampling
        reference_points = deform_inputs[0]
        reference_points = reference_points.reshape(1, H, W, 2)  # (1, H, W, 2)
        spatial_shapes = deform_inputs[1]
        # TODO: set default reference points as 0.5 for z
        # TODO: make it consistent with flow_warp_5d
        reference_points = torch.cat(
            [reference_points, 0.5 * torch.ones((1, H, W, 1)).type_as(reference_points)], dim=-1)  # (1, H, W, 3)

        # b * g, c, h, w
        q_hw = einops.rearrange(q, 'b (h w) (g c) -> (b g) c h w', h=H, g=self.num_groups)
        k_hw = einops.rearrange(kv, 'b nf (h w) (g c) -> (b g) c nf h w', h=H, g=self.num_groups)

        # convert grid to flow pixels, grid should be in the normalized range of [0, 1]
        grid_before = einops.rearrange(grid, 'b g h w p -> (b g) h w p') if grid is not None \
            else reference_points.expand(B * self.num_groups, -1, -1, -1)  # d should be 1
        flow = get_flow_from_grid(grid_before, reference_points, d=num_kv_frames)

        # TODO: warp kv towards q using flow, \tilde{J}_r^a
        # be careful, deformable detr uses different implementation compared to flow_warp
        # reference points, normalization, align_corners
        q_off = q_hw
        k_off = flow_warp_5d(k_hw, flow.unsqueeze(1))  # flow: (b h w 3) -> (b 1 h w 3)
        assert k_off.shape[2] == 1
        k_off = k_off.squeeze(2)  # b c 1 h w -> b c h w
        g_off = flow.permute(0, 3, 1, 2).contiguous()  # b (* nf )* g 3 h w

        # predict flow offset residual
        x_off = torch.cat([q_off, k_off, g_off], dim=1)
        sampling_offsets = self.conv_offset(x_off)  # b (* nf )* g 3 h w
        sampling_offsets = einops.rearrange(sampling_offsets, 'b p h w -> b h w p')  # in pixel, b (* nf )* g h w 3

        if self.offset_range_factor > 0:
            raise NotImplementedError

        # update sampling locations, O_{r \rightarrow i}
        offset_normalizer = torch.stack(
            # dummy spatial_shapes[..., 2] as 1
            [spatial_shapes[..., 1], spatial_shapes[..., 0], num_kv_frames * spatial_shapes[..., 2]], -1)
        if num_kv_frames == 1:
            # 2-D case
            pass
            # sampling_offsets[:, :, :, 2].fill_(0)
        # b * g, h, w, 3
        sampling_locations = grid_before + sampling_offsets / offset_normalizer[None, :, None, :]

        # aligned features, J_r^a
        flow = get_flow_from_grid(sampling_locations, reference_points, d=num_kv_frames)
        x_sampled = flow_warp_5d(k_hw, flow.unsqueeze(1))
        assert x_sampled.shape[2] == 1
        x_sampled = einops.rearrange(x_sampled, '(b g) c d h w -> b d (h w) (g c)', g=self.num_groups)

        # W-MSA
        x, _, _ = self.attn(q, x_sampled, hw_shape)

        sampling_locations = einops.rearrange(sampling_locations, '(b g) h w p -> b g h w p',
                                              g=self.num_groups)

        return x, sampling_locations, reference_points.reshape(1, H, W, 3)


class MLPWithConv(nn.Module):

    def __init__(self, embed_dims, expansion=2,
                 ffn_drop=0.):
        super().__init__()

        dim1 = embed_dims
        dim2 = embed_dims * expansion

        self.linear1 = nn.Conv2d(dim1, dim2, 1, 1, 0)
        self.dwc = nn.Conv2d(dim2, dim2, 3, 1, 1, groups=dim2)
        self.linear2 = nn.Conv2d(dim2, dim1, 1, 1, 0)

        self.act = nn.GELU()
        self.drop = nn.Dropout(ffn_drop, inplace=True)

    def forward(self, x, hw_shape):
        x = nlc_to_nchw(x, hw_shape)
        x = self.drop(self.act(self.dwc(self.linear1(x))))
        x = self.drop(self.linear2(x))
        x = nchw_to_nlc(x)

        return x


class STDALayer(nn.Module):
    """cross haze attention + cross attention (w/dat/detr) + FFN"""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_cfg=dict(type='LN'),
                 window_size=7,  # be consistent with the encoder
                 shift=False,
                 offset_range_factor=0,
                 dw_ks=7):
        super().__init__()

        self.cross_attn = STDAWindowMSA(
            embed_dims=embed_dims, num_heads=num_heads, window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate,
            offset_range_factor=offset_range_factor, dw_ks=dw_ks)

        self.cross_ffn = MLPWithConv(
            embed_dims=embed_dims,
            ffn_drop=drop_rate)

        self.norm_ca = build_norm_layer(norm_cfg, embed_dims)[1]
        self.norm_cf = build_norm_layer(norm_cfg, embed_dims)[1]

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, q, kv, hw_shape, grid):
        # ref: [B, L, C], supp: [B, NF, L, C]

        # cross attention
        deform_inputs = self._get_deform_inputs(q, hw_shape)
        x, grid, ref = self.cross_attn(q, kv, hw_shape, deform_inputs, grid)
        x = q + self.drop_path(self.norm_ca(x))  # post-norm

        # cross ffn
        x = x + self.drop_path(self.norm_cf(self.cross_ffn(x, hw_shape)))

        return x, grid, ref

    @staticmethod
    def _get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def _get_deform_inputs(self, x, hw_shape):
        h, w = hw_shape
        reference_points = self._get_reference_points(
            [(h, w)], device=x.device)
        spatial_shapes = torch.as_tensor(
            [(h, w, 1)], dtype=torch.long, device=x.device)
        return [reference_points, spatial_shapes]


class STDABlock(nn.Module):
    """Spatial temporal transformer for video restoration."""

    def __init__(self,
                 embed_dims=32,
                 num_heads=2,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 dpr=None,
                 align_depth=1,
                 offset_range_factor=0,
                 dw_ks=7):
        super().__init__()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, align_depth)
        ] if dpr is None else dpr  # stochastic depth decay rule

        self.layers = nn.ModuleList([
            STDALayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],  # TODO: dpr
                qkv_bias=qkv_bias,
                norm_cfg=dict(type='LN'),
                shift=i % 2 == 1,
                offset_range_factor=offset_range_factor,
                dw_ks=dw_ks,
            ) for i in range(align_depth)])

        self.apply(self.init_weights)
        self.apply(self.init_deform_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # dat init
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_deform_weights(self, m):
        if isinstance(m, STDAWindowMSA):
            m.init_weights()

    def forward(self, q, kv, grid=None):
        """
        Args:
            q (Tensor): Input feature with shape (n, embed_dims, h, w), t.
            kv (Tensor): Input feature with shape (n, num_kv_frames, embed_dims, h, w), [t-1, ...].
            grid (Tensor): Sampling grids from previous layers.
        Returns:
            Tensors:
                Output feature with shape (n, out_channels, h, w)
                Sampling grids with shape (b, g, h, w, 3)
                Referent points with shape (1, h, w, 3)
        """
        b, c, h, w = q.shape
        if kv.ndim == 4:
            # assume one or more frames for kv
            kv = kv.unsqueeze(1)

        q = einops.rearrange(q, 'b c h w -> b (h w) c')
        kv = einops.rearrange(kv, 'b nf c h w -> b nf (h w) c')
        hw_shape = (h, w)

        for _, blk in enumerate(self.layers):
            q, grid, ref = blk(q, kv, hw_shape, grid)

        # reshape back
        x = einops.rearrange(q, 'b (h w) c -> b c h w', h=h, w=w)

        return x, grid, ref
