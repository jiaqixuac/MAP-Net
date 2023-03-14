import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .map_modules import ResidualBlocksWithInputConv
from .map_stda import STDABlock
from .map_utils import get_discrete_values, get_flow_from_grid, flow_warp_5d

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, load_checkpoint
from mmedit.models import builder
from mmedit.models.common import PixelShufflePack
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class PriorDecodeLayer(nn.Module):

    def __init__(self,
                 channels,
                 level,
                 upsample=True,
                 num_trans_bins=32,
                 memory_enhance=True):
        super().__init__()
        self.level = level

        self.upsample = upsample
        if upsample:
            self.up = PixelShufflePack(channels, channels, 2, 3)
            scale_head = [
                ConvModule(channels, channels, 3, padding=1)
                for _ in range(2)]
            self.scale_head = nn.Sequential(*scale_head)

        self.num_trans_bins = num_trans_bins
        self.head_t = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, num_trans_bins, 1, 1, 0))
        self.head_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, 1, 1, 1, 0), nn.Sigmoid())

        self.memory_enhance = memory_enhance

    def forward(self, feats):
        level = self.level
        #
        enc_skip = feats['spatial_p'][-1][level]
        if self.upsample:
            feat = enc_skip + self.up(feats['decode_p'][-1][level + 1])
            feat = self.scale_head(feat)
        else:
            feat = enc_skip
        feats['decode_p'][-1][level] = feat

        # MPG: estimate t and A
        logit_t = self.head_t(feat)
        prob_t = torch.softmax(logit_t, dim=1)  # b, num_trans_bins, h, w
        b, d, h, w = prob_t.shape
        values = get_discrete_values(self.num_trans_bins, 0., 1.) \
            .view(1, self.num_trans_bins, 1, 1).to(prob_t.device).repeat(b, 1, h, w)
        out_t = (prob_t * values).sum(dim=1, keepdim=True)
        out_a = self.head_a(feat)
        feats['stage_t'][level] = out_t
        feats['stage_a'][level] = out_a

        # MPG: memory enhance
        if self.memory_enhance:
            # prior token
            token_p = feat.unsqueeze(2) * prob_t.unsqueeze(1)  # b, c, d, h, w
            token_p = token_p.mean(dim=(-2, -1))  # b, c, d
            feats['token_p'][-1][level] = token_p

            # retrieve memory
            mem_p = [x[level] for x in feats['token_p']]
            mem_p = torch.stack(mem_p, dim=1)  # b, N, c, d
            b, N, c, d = mem_p.shape
            mem_p = mem_p.transpose(-2, -1).reshape(b, N * d, c).contiguous()  # b, Nd, c
            # read memory & attention
            _, _, h, w = feat.shape
            q_p = feat.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()  # b, hw, c
            scale = c ** -0.5
            attn = (q_p @ mem_p.transpose(-2, -1)) * scale  # b, hw, Nd
            attn = F.softmax(attn, dim=-1)
            en_p = attn @ mem_p  # b, hw, c
            en_p = en_p.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # b, c, h, w
            feats['enhance_p'][-1][level] = en_p
        else:
            feats['enhance_p'][-1][level] = feat

        return feats


class SceneDecodeLayer(nn.Module):

    def __init__(self,
                 channels,
                 level,
                 upsample=True,
                 prior_guide=True,
                 num_kv_frames=3,
                 align_depth=1,
                 num_heads=1,
                 kernel_size=3):
        super().__init__()
        self.level = level

        self.upsample = upsample
        if upsample:
            self.up = PixelShufflePack(channels, channels, 2, 3)

        self.prior_guide = prior_guide
        if prior_guide:
            self.guide_conv = ResidualBlocksWithInputConv(channels * 2, channels, 2)

        if not isinstance(num_kv_frames, (list, tuple)):
            num_kv_frames = list(range(1, num_kv_frames + 1))
        else:
            num_kv_frames = sorted(num_kv_frames)
        self.num_kv_frames = num_kv_frames
        self.align_layer = STDABlock(
            channels, num_heads=num_heads,
            align_depth=align_depth, dw_ks=kernel_size)

        self.aggre_beta = nn.Parameter(torch.ones(1))
        self.aggre_conv = ResidualBlocksWithInputConv(channels * 2, channels, 2)

        self.head_j = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, 3, 1, 1, 0))

    def forward(self, feats):
        level = self.level
        num_kv_frames = self.num_kv_frames
        #
        enc_skip = feats['spatial_j'][-1][level]
        if self.upsample:
            feat_j = enc_skip + self.up(feats['decode_j'][-1][level + 1])
        else:
            feat_j = enc_skip

        # MPG: prior guide
        if self.prior_guide:
            feat_p = feats['enhance_p'][-1][level]
            feat_j = self.guide_conv(torch.cat([feat_p, feat_j], dim=1))
        q_j = feat_j

        # MSR: multi-range
        # prepare features
        kv_j, kv_p = [], []
        nf = len(feats['decode_j']) - 1  # buffer frames: skip current timestep
        for step in range(1, max(num_kv_frames) + 1):
            kv_j.append(feats['decode_j'][max(nf - step, 0)][level] if nf > 0 else feat_j)
            kv_p.append(feats['enhance_p'][max(nf - step, 0)][level])
            # print(f"num_kv_frames: {num_kv_frames}, nf: {nf}, step: {step}, frame: {max(nf - step, 0)}")
        kv_j = torch.stack(kv_j, dim=1)  # b, nr, c, h, w
        kv_p = torch.stack(kv_p, dim=1)  # b, nr, c, h, w

        # multi-range alignment
        feats_jr, feats_pr = [], []
        grids = []
        for r, kv_frames in enumerate(num_kv_frames):
            # gradually refine flow
            if self.upsample:
                grid_r = feats['pos_j'][-1][level + 1][:, r]
                # print(f"level: {level}, grid_r: {grid_r.shape}")
                b, g, h, w, p = grid_r.shape
                assert p == 3, "Should be 5-D input sample"
                grid_r = einops.rearrange(grid_r, 'b g h w p -> (b g) p h w')
                # 'False' is more similar to generated ref points
                grid_r = F.interpolate(grid_r, scale_factor=2, mode='bilinear', align_corners=False)
                grid_r = einops.rearrange(grid_r, '(b g) p h w -> b g h w p', g=g)
            else:
                grid_r = None

            # align scene features
            kv_jr = kv_j[:, :kv_frames]
            feat_jr, grid_r, ref_r = self.align_layer(q_j, kv_jr, grid_r)
            feats_jr.append(feat_jr)
            grids.append(grid_r)

            # warp prior features
            b, g, h, w, p = grid_r.shape
            feat_pr = kv_p[:, :kv_frames]
            feat_pr = einops.rearrange(feat_pr, 'b nf (g c) h w -> (b g) c nf h w', g=g)
            _grid = einops.rearrange(grid_r, 'b g h w p -> (b g) h w p')
            flow = get_flow_from_grid(_grid, ref_r, d=kv_frames)
            feat_pr = flow_warp_5d(feat_pr, flow.unsqueeze(1))
            assert feat_pr.shape[2] == 1
            feats_pr.append(feat_pr.squeeze(2))
        feats['pos_j'][-1][level] = torch.stack(grids, dim=1)
        feats['ref_j'][-1][level] = ref_r

        # GMRA
        # prepare faetures
        feats_jr = torch.stack(feats_jr, dim=1)
        b, nr, c, h, w = feats_jr.shape
        scale = c ** -0.5

        # attn: j
        q_j = einops.rearrange(feat_j, 'b c h w -> (b h w) c').unsqueeze(1)  # bhw 1 c
        k_j = einops.rearrange(feats_jr, 'b nr c h w -> (b h w) nr c')  # bhw nr c
        attn_j = q_j @ k_j.transpose(-2, -1) * scale  # bhw 1 nr
        # attn: p
        q_p = feats['enhance_p'][-1][level]
        q_p = einops.rearrange(q_p, 'b c h w -> (b h w) c').unsqueeze(1)  # bhw 1 c
        k_p = torch.stack(feats_pr, dim=1)
        k_p = einops.rearrange(k_p, 'b nr c h w -> (b h w) nr c')  # bhw nr c
        attn_p = q_p @ k_p.transpose(-2, -1) * scale  # bhw 1 nr
        # attn
        attn = attn_j + self.aggre_beta * attn_p
        attn = F.softmax(attn, dim=-1)

        v = einops.rearrange(feats_jr, 'b nr c h w -> (b h w) nr c')  # bhw nr c
        feat = attn @ v
        assert feat.shape[1] == 1
        feat = einops.rearrange(feat.squeeze(1), '(b h w) c -> b c h w', h=h, w=w)
        feat = self.aggre_conv(torch.cat([feat, feat_j], dim=1))
        feats[f'decode_j'][-1][level] = feat

        # estimate J
        out_j = self.head_j(feat)
        feats['stage_j'][level] = out_j

        return feats


@BACKBONES.register_module()
class MAPNet(BaseModule):
    """MAP-Net.

    MAP-Net in "Video Dehazing via a Multi-Range Temporal Alignment Network with Physical Prior".
    """
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]

    def __init__(self,
                 backbone,
                 neck,
                 upsampler,
                 channels=32,
                 num_trans_bins=32,
                 align_depths=(1, 1, 1, 1),
                 num_kv_frames=(1, 2, 3),
                 ):
        super().__init__()

        self.backbone = builder.build_component(backbone)
        self.neck = builder.build_component(neck)
        self.upsampler = builder.build_component(upsampler)

        num_stages = len(align_depths)
        self.num_stages = num_stages

        # mpg
        self.num_trans_bins = num_trans_bins

        # msr: assume num_kv_frames is consecutive
        self.num_kv_frames = num_kv_frames

        # align & aggregate
        assert channels % 32 == 0
        num_heads = [channels // 32 for _ in range(num_stages)]
        kernel_sizes = [9, 7, 5, 3]

        self.prior_decoder_layers = nn.ModuleList()
        self.scene_decoder_layers = nn.ModuleList()

        guided_levels = (2, 3)  # memory consumption
        for s in range(num_stages):
            self.prior_decoder_layers.append(
                PriorDecodeLayer(
                    channels, s,
                    upsample=s < num_stages - 1, memory_enhance=s in guided_levels
                ))
            self.scene_decoder_layers.append(
                SceneDecodeLayer(
                    channels, s,
                    upsample=s < num_stages - 1, prior_guide=s in guided_levels,
                    num_kv_frames=num_kv_frames, align_depth=align_depths[s],
                    num_heads=num_heads[s], kernel_size=kernel_sizes[s]
                ))

        self.window_size = 32  # for padding
        rgb_mean = torch.Tensor(self.RGB_MEAN).reshape(1, 3, 1, 1)
        rgb_std = torch.Tensor(self.RGB_STD).reshape(1, 3, 1, 1)
        self.register_buffer('rgb_mean', rgb_mean)
        self.register_buffer('rgb_std', rgb_std)

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    def check_image_size(self, img):
        # https://github.com/JingyunLiang/SwinIR/blob/5aa89a7b275eeddc75cd7806378c89d23f298c48/main_test_swinir.py#L66
        # https://github.com/ZhendongWang6/Uformer/issues/32
        _, _, h, w = img.size()
        window_size = self.window_size
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        out = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return out

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def split_feat(self, feats, feat):
        feat_p, feat_j = [], []
        for s in range(self.num_stages):
            c = feat[s].shape[1]
            split_size_or_sections = c // 2
            x = torch.split(feat[s], split_size_or_sections, dim=1)
            feat_p.append(x[0])
            feat_j.append(x[1])
        feats['spatial_p'].append(feat_p)
        feats['spatial_j'].append(feat_j)
        return feats

    def decode(self, feats):
        # init
        keys = ['decode_p', 'token_p', 'enhance_p',
                'decode_j', 'pos_j', 'ref_j']
        for k in keys:
            feats[k].append([None] * self.num_stages)
        keys = ['stage_t', 'stage_a', 'stage_j']
        for k in keys:
            feats[k] = [None] * self.num_stages

        for s in range(self.num_stages - 1, -1, -1):
            feats = self.prior_decoder_layers[s](feats)
            feats = self.scene_decoder_layers[s](feats)

        return feats

    def forward(self, lqs):
        """
        Forward function

        Args:
            lqs (Tensor): Input hazy sequence with shape (n, t, c, h, w).

        Returns:
            out (Tensor): Output haze-free sequence with shape (n, t, c, h, w).
        """
        n, T, c, h, w = lqs.shape

        feats = {
            'spatial_p': [], 'decode_p': [], 'token_p': [], 'enhance_p': [],
            'spatial_j': [], 'decode_j': [], 'pos_j': [], 'ref_j': [],
            'stage_j': [], 'stage_t': [], 'stage_a': []
        }

        out_js = []
        img_01s = []
        aux_js, aux_is = [], []

        for i in range(0, T):
            # print(f"\ntime: {i}")
            img = self.check_image_size(lqs[:, i, :, :, :])
            img_01 = img * self.rgb_std + self.rgb_mean  # to the range of [0., 1.]
            img_01s.append(img_01)

            # encode
            feat = self.extract_feat(img)  # tuple of feats, (4s, 8s, 16s, ...)
            feats = self.split_feat(feats, feat)

            # decode
            feats = self.decode(feats)

            # get output
            feat_j = feats['decode_j'][-1][0]
            out = self.upsampler(feat_j)
            out = img_01 + out

            if self.training:
                assert h == out.shape[2] and w == out.shape[3]
            out_js.append(out[:, :, 0: h, 0: w].contiguous())

            # auxiliary output for the current timestep
            if self.training:
                aux_j, aux_i = [], []
                for s in range(self.num_stages):
                    tmp_j = F.interpolate(feats['stage_j'][s], size=img.shape[2:], mode='bilinear')
                    out_j = img_01 + tmp_j  # residue
                    tmp_t = F.interpolate(feats['stage_t'][s], size=img.shape[2:], mode='bilinear').clip(0, 1)
                    tmp_a = feats['stage_a'][s]
                    out_i = out_j * tmp_t + tmp_a * (1 - tmp_t)
                    aux_j.append(out_j[:, :, 0: h, 0: w])
                    aux_i.append(out_i[:, :, 0: h, 0: w])
                aux_js.append(aux_j)
                aux_is.append(aux_i)

            # memory management
            feats['spatial_j'].pop(0)
            feats['spatial_p'].pop(0)
            if len(feats['decode_j']) > max(self.num_kv_frames):
                feats['decode_j'].pop(0)
                feats['decode_p'].pop(0)
                feats['enhance_p'].pop(0)
                assert len(feats['decode_p']) == len(feats['decode_j'])
            if not self.training:
                feats['pos_j'].pop(0)
                feats['ref_j'].pop(0)

        out = dict(out=torch.stack(out_js, dim=1))  # output dict

        # auxiliary output for a sequence
        if self.training:
            pos, ref = [], []  # sampling locations
            for s in range(self.num_stages):
                pos.append(torch.stack([feats['pos_j'][i][s] for i in range(T)], dim=1))
                ref.append(torch.stack([feats['ref_j'][i][s] for i in range(T)], dim=1))
            out['pos'] = pos  # b, T, nr, g, h, w, 3
            out['ref'] = ref  # b, T, 1, h, w, 3
        if self.training:
            aux_j, aux_i = [], []  # Js, Is
            for s in range(self.num_stages):
                aux_j.append(torch.stack([aux_js[i][s] for i in range(T)], dim=1))
                aux_i.append(torch.stack([aux_is[i][s] for i in range(T)], dim=1))
            out['aux_j'] = aux_j
            out['aux_i'] = aux_i
            out['img_01'] = torch.stack(img_01s, dim=1)

        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        logger = get_root_logger()
        logger.info(f"Init weights: {pretrained}")
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif self.backbone.init_cfg is not None:
            self.backbone.init_weights()
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
