import torch
import torch.nn.functional as F
import einops

from ..registry import MODELS
from .basic_dehazer import BasicDehazer

from mmedit.models.backbones.map_backbones.map_utils import flow_warp_5d
from mmedit.models.losses.pixelwise_loss import l1_loss
from mmedit.models.backbones.map_backbones.map_utils import resize


def flow_loss(grid, ref, img0, img1, level):
    """
    see map_utils get_flow_from_grid
    """
    b, T, h, w, p = grid.shape
    assert p == 3, "Implementation for space-time flow warping"
    sf = 1. / 2 ** (level + 2)

    flow = (grid - ref).reshape(b * T, h, w, p)
    flow[:, :, :, 0] *= h
    flow[:, :, :, 1] *= w
    d = img0.shape[2]
    flow[:, :, :, 2] *= d
    assert flow.requires_grad

    # downsample and warp
    img0_lr = einops.rearrange(img0, 'bT c d h w -> (bT d) c h w')
    img0_lr = F.interpolate(img0_lr, scale_factor=sf, mode='bicubic')
    img0_lr = einops.rearrange(img0_lr, '(bT d) c h w -> bT c d h w', d=d)
    img0_lr_warp = flow_warp_5d(img0_lr, flow.unsqueeze(1))
    img0_lr_warp = img0_lr_warp.squeeze(2)
    img1_lr = F.interpolate(img1, scale_factor=sf, mode='bicubic')

    return l1_loss(img0_lr_warp, img1_lr)


@MODELS.register_module()
class MAP(BasicDehazer):
    """MAP model for video dehazing.

    Paper:
        Video Dehazing via a Multi-Range Temporal Alignment Network
        with Physical Prior, CVPR, 2023

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        num_kv_frames = generator.get('num_kv_frames', 1)
        self.num_kv_frames = sorted(num_kv_frames) if isinstance(num_kv_frames, (list, tuple)) else [num_kv_frames]

    @staticmethod
    def _get_output_from_dict(x):
        if isinstance(x, dict):
            return x['out']
        return x

    def forward_train(self, lq, gt):
        """Training forward function.
        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            Tensor: Output tensor.
        """
        assert lq.ndim == 5 and lq.shape[1] > 1, f"Video dehazing methods should have input t > 1 but get: {lq.shape}"
        losses = dict()
        output = self.generator(lq)
        loss_name = None
        if isinstance(output, dict):
            for key in output.keys():
                if key == 'out':
                    loss_key = self.pixel_loss(output[key], gt)
                elif key == 'img_01':
                    continue
                elif key in ('aux_j', 'aux_i'):
                    loss_name = key.replace('aux_', 'phy-')
                    loss_key, lambda_phy = 0., 0.2
                    num_stages = len(output[key])
                    gt_key = gt if key == 'aux_j' else output['img_01']
                    for s in range(num_stages):
                        loss_weight = lambda_phy / (2 ** s)  # be careful about the stage notation
                        loss_key += loss_weight * self.pixel_loss(output[key][s], gt_key)
                elif key.startswith('pos'):
                    # flow loss
                    assert len(output[key]) <= 4, \
                        f"pos should be less than or equal to 4 stages but get {len(output[key])}."
                    loss_name = 'flow'
                    loss_key, lambda_flow = 0., 0.04
                    num_stages = len(output[key])
                    for s in range(num_stages):
                        assert output[key][s].shape[-1] == 3
                        loss_weight = lambda_flow / 2 ** s  # be careful about the stage notation
                        b, T, c, h, w = gt.shape
                        num_groups = output[key][s].size(3)
                        for g in range(num_groups):
                            # pos[s] is in shape (b, T, nr, g, h, w, 3)
                            num_kv_frames = self.num_kv_frames  # assume kv_frames to be [1, 2, 3, etc...]
                            img0s = []
                            for step in range(max(num_kv_frames)):
                                indices = torch.clip(torch.arange(T) - (step + 1), 0).to(gt.device)
                                img0 = torch.index_select(gt, dim=1, index=indices)
                                img0 = img0.reshape(b * T, 3, h, w)
                                img0s.append(img0)
                            img0s = torch.stack(img0s, dim=2)
                            for r, kv_frames in enumerate(num_kv_frames):
                                grid = output[key][s][:, :, r, g, :, :, :].clone()
                                ref = output[key.replace('pos', 'ref')][s].clone()
                                assert not ref.requires_grad
                                img0 = img0s[:, :, :kv_frames].clone()
                                img1 = gt.clone().reshape(b * T, 3, h, w)
                                loss_key += loss_weight * flow_loss(grid, ref, img0, img1, s)
                elif key.startswith('ref'):
                    continue
                loss_name = loss_name or key
                losses[f'loss_{loss_name}'] = loss_key
            output = self._get_output_from_dict(output)
        else:
            loss_pix = self.pixel_loss(output, gt)
            losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.
        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            meta: meta information to help store results.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.
        Returns:
            dict: Output results.
        """
        torch.cuda.empty_cache()
        with torch.no_grad():
            assert lq.ndim == 5, "The MAP model is for video dehazing"
            output = self._get_output_from_dict(self.generator(lq))
        if lq.shape != gt.shape:
            # for REVIDE
            if not self.log_shape_warning:
                print(f"\n[Shape mismatch] lq: {lq.shape}, gt: {gt.shape}")
                self.log_shape_warning = True
            assert lq.shape[-2] == gt.shape[-2] // 2 and lq.shape[-1] == gt.shape[-1] // 2
            assert lq.ndim == 5
            outputs = []
            for i in range(output.size(1)):
                outputs.append(
                    resize(input=output[:, i, :, :, :],
                           size=gt.shape[-2:],
                           mode='bilinear', align_corners=False))
            output = torch.stack(outputs, dim=1)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # For HazeWorld
        assert len(meta) == 1
        if 'dataset' in meta[0]:
            results['dataset'] = meta[0]['dataset']
        if 'haze_beta' in meta[0]:
            results['haze_beta'] = meta[0]['haze_beta']

        # save image
        if save_image:
            self._save_image(output, meta, save_path, iteration)

        return results
