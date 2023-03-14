# https://github.com/open-mmlab/mmediting/blob/master/mmedit/models/restorers/basic_restorer.py
# https://github.com/open-mmlab/mmediting/blob/master/mmedit/models/restorers/basicvsr.py
import numbers
import os.path as osp

import numpy as np
import mmcv
import torch
from mmcv.runner import auto_fp16

from mmedit.core import psnr, ssim, tensor2img
from mmedit.models.losses.pixelwise_loss import l1_loss
from mmedit.models.backbones.map_backbones.map_utils import resize

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


@MODELS.register_module()
class BasicDehazer(BaseModel):
    """Basic model for image dehazing.
    It must contain a generator that takes an image as inputs and outputs a
    dehazed image. It also has a pixel-wise loss for training.
    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.
    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'L1': l1_loss, 'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # loss
        self.pixel_loss = build_loss(pixel_loss)

        # for logging purpose
        self.log_shape_warning = False

    def init_weights(self, pretrained=None):
        """Init weights for models.
        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.
        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        """Training forward function.
        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        output = self.generator(lq)
        loss_pix = self.pixel_loss(output, gt)
        losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.
        Args:
            output (Tensor): Model output with shape (n, c, h, w) or (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w) or (n, t, c, h, w).
        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        convert_to = self.test_cfg.get('convert_to', None)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 4:  # (n, c, h, w)
                if metric in ('PSNR', 'SSIM'):
                    eval_result[metric] = self.allowed_metrics[metric](
                        tensor2img(output), tensor2img(gt), crop_border, convert_to=convert_to)
                elif metric in ('L1',):
                    eval_result[metric] = self.allowed_metrics[metric](
                        output.to(gt.device), gt, reduction='mean').item()
                else:
                    raise NotImplementedError
            elif output.ndim == 5:  # (n, t, c, h, w)
                avg = []
                for i in range(0, output.size(1)):
                    if metric in ('PSNR', 'SSIM'):
                        avg.append(self.allowed_metrics[metric](
                            tensor2img(output[:, i, :, :, :]), tensor2img(gt[:, i, :, :, :]),
                            crop_border, convert_to=convert_to))
                    elif metric in ('L1',):
                        avg.append(self.allowed_metrics[metric](
                            output[:, i, :, :, :].to(gt.device), gt[:, i, :, :, :], reduction='mean').item())
                eval_result[metric] = np.mean(avg)  # video-level average
            else:
                raise NotImplementedError
        return eval_result

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
            if lq.ndim == 4:
                output = self.generator(lq)
            else:
                n, t, c, h, w = lq.shape
                output = []
                for i in range(0, lq.size(1)):
                    torch.cuda.empty_cache()
                    output.append(self.generator(lq[:, i, :, :, :]))
                output = torch.cat(output).view(n, t, c, h, w)
            assert output.shape == lq.shape
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

    @staticmethod
    def _save_image(output, meta, save_path, iteration):
        file_ext = 'jpg'  # to save storage, change it to png if enough disk
        if output.ndim == 4:  # image
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.{file_ext}')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.{file_ext}')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)
        elif output.ndim == 5:  # video
            lq_path = meta[0]['lq_path']
            folder_name = osp.join(meta[0]['dataset'], meta[0]['folder'])
            for i in range(0, output.size(1)):
                filename = osp.splitext(osp.basename(lq_path[i]))[0]
                if isinstance(iteration, numbers.Number):
                    save_path_i = osp.join(save_path, folder_name,
                                           f'{filename}-{iteration + 1:06d}.{file_ext}')
                elif iteration is None:
                    save_path_i = osp.join(save_path, folder_name, f'{filename}.{file_ext}')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output[:, i, :, :, :]), save_path_i)

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output
