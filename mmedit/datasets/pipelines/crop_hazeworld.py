# https://github.com/open-mmlab/mmediting/blob/master/mmedit/datasets/pipelines/crop.py
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module()
class PairedRandomCropWithTransmission:
    """Paried random crop.

    It crops a pair of lq, gt, and transmission map images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", "gt", and "trans",
    added or modified keys are "lq", "gt", and "trans".

    Args:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        scale = results['scale']
        assert scale == 1  # for dehazing setting
        lq_patch_size = self.gt_patch_size // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]
        tm_is_list = isinstance(results['trans'], list)
        if not tm_is_list:
            results['trans'] = [results['trans']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape
        h_tm, w_tm, _ = results['trans'][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale or h_gt != h_tm or w_gt != w_tm:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                f'multiplication of LQ ({h_lq}, {w_lq}), TM ({w_tm}, {w_tm}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                f'({lq_patch_size}, {lq_patch_size}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_lq - lq_patch_size + 1)
        left = np.random.randint(w_lq - lq_patch_size + 1)
        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]

        # crop corresponding transmission map patch
        results['trans'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['trans']
        ]
        if not tm_is_list:
            results['trans'] = results['trans'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'
        return repr_str
